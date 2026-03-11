#!/usr/bin/env python3
"""
Animation keyframe agent for rigged models.

Design reference:
  LLMR (arXiv:2309.12276) inspired the agentized pipeline:
  scene analysis -> task planning -> building -> inspection -> memory.

This module supports two backends:
  1) llm (default): planner decomposes a user animation prompt into motion steps,
     then predicts per-bone keyframes for a selected step.
  2) geometry: estimates keyframes from explicit joint positions.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class JointDef:
    name: str
    parent: int
    rest_position: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RigModel:
    name: str
    joints: Tuple[JointDef, ...]

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> List[str]:
        return [j.name for j in self.joints]

    @property
    def parents(self) -> np.ndarray:
        return np.array([j.parent for j in self.joints], dtype=np.int64)

    def children_map(self) -> Dict[int, List[int]]:
        children: Dict[int, List[int]] = {i: [] for i in range(self.joint_count)}
        for idx, joint in enumerate(self.joints):
            if joint.parent >= 0:
                children[joint.parent].append(idx)
        return children

    def rest_positions_or_none(self) -> Optional[np.ndarray]:
        if any(j.rest_position is None for j in self.joints):
            return None
        return np.stack([j.rest_position for j in self.joints], axis=0).astype(np.float64)

    def root_index(self) -> int:
        parents = self.parents
        roots = np.where(parents < 0)[0]
        return int(roots[0]) if len(roots) else 0


@dataclass(frozen=True)
class BoneKeyframe:
    step: int
    time_sec: float
    bone_index: int
    bone_name: str
    parent_index: int
    parent_name: Optional[str]
    local_rotation_xyzw: Tuple[float, float, float, float]
    local_translation_xyz: Tuple[float, float, float]


@dataclass(frozen=True)
class AgentContext:
    rig: RigModel
    model_image_path: Optional[str]
    motion: Optional[np.ndarray]
    rest_positions: Optional[np.ndarray]
    motion_text: Optional[str]


@dataclass(frozen=True)
class PlannedTask:
    mode: str
    step: int
    fps: float
    smoothing: float
    animation_prompt: Optional[str]
    plan_steps: Tuple[str, ...]
    motion_text: Optional[str]


@dataclass(frozen=True)
class LLMClientConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout_sec: float


class SceneAnalyzer:
    def analyze(
        self,
        rig: RigModel,
        mode: str,
        model_image_path: Optional[Path] = None,
        motion: Optional[np.ndarray] = None,
        motion_text: Optional[str] = None,
    ) -> AgentContext:
        if mode == "geometry":
            if motion is None:
                raise ValueError("geometry mode requires --motion.")
            if motion.ndim != 3 or motion.shape[-1] != 3:
                raise ValueError("Motion must have shape (num_frames, num_joints, 3).")
            if motion.shape[1] != rig.joint_count:
                raise ValueError(
                    f"Joint count mismatch: motion has {motion.shape[1]} joints but rig has {rig.joint_count}."
                )
            rest_positions = rig.rest_positions_or_none()
            if rest_positions is None:
                rest_positions = motion[0].astype(np.float64)
        else:
            if not motion_text or not motion_text.strip():
                raise ValueError("llm mode requires non-empty --motion-text or --motion-text-file.")
            rest_positions = rig.rest_positions_or_none()

        image_str = str(model_image_path) if model_image_path is not None else None
        return AgentContext(
            rig=rig,
            model_image_path=image_str,
            motion=motion.astype(np.float64) if motion is not None else None,
            rest_positions=rest_positions,
            motion_text=motion_text.strip() if motion_text is not None else None,
        )


class TaskPlanner:
    def __init__(
        self,
        llm_client: Optional["OpenAICompatibleChatClient"] = None,
        max_plan_steps: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.max_plan_steps = int(max(1, max_plan_steps))

    def plan(
        self,
        mode: str,
        step: int,
        fps: float,
        smoothing: float,
        animation_prompt: Optional[str],
        rig: Optional[RigModel] = None,
        model_image_path: Optional[str] = None,
        disable_planner: bool = False,
    ) -> PlannedTask:
        if mode not in {"llm", "geometry"}:
            raise ValueError("Mode must be one of: llm, geometry.")
        if step < 0:
            raise ValueError("Step must be >= 0.")
        if fps <= 0:
            raise ValueError("FPS must be > 0.")
        if smoothing < 0.0 or smoothing >= 1.0:
            raise ValueError("Smoothing must be in [0.0, 1.0).")

        motion_text = animation_prompt
        plan_steps: Tuple[str, ...] = tuple()

        if mode == "llm":
            if animation_prompt is None or not animation_prompt.strip():
                raise ValueError("llm mode requires a non-empty animation prompt.")
            if disable_planner:
                plan_steps = (animation_prompt.strip(),)
            else:
                plan_steps = tuple(
                    self._plan_animation_steps(
                        animation_prompt=animation_prompt.strip(),
                        rig=rig,
                        model_image_path=model_image_path,
                    )
                )
            if len(plan_steps) == 0:
                plan_steps = (animation_prompt.strip(),)
            if step >= len(plan_steps):
                raise ValueError(
                    f"Plan step {step} is out of range for plan with {len(plan_steps)} steps."
                )
            motion_text = plan_steps[step]

        return PlannedTask(
            mode=mode,
            step=int(step),
            fps=float(fps),
            smoothing=float(smoothing),
            animation_prompt=animation_prompt.strip() if animation_prompt is not None else None,
            plan_steps=plan_steps,
            motion_text=motion_text,
        )

    def _plan_animation_steps(
        self,
        animation_prompt: str,
        rig: Optional[RigModel],
        model_image_path: Optional[str],
    ) -> List[str]:
        if self.llm_client is None:
            return [animation_prompt]

        rig_text = "unknown rig"
        if rig is not None:
            rig_text = (
                f"rig_name={rig.name}, joint_count={rig.joint_count}, "
                f"root={rig.root_index()}, joints={', '.join(rig.names)}"
            )

        system_prompt = (
            "You are a planning agent for 3D character animation in Blender.\n"
            "Convert a user animation request into a short ordered sequence of motion steps.\n"
            "Each step must be a single concise sentence describing pose/action at that stage.\n"
            "Output JSON only with this schema:\n"
            "{ \"steps\": [ {\"index\": 0, \"motion\": \"...\"} ] }\n"
            "Rules:\n"
            "1) Keep steps physically coherent and temporally ordered.\n"
            "2) Prefer 3 to 8 steps.\n"
            "3) Each step must be directly usable by an animation keyframe generator.\n"
            "4) Do not include camera/lights/audio.\n"
            "5) No extra keys outside 'steps'."
        )
        user_text = (
            "Create an animation plan for this request.\n"
            f"User request: {animation_prompt}\n"
            f"Rig context: {rig_text}\n"
            "Return steps for character motion only."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            build_user_message(user_text, model_image_path),
        ]
        llm_text = self.llm_client.create_chat_completion(messages=messages, response_json=True)
        return parse_motion_plan_json(llm_text, self.max_plan_steps, fallback=animation_prompt)


class AgentMemory:
    """Stores generated step keyframes for temporal coherence."""

    def __init__(self) -> None:
        self._cache: Dict[int, List[BoneKeyframe]] = {}

    def put(self, step: int, keyframes: List[BoneKeyframe]) -> None:
        self._cache[step] = keyframes

    def get(self, step: int) -> Optional[List[BoneKeyframe]]:
        return self._cache.get(step)

class GeometryKeyframeBuilder:
    def build(self, context: AgentContext, task: PlannedTask) -> List[BoneKeyframe]:
        if context.motion is None or context.rest_positions is None:
            raise ValueError("Geometry backend requires motion and rest positions.")

        step = min(task.step, context.motion.shape[0] - 1)
        pose = context.motion[step]
        parents = context.rig.parents
        children = context.rig.children_map()
        root_idx = context.rig.root_index()

        world_quats = estimate_world_rotations(context.rest_positions, pose, children, parents)
        local_quats = world_to_local_quaternions(world_quats, parents)
        root_translation = pose[root_idx] - context.rest_positions[root_idx]
        time_sec = step / task.fps

        keyframes: List[BoneKeyframe] = []
        for idx, joint in enumerate(context.rig.joints):
            parent_name = context.rig.joints[joint.parent].name if joint.parent >= 0 else None
            translation = root_translation if idx == root_idx else np.zeros(3, dtype=np.float64)
            keyframes.append(
                BoneKeyframe(
                    step=step,
                    time_sec=time_sec,
                    bone_index=idx,
                    bone_name=joint.name,
                    parent_index=joint.parent,
                    parent_name=parent_name,
                    local_rotation_xyzw=tuple(float(v) for v in local_quats[idx]),
                    local_translation_xyz=tuple(float(v) for v in translation),
                )
            )
        return keyframes


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config

    def create_chat_completion(
        self, messages: List[dict], response_json: bool = True
    ) -> str:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload: Dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM network error: {exc}") from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError("LLM response is missing 'choices'.")
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts)
        raise RuntimeError("LLM response has unsupported content format.")


class LLMKeyframeBuilder:
    def __init__(self, client: OpenAICompatibleChatClient) -> None:
        self.client = client

    def build(
        self,
        context: AgentContext,
        task: PlannedTask,
        previous_keyframes: Optional[List[BoneKeyframe]] = None,
    ) -> List[BoneKeyframe]:
        if task.motion_text is None:
            raise ValueError("LLM backend requires motion text.")

        system_prompt = (
            "You are an animation keyframe generator. "
            "Predict one-step local keyframes for every bone in a rigged model.\n"
            "Rules:\n"
            "1) Output JSON only.\n"
            "2) Return all bones exactly once.\n"
            "3) Use normalized quaternions in [x,y,z,w].\n"
            "4) Keep motion plausible and coherent with hierarchy.\n"
            "5) Use local_translation_xyz only for the root bone; set others to [0,0,0]."
        )

        hierarchy_text = render_joint_hierarchy(context.rig)
        root_idx = context.rig.root_index()
        rest_pose_text = render_rest_pose(context.rig)
        previous_text = render_previous_keyframes(previous_keyframes)
        schema = (
            "{\n"
            '  "keyframes": [\n'
            "    {\n"
            '      "bone_index": 0,\n'
            '      "local_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],\n'
            '      "local_translation_xyz": [0.0, 0.0, 0.0]\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        user_text = (
            "Predict per-bone keyframes for this step.\n"
            f"Step index: {task.step}\n"
            f"Step time (sec): {task.step / task.fps:.6f}\n"
            f"FPS: {task.fps}\n"
            f"Motion description: {task.motion_text}\n"
            f"Root bone index: {root_idx}\n"
            "Joint hierarchy (index:name,parent_index):\n"
            f"{hierarchy_text}\n"
            f"{rest_pose_text}\n"
            f"{previous_text}\n"
            "Return JSON with this schema:\n"
            f"{schema}\n"
            "keyframes length must be exactly equal to joint count."
        )

        user_message = build_user_message(user_text, context.model_image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]
        llm_output = self.client.create_chat_completion(messages=messages, response_json=True)
        return parse_keyframes_from_llm_json(llm_output, context.rig, task.step, task.fps)


def build_user_message(text: str, model_image_path: Optional[str]) -> dict:
    if model_image_path is None:
        return {"role": "user", "content": text}

    image_path = Path(model_image_path)
    if not image_path.exists():
        return {"role": "user", "content": text}
    try:
        data_url = image_to_data_url(image_path)
    except OSError:
        return {"role": "user", "content": text}

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


class Inspector:
    """Validates and repairs minor numerical issues in generated keyframes."""

    def inspect(self, keyframes: List[BoneKeyframe], rig: RigModel) -> List[BoneKeyframe]:
        if len(keyframes) != rig.joint_count:
            raise ValueError(f"Expected {rig.joint_count} keyframes, got {len(keyframes)}.")

        sorted_kfs = sorted(keyframes, key=lambda k: k.bone_index)
        for expected_idx, kf in enumerate(sorted_kfs):
            if kf.bone_index != expected_idx:
                raise ValueError("Keyframes must include every bone index exactly once.")

        repaired: List[BoneKeyframe] = []
        root_idx = rig.root_index()
        for kf in sorted_kfs:
            quat = np.array(kf.local_rotation_xyzw, dtype=np.float64)
            if not np.isfinite(quat).all():
                quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            quat = normalize_quaternion(quat)

            trn = np.array(kf.local_translation_xyz, dtype=np.float64)
            if not np.isfinite(trn).all():
                trn = np.zeros(3, dtype=np.float64)
            if kf.bone_index != root_idx:
                trn = np.zeros(3, dtype=np.float64)

            repaired.append(
                BoneKeyframe(
                    step=kf.step,
                    time_sec=kf.time_sec,
                    bone_index=kf.bone_index,
                    bone_name=kf.bone_name,
                    parent_index=kf.parent_index,
                    parent_name=kf.parent_name,
                    local_rotation_xyzw=tuple(float(v) for v in quat),
                    local_translation_xyz=tuple(float(v) for v in trn),
                )
            )
        return repaired


class AnimationKeyframeAgent:
    def __init__(self, llm_client: Optional[OpenAICompatibleChatClient] = None) -> None:
        self.scene_analyzer = SceneAnalyzer()
        self.task_planner = TaskPlanner(llm_client=llm_client)
        self.geometry_builder = GeometryKeyframeBuilder()
        self.llm_builder = LLMKeyframeBuilder(llm_client) if llm_client is not None else None
        self.inspector = Inspector()
        self.memory = AgentMemory()
        self.last_task: Optional[PlannedTask] = None

    def run(
        self,
        rig: RigModel,
        step: int,
        fps: float = 30.0,
        smoothing: float = 0.0,
        mode: str = "llm",
        motion_text: Optional[str] = None,
        animation_prompt: Optional[str] = None,
        motion: Optional[np.ndarray] = None,
        model_image_path: Optional[Path] = None,
        disable_planner: bool = False,
    ) -> List[BoneKeyframe]:
        prompt = animation_prompt if animation_prompt is not None else motion_text
        model_image_str = str(model_image_path) if model_image_path is not None else None

        task = self.task_planner.plan(
            mode=mode,
            step=step,
            fps=fps,
            smoothing=smoothing,
            animation_prompt=prompt,
            rig=rig,
            model_image_path=model_image_str,
            disable_planner=disable_planner,
        )
        context = self.scene_analyzer.analyze(
            rig=rig,
            mode=mode,
            model_image_path=model_image_path,
            motion=motion,
            motion_text=task.motion_text,
        )
        self.last_task = task

        if task.mode == "llm":
            if self.llm_builder is None:
                raise ValueError("LLM backend is not configured.")
            prev = self.memory.get(step - 1)
            raw_keyframes = self.llm_builder.build(context=context, task=task, previous_keyframes=prev)
        else:
            raw_keyframes = self.geometry_builder.build(context=context, task=task)

        checked = self.inspector.inspect(raw_keyframes, rig=rig)

        if smoothing > 0.0 and step > 0:
            prev = self.memory.get(step - 1)
            if prev is not None and len(prev) == len(checked):
                checked = smooth_keyframes(prev, checked, smoothing)

        self.memory.put(step, checked)
        return checked


def smooth_keyframes(
    prev: Sequence[BoneKeyframe], cur: Sequence[BoneKeyframe], smoothing: float
) -> List[BoneKeyframe]:
    alpha = 1.0 - smoothing
    blended: List[BoneKeyframe] = []
    for a, b in zip(prev, cur):
        qa = np.array(a.local_rotation_xyzw, dtype=np.float64)
        qb = np.array(b.local_rotation_xyzw, dtype=np.float64)
        qt = quaternion_slerp(qa, qb, alpha)
        ta = np.array(a.local_translation_xyz, dtype=np.float64)
        tb = np.array(b.local_translation_xyz, dtype=np.float64)
        tt = (1.0 - alpha) * ta + alpha * tb
        blended.append(
            BoneKeyframe(
                step=b.step,
                time_sec=b.time_sec,
                bone_index=b.bone_index,
                bone_name=b.bone_name,
                parent_index=b.parent_index,
                parent_name=b.parent_name,
                local_rotation_xyzw=tuple(float(v) for v in qt),
                local_translation_xyz=tuple(float(v) for v in tt),
            )
        )
    return blended


def parse_keyframes_from_llm_json(
    llm_text: str, rig: RigModel, step: int, fps: float
) -> List[BoneKeyframe]:
    obj = extract_json_object(llm_text)
    keyframes_raw = obj.get("keyframes")
    if not isinstance(keyframes_raw, list):
        raise ValueError("LLM output must contain a 'keyframes' list.")

    by_name = {j.name: i for i, j in enumerate(rig.joints)}
    resolved: Dict[int, BoneKeyframe] = {}
    time_sec = step / fps

    for item in keyframes_raw:
        if not isinstance(item, dict):
            continue
        idx = resolve_bone_index(item, by_name)
        if idx is None or idx < 0 or idx >= rig.joint_count:
            continue
        quat = coerce_float_tuple(item.get("local_rotation_xyzw"), 4, (0.0, 0.0, 0.0, 1.0))
        trn = coerce_float_tuple(item.get("local_translation_xyz"), 3, (0.0, 0.0, 0.0))
        parent = rig.joints[idx].parent
        parent_name = rig.joints[parent].name if parent >= 0 else None
        resolved[idx] = BoneKeyframe(
            step=step,
            time_sec=time_sec,
            bone_index=idx,
            bone_name=rig.joints[idx].name,
            parent_index=parent,
            parent_name=parent_name,
            local_rotation_xyzw=quat,
            local_translation_xyz=trn,
        )

    root_idx = rig.root_index()
    for idx in range(rig.joint_count):
        if idx in resolved:
            continue
        parent = rig.joints[idx].parent
        parent_name = rig.joints[parent].name if parent >= 0 else None
        resolved[idx] = BoneKeyframe(
            step=step,
            time_sec=time_sec,
            bone_index=idx,
            bone_name=rig.joints[idx].name,
            parent_index=parent,
            parent_name=parent_name,
            local_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            local_translation_xyz=(0.0, 0.0, 0.0) if idx != root_idx else (0.0, 0.0, 0.0),
        )

    return [resolved[idx] for idx in sorted(resolved)]


def resolve_bone_index(item: dict, by_name: Dict[str, int]) -> Optional[int]:
    raw_index = item.get("bone_index")
    if isinstance(raw_index, int):
        return raw_index
    raw_name = item.get("bone_name")
    if isinstance(raw_name, str):
        return by_name.get(raw_name)
    return None


def extract_json_object(text: str) -> dict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        obj = json.loads(clean)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        snippet = clean[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not parse JSON object from LLM output.")


def parse_motion_plan_json(llm_text: str, max_steps: int, fallback: str) -> List[str]:
    obj = extract_json_object(llm_text)
    steps_raw = obj.get("steps")
    if not isinstance(steps_raw, list):
        return [fallback]

    out: List[str] = []
    for item in steps_raw:
        motion = None
        if isinstance(item, dict):
            for key in ("motion", "description", "text", "instruction"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    motion = value.strip()
                    break
        elif isinstance(item, str) and item.strip():
            motion = item.strip()
        if motion:
            out.append(motion)
        if len(out) >= max_steps:
            break

    return out if out else [fallback]


def coerce_float_tuple(value: object, length: int, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        return fallback
    out: List[float] = []
    for x in value:
        if isinstance(x, (int, float)):
            out.append(float(x))
        else:
            return fallback
    return tuple(out)


def render_joint_hierarchy(rig: RigModel) -> str:
    lines = [f"{idx}:{joint.name},{joint.parent}" for idx, joint in enumerate(rig.joints)]
    return "\n".join(lines)


def render_rest_pose(rig: RigModel) -> str:
    rest = rig.rest_positions_or_none()
    if rest is None:
        return "Rest positions: unavailable in rig file."
    lines = []
    for idx, p in enumerate(rest):
        lines.append(f"{idx}: [{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]")
    return "Rest positions (xyz by joint index):\n" + "\n".join(lines)


def render_previous_keyframes(previous: Optional[List[BoneKeyframe]]) -> str:
    if not previous:
        return "Previous step keyframes: none."
    lines = []
    for kf in previous:
        q = kf.local_rotation_xyzw
        t = kf.local_translation_xyz
        lines.append(
            f"{kf.bone_index}: q=[{q[0]:.5f},{q[1]:.5f},{q[2]:.5f},{q[3]:.5f}], "
            f"t=[{t[0]:.5f},{t[1]:.5f},{t[2]:.5f}]"
        )
    return "Previous step keyframes:\n" + "\n".join(lines)


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"

def load_motion(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        num_frames = arr.shape[0]
        num_joints = arr.shape[1] // 3
        return arr.reshape(num_frames, num_joints, 3).astype(np.float64)
    raise ValueError(f"Unsupported motion shape: {arr.shape}")


def load_rig(path: Path) -> RigModel:
    if path.suffix.lower() == ".gltf":
        return load_rig_from_gltf(path)
    return load_rig_from_json(path)


def load_rig_from_json(path: Path) -> RigModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    joints_data = data.get("joints")
    if not isinstance(joints_data, list) or len(joints_data) == 0:
        raise ValueError("Rig JSON must contain a non-empty 'joints' list.")

    names = [str(j.get("name", f"joint_{i:02d}")) for i, j in enumerate(joints_data)]
    name_to_idx = {n: i for i, n in enumerate(names)}
    joints: List[JointDef] = []

    for i, item in enumerate(joints_data):
        raw_parent = item.get("parent", None)
        if raw_parent is None:
            parent = -1
        elif isinstance(raw_parent, int):
            parent = raw_parent
        elif isinstance(raw_parent, str):
            if raw_parent not in name_to_idx:
                raise ValueError(f"Unknown parent name '{raw_parent}' for joint {names[i]}.")
            parent = name_to_idx[raw_parent]
        else:
            raise ValueError(f"Invalid parent value for joint {names[i]}: {raw_parent!r}")

        if parent >= len(joints_data):
            raise ValueError(f"Parent index {parent} out of bounds for joint {names[i]}.")
        if parent == i:
            raise ValueError(f"Joint '{names[i]}' cannot be parent of itself.")

        rp = item.get("rest_position")
        rest_position = None
        if rp is not None:
            rp_arr = np.asarray(rp, dtype=np.float64)
            if rp_arr.shape != (3,):
                raise ValueError(f"rest_position for joint '{names[i]}' must be length-3.")
            rest_position = rp_arr
        joints.append(JointDef(name=names[i], parent=parent, rest_position=rest_position))

    return RigModel(name=str(data.get("name", path.stem)), joints=tuple(joints))


def load_rig_from_gltf(path: Path) -> RigModel:
    gltf = json.loads(path.read_text(encoding="utf-8"))
    skins = gltf.get("skins", [])
    if not skins:
        raise ValueError("No 'skins' found in GLTF. Provide a skinned rig or a JSON rig definition.")
    skin = skins[0]
    joint_node_ids: List[int] = skin.get("joints", [])
    if not joint_node_ids:
        raise ValueError("Skin has no joints.")

    nodes = gltf.get("nodes", [])
    if not nodes:
        raise ValueError("GLTF has no nodes.")

    node_id_to_joint_idx = {node_id: i for i, node_id in enumerate(joint_node_ids)}
    joints: List[JointDef] = []
    for node_id in joint_node_ids:
        node = nodes[node_id]
        name = str(node.get("name", f"joint_{node_id}"))
        trn = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
        if trn.shape != (3,):
            trn = np.zeros(3, dtype=np.float64)
        parent_node_id = find_parent_node_id(nodes, node_id)
        parent_idx = node_id_to_joint_idx.get(parent_node_id, -1)
        joints.append(JointDef(name=name, parent=parent_idx, rest_position=trn))

    rig_name = str(gltf.get("asset", {}).get("generator", path.stem))
    return RigModel(name=rig_name, joints=tuple(joints))


def find_parent_node_id(nodes: Sequence[dict], child_node_id: int) -> Optional[int]:
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            if c == child_node_id:
                return i
    return None


def estimate_world_rotations(
    rest_positions: np.ndarray,
    pose_positions: np.ndarray,
    children: Dict[int, List[int]],
    parents: np.ndarray,
) -> np.ndarray:
    joint_count = rest_positions.shape[0]
    world_quats = np.zeros((joint_count, 4), dtype=np.float64)
    world_quats[:, 3] = 1.0

    for joint_idx in topological_joint_order(parents):
        child_ids = children.get(joint_idx, [])
        src_dirs: List[np.ndarray] = []
        dst_dirs: List[np.ndarray] = []

        for c in child_ids:
            src = rest_positions[c] - rest_positions[joint_idx]
            dst = pose_positions[c] - pose_positions[joint_idx]
            if np.linalg.norm(src) < EPS or np.linalg.norm(dst) < EPS:
                continue
            src_dirs.append(src / np.linalg.norm(src))
            dst_dirs.append(dst / np.linalg.norm(dst))

        if not src_dirs:
            parent = parents[joint_idx]
            if parent >= 0:
                world_quats[joint_idx] = world_quats[parent]
            else:
                world_quats[joint_idx] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            continue

        src_mat = np.stack(src_dirs, axis=0)
        dst_mat = np.stack(dst_dirs, axis=0)
        if len(src_dirs) == 1:
            quat = quaternion_from_two_vectors(src_mat[0], dst_mat[0])
        else:
            quat = quaternion_from_rotation_matrix(kabsch_rotation(src_mat, dst_mat))
        world_quats[joint_idx] = normalize_quaternion(quat)

    return world_quats


def topological_joint_order(parents: np.ndarray) -> List[int]:
    roots = [i for i, p in enumerate(parents) if p < 0]
    order: List[int] = []
    visited = np.zeros(len(parents), dtype=bool)
    queue = roots[:]
    children: Dict[int, List[int]] = {i: [] for i in range(len(parents))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    while queue:
        node = queue.pop(0)
        if visited[node]:
            continue
        visited[node] = True
        order.append(node)
        queue.extend(children[node])

    for i, seen in enumerate(visited):
        if not seen:
            order.append(i)
    return order


def world_to_local_quaternions(world_quats: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.zeros_like(world_quats)
    for i in range(world_quats.shape[0]):
        p = parents[i]
        if p < 0:
            local[i] = normalize_quaternion(world_quats[i])
        else:
            local[i] = normalize_quaternion(
                quaternion_multiply(quaternion_conjugate(world_quats[p]), world_quats[i])
            )
    return local


def kabsch_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    h = src.T @ dst
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def quaternion_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    a = normalize_vector(v0)
    b = normalize_vector(v1)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot < -1.0 + 1e-7:
        axis = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), a)
        if np.linalg.norm(axis) < EPS:
            axis = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), a)
        axis = normalize_vector(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + dot) * 2.0)
    inv_s = 1.0 / s
    quat = np.array([axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, 0.5 * s], dtype=np.float64)
    return normalize_quaternion(quat)


def quaternion_from_rotation_matrix(r: np.ndarray) -> np.ndarray:
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return normalize_quaternion(np.array([qx, qy, qz, qw], dtype=np.float64))


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quaternion(q0 + t * (q1 - q0))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / max(sin_theta_0, EPS)
    s1 = sin_theta / max(sin_theta_0, EPS)
    return normalize_quaternion((s0 * q0) + (s1 * q1))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n

def keyframes_to_dict(
    keyframes: Iterable[BoneKeyframe],
    rig_name: str,
    step: int,
    fps: float,
    backend: str,
    animation_prompt: Optional[str],
    planned_steps: Optional[Sequence[str]],
    motion_text: Optional[str],
    source_motion: Optional[str],
    model_image_path: Optional[str],
    llm_model: Optional[str],
) -> dict:
    keyframes = list(keyframes)
    return {
        "agent": "AnimationKeyframeAgent",
        "reference_paper": "LLMR: Real-time Prompting of Interactive Worlds using Large Language Models (arXiv:2309.12276)",
        "backend": backend,
        "llm_model": llm_model,
        "rig_name": rig_name,
        "step": int(step),
        "fps": float(fps),
        "time_sec": float(step / fps),
        "animation_prompt": animation_prompt,
        "planned_steps": list(planned_steps) if planned_steps is not None else None,
        "selected_step_motion_text": motion_text,
        "motion_text": motion_text,
        "source_motion": source_motion,
        "model_image_path": model_image_path,
        "bone_count": len(keyframes),
        "keyframes": [
            {
                "bone_index": k.bone_index,
                "bone_name": k.bone_name,
                "parent_index": k.parent_index,
                "parent_name": k.parent_name,
                "local_rotation_xyzw": [float(v) for v in k.local_rotation_xyzw],
                "local_translation_xyz": [float(v) for v in k.local_translation_xyz],
            }
            for k in keyframes
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-bone keyframes for a rigged model at a target animation step "
            "using an LLM motion description or geometric joint positions."
        )
    )
    parser.add_argument("--rig", type=Path, required=True, help="Path to rig JSON or GLTF file.")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help=(
            "Step index. In llm mode this is the planned motion step index. "
            "In geometry mode this is the frame index."
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second.")
    parser.add_argument(
        "--mode",
        choices=["llm", "geometry"],
        default="llm",
        help="Prediction backend: llm (text->keyframes) or geometry (joint positions->keyframes).",
    )
    parser.add_argument(
        "--motion-text",
        type=str,
        default=None,
        help=(
            "In llm mode: full user animation prompt (planner decomposes it into steps). "
            "In geometry mode: ignored."
        ),
    )
    parser.add_argument(
        "--motion-text-file",
        type=Path,
        default=None,
        help="Optional file containing the full animation prompt for llm mode.",
    )
    parser.add_argument(
        "--disable-planner",
        action="store_true",
        help=(
            "Bypass planning and treat --motion-text as a single step instruction "
            "(step must be 0 in this mode)."
        ),
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=None,
        help="Path to motion .npy file. Required in geometry mode.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Temporal smoothing factor in [0,1). Uses previous step if available.",
    )
    parser.add_argument(
        "--model-image",
        type=Path,
        default=None,
        help="Optional path to rigged model reference image for multimodal LLM prompting.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for llm mode.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible API.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable holding the API key.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for llm mode.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=3000,
        help="Max output tokens for llm mode.",
    )
    parser.add_argument(
        "--llm-timeout-sec",
        type=float,
        default=120.0,
        help="HTTP timeout for llm mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to keyframes_step_<step>.json",
    )
    parser.add_argument("--stdout", action="store_true", help="Print JSON output to stdout.")
    return parser.parse_args()


def read_motion_text(args: argparse.Namespace) -> Optional[str]:
    if args.motion_text is not None and args.motion_text.strip():
        return args.motion_text.strip()
    if args.motion_text_file is not None:
        return args.motion_text_file.read_text(encoding="utf-8").strip()
    return None


def build_llm_client(args: argparse.Namespace) -> OpenAICompatibleChatClient:
    api_key = os.environ.get(args.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key: set environment variable '{args.llm_api_key_env}' for llm mode."
        )
    config = LLMClientConfig(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=api_key,
        temperature=float(args.llm_temperature),
        max_tokens=int(args.llm_max_tokens),
        timeout_sec=float(args.llm_timeout_sec),
    )
    return OpenAICompatibleChatClient(config)


def main() -> None:
    args = parse_args()
    rig = load_rig(args.rig)
    animation_prompt = read_motion_text(args)
    motion = load_motion(args.motion) if args.motion is not None else None

    llm_client = build_llm_client(args) if args.mode == "llm" else None
    agent = AnimationKeyframeAgent(llm_client=llm_client)

    if args.mode == "geometry":
        if motion is None:
            raise ValueError("--motion is required in geometry mode.")
        if args.step >= motion.shape[0]:
            raise ValueError(
                f"Step {args.step} is out of range for motion with {motion.shape[0]} frames."
            )

    keyframes = agent.run(
        rig=rig,
        step=args.step,
        fps=args.fps,
        smoothing=args.smoothing,
        mode=args.mode,
        animation_prompt=animation_prompt,
        motion=motion,
        model_image_path=args.model_image,
        disable_planner=args.disable_planner,
    )
    planned_steps: Optional[Sequence[str]] = None
    selected_motion_text: Optional[str] = None
    selected_prompt: Optional[str] = None
    if agent.last_task is not None:
        planned_steps = agent.last_task.plan_steps
        selected_motion_text = agent.last_task.motion_text
        selected_prompt = agent.last_task.animation_prompt

    output_obj = keyframes_to_dict(
        keyframes=keyframes,
        rig_name=rig.name,
        step=args.step,
        fps=args.fps,
        backend=args.mode,
        animation_prompt=selected_prompt if args.mode == "llm" else None,
        planned_steps=planned_steps if args.mode == "llm" else None,
        motion_text=selected_motion_text if args.mode == "llm" else None,
        source_motion=str(args.motion) if args.motion is not None else None,
        model_image_path=str(args.model_image) if args.model_image is not None else None,
        llm_model=args.llm_model if args.mode == "llm" else None,
    )

    text = json.dumps(output_obj, indent=2)
    if args.stdout:
        print(text)
        return

    out_path = args.output or Path(f"keyframes_step_{args.step:04d}.json")
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
