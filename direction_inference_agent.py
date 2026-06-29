from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - Blender-only imports
    bpy = None

try:
    from .mesh_renderer import MeshRenderer, RenderedImage
    from .multimodal_utils import compress_image_for_llm
    from .SceneParser import SceneParser
    from .skeleton_recorder import extract_skeleton_frames
    from .skeleton_visualizer import render_skeleton_images
except ImportError:  # pragma: no cover - direct script fallback
    from mesh_renderer import MeshRenderer, RenderedImage
    from multimodal_utils import compress_image_for_llm
    from SceneParser import SceneParser
    from skeleton_recorder import extract_skeleton_frames
    from skeleton_visualizer import render_skeleton_images


DIRECTION_SYSTEM_PROMPT = """You are a multimodal direction inference agent for Blender assets.

You receive:
- rendered mesh images from fixed Blender camera viewpoints,
- skeleton visualization images sampled from animation frames,
- structured scene and rig text from SceneParser,
- trusted Blender world-space camera metadata for every mesh render.

Each rendered image is accompanied by trusted Blender camera metadata describing the exact
camera position and orientation in world space. Render identifiers such as Camera View 1,
Camera View 2, etc. are arbitrary identifiers only. Do not infer semantic meaning from their
names.

Use the mesh renders as the primary evidence for semantic orientation. Determine which
anatomical side of the asset is visible by combining the rendered image, supplied camera
orientation, SceneParser information, and skeleton visualizations. The camera metadata is
trusted input metadata; do not infer or alter it.

Determine the semantic forward axis, up axis, right axis, and whether the asset is humanoid.
Axes must be Blender local axis labels: +X, -X, +Y, -Y, +Z, -Z. If the evidence is not strong
enough, return null for uncertain fields and set needs_user_confirmation to true.

Return JSON only with this exact shape:
{
  "forward_axis": "+Y | -Y | +X | -X | +Z | -Z | null",
  "up_axis": "+Z | -Z | +Y | -Y | +X | -X | null",
  "right_axis": "+X | -X | +Y | -Y | +Z | -Z | null",
  "is_humanoid": true | false | null,
  "confidence": 0.0,
  "reasoning": "...",
  "needs_user_confirmation": true
}

Be conservative. Do not fabricate a confident answer."""


AXIS_LABELS = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}


@dataclass
class DirectionInferenceResult:
    forward_axis: Optional[str]
    up_axis: Optional[str]
    right_axis: Optional[str]
    is_humanoid: Optional[bool]
    camera_coordinates_in_blender: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    needs_user_confirmation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_planner_context(self) -> str:
        return (
            "Semantic direction inference: "
            f"forward_axis={self._format_context_value(self.forward_axis)}, "
            f"up_axis={self._format_context_value(self.up_axis)}, "
            f"right_axis={self._format_context_value(self.right_axis)}, "
            f"is_humanoid={self._format_context_value(self.is_humanoid)}, "
            f"confidence={self.confidence:.2f}, "
            f"needs_user_confirmation={self._format_context_value(self.needs_user_confirmation)}, "
            f"reasoning={self.reasoning}"
        )

    def _format_context_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


class DirectionInferenceAgent:
    def __init__(
        self,
        context=None,
        model: str = "gemma-4-31b-it",
        confidence_threshold: float = 0.65,
        render_resolution: int = 768,
    ) -> None:
        self.context = context
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.render_resolution = render_resolution

    def infer(
        self,
        selected_objects: Optional[Iterable[object]] = None,
    ) -> DirectionInferenceResult:
        if bpy is None:
            raise RuntimeError("DirectionInferenceAgent must run inside Blender.")

        objects = self._resolve_objects(selected_objects)
        if not objects:
            raise ValueError("DirectionInferenceAgent requires at least one selected object.")

        parser = SceneParser(context=self.context, precision=4)
        scene_info = parser.generate_scene_info(objects)
        armatures = self._find_armatures(objects)
        skeleton_images = self._render_skeleton_context(armatures)
        renderer = MeshRenderer(context=self.context)
        mesh_renders = renderer.render_views_single_frame(
            objects,
            resolution=self.render_resolution,
        )

        result = self._invoke_multimodal_model(
            scene_info=scene_info,
            mesh_renders=mesh_renders,
            skeleton_images=skeleton_images,
        )
        result.camera_coordinates_in_blender = [
            image.camera_metadata for image in mesh_renders
        ]
        return result

    def _invoke_multimodal_model(
        self,
        scene_info: str,
        mesh_renders: List[RenderedImage],
        skeleton_images: List[np.ndarray],
    ) -> DirectionInferenceResult:
        if not mesh_renders and not skeleton_images:
            return DirectionInferenceResult(
                forward_axis=None,
                up_axis=None,
                right_axis=None,
                is_humanoid=None,
                confidence=0.0,
                reasoning="No mesh renders or skeleton visualizations were available.",
                needs_user_confirmation=True,
            )

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
        camera_metadata = [image.camera_metadata for image in mesh_renders]
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Analyze this Blender asset orientation.\n\n"
                    "SceneParser structured scene information:\n"
                    f"{scene_info}\n\n"
                    "Mesh renders follow first and are the primary semantic evidence. "
                    "Each render is paired directly with trusted Blender world-space camera "
                    "metadata. Camera View identifiers are arbitrary and carry no semantic "
                    "viewpoint meaning. Do not assume any view is a front, back, left, or "
                    "right view from its identifier. "
                    "Skeleton visualizations follow after them as supporting structural context. "
                    "Return strict JSON only."
                ),
            }
        ]

        for index, image in enumerate(mesh_renders, start=1):
            content.append(
                {
                    "type": "text",
                    "text": self._format_camera_view_prompt(index, image.camera_metadata),
                }
            )
            content.append({"type": "image_url", "image_url": {"url": image.data_url}})

        for index, image in enumerate(skeleton_images, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Skeleton visualization collage {index}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._numpy_image_to_data_url(image)},
                }
            )

        messages = [
            SystemMessage(content=DIRECTION_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        last_error: Exception | None = None
        retry_messages = list(messages)
        for _ in range(3):
            try:
                response = llm.invoke(retry_messages)
                payload = self._parse_response_json(self._extract_response_text(response.content))
                return self._result_from_payload(payload, camera_metadata)
            except Exception as error:
                last_error = error
                retry_messages = retry_messages + [
                    HumanMessage(
                        content="Return only valid JSON matching the required schema. Fix formatting errors."
                    )
                ]

        raise ValueError(f"Direction inference failed to return valid JSON: {last_error}") from last_error

    def _format_camera_view_prompt(self, index: int, camera_metadata: Dict[str, Any]) -> str:
        location = self._format_metadata_vector(camera_metadata.get("location"))
        forward = self._format_metadata_vector(camera_metadata.get("forward_direction"))
        up = self._format_metadata_vector(camera_metadata.get("up_direction"))
        right = self._format_metadata_vector(camera_metadata.get("right_direction"))
        projection = str(camera_metadata.get("projection") or "UNKNOWN")

        return (
            f"Camera View {index}\n\n"
            "Camera location (Blender world):\n"
            f"{location}\n\n"
            "Camera forward vector:\n"
            f"{forward}\n\n"
            "Camera up vector:\n"
            f"{up}\n\n"
            "Camera right vector:\n"
            f"{right}\n\n"
            "Projection:\n"
            f"{projection}\n\n"
            "The above metadata is trusted Blender world-space information. Use it together "
            "with the rendered image to determine which side of the asset is visible. Do not "
            "assume this is a front, back, left, or right view."
        )

    def _format_metadata_vector(self, values: Any) -> str:
        if isinstance(values, (list, tuple)) and len(values) >= 3:
            return f"({values[0]}, {values[1]}, {values[2]})"
        return "(unknown, unknown, unknown)"

    def _render_skeleton_context(self, armatures: Sequence[object]) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        for armature in armatures:
            try:
                frames = extract_skeleton_frames(armature.name)
                images.extend(render_skeleton_images(frames))
            except Exception as error:
                print(f"[DirectionInference] Skeleton context failed for `{armature.name}`: {error}")
        return images

    def _resolve_objects(self, selected_objects: Optional[Iterable[object]]) -> List[object]:
        if selected_objects is None:
            context = self.context or bpy.context
            return list(context.selected_objects)

        objects = []
        for item in selected_objects:
            if isinstance(item, str):
                obj = bpy.data.objects.get(item)
                if obj is None:
                    raise ValueError(f"Object {item!r} was not found in the current Blender file.")
                objects.append(obj)
                continue
            if hasattr(item, "name") and hasattr(item, "type"):
                objects.append(item)
                continue
            raise TypeError("selected_objects must contain Blender object references or object names.")
        return objects

    def _find_armatures(self, objects: Sequence[object]) -> List[object]:
        armatures: List[object] = []
        seen = set()

        def add_armature(obj):
            if obj is not None and getattr(obj, "type", None) == "ARMATURE" and obj.name not in seen:
                armatures.append(obj)
                seen.add(obj.name)

        for obj in objects:
            add_armature(obj)
            if getattr(obj, "type", None) == "MESH":
                add_armature(getattr(obj, "parent", None))
                for modifier in getattr(obj, "modifiers", []):
                    if getattr(modifier, "type", None) == "ARMATURE":
                        add_armature(getattr(modifier, "object", None))
            for child in getattr(obj, "children_recursive", []):
                add_armature(child)

        return armatures

    def _numpy_image_to_data_url(self, image: np.ndarray) -> str:
        return compress_image_for_llm(image)

    def _extract_response_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return str(content)

    def _parse_response_json(self, text: str) -> Dict[str, Any]:
        candidate = text.strip()
        if not candidate:
            raise ValueError("Direction inference returned an empty response.")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(candidate[start : end + 1])

    def _result_from_payload(
        self,
        payload: Dict[str, Any],
        camera_metadata: List[Dict[str, Any]],
    ) -> DirectionInferenceResult:
        if not isinstance(payload, dict):
            raise TypeError("Direction inference payload must be a JSON object.")

        confidence = payload.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or math.isnan(float(confidence)):
            raise ValueError("confidence must be numeric.")
        confidence = max(0.0, min(1.0, float(confidence)))

        forward_axis = self._optional_axis(payload.get("forward_axis"), "forward_axis")
        up_axis = self._optional_axis(payload.get("up_axis"), "up_axis")
        right_axis = self._optional_axis(payload.get("right_axis"), "right_axis")
        is_humanoid = payload.get("is_humanoid")
        if is_humanoid is not None and not isinstance(is_humanoid, bool):
            raise ValueError("is_humanoid must be true, false, or null.")

        needs_user_confirmation = payload.get("needs_user_confirmation")
        if needs_user_confirmation is None:
            needs_user_confirmation = confidence < self.confidence_threshold
        if not isinstance(needs_user_confirmation, bool):
            raise ValueError("needs_user_confirmation must be boolean.")

        if confidence < self.confidence_threshold:
            needs_user_confirmation = True

        reasoning = payload.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return DirectionInferenceResult(
            forward_axis=forward_axis,
            up_axis=up_axis,
            right_axis=right_axis,
            is_humanoid=is_humanoid,
            camera_coordinates_in_blender=camera_metadata,
            confidence=confidence,
            reasoning=reasoning.strip(),
            needs_user_confirmation=needs_user_confirmation,
        )

    def _optional_axis(self, value: Any, field_name: str) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be an axis label or null.")
        value = value.strip().upper()
        if value not in AXIS_LABELS:
            raise ValueError(f"{field_name} must be one of {sorted(AXIS_LABELS)} or null.")
        return value


def infer_directions(selected_objects: Optional[Iterable[object]] = None, context=None) -> DirectionInferenceResult:
    agent = DirectionInferenceAgent(context=context)
    return agent.infer(selected_objects)
