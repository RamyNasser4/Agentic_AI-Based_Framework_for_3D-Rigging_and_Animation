from __future__ import annotations

from dataclasses import asdict, dataclass
from pprint import pprint
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import re
import math

try:
    import bpy as _bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ModuleNotFoundError:
    _bpy = None
    Vector = None


QuaternionXYZW = Tuple[float, float, float, float]
VectorXYZ = Tuple[float, float, float]
AnimationRepairCallback = Callable[[str, Sequence[str]], str]


_FIRST_PAYLOAD_PATTERN = re.compile(r"^(.*?)(?=,\[|,\()")
_LOCATION_PATTERN = re.compile(r"\[([^\]]*)\]")
_ROTATION_PATTERN = re.compile(r"\(([^\)]*)\)")
_ANY_PAYLOAD_PATTERN = re.compile(r"\[[^\]]*\]|\([^\)]*\)")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def validate_animation_text(animation_text):
    """Validate raw keyframe-agent animation text before Blender execution."""
    errors = []
    lines = _normalize_animation_lines(animation_text)

    if not lines:
        errors.append("No animation lines were provided.")
        return {"valid": False, "errors": errors}

    for line_number, line in lines:
        target_path = _extract_animation_target(line)
        if target_path is None:
            errors.append(f"Line {line_number}: Could not parse animation target.")
        else:
            _validate_target_path(target_path, line_number, errors)

        location_matches = _LOCATION_PATTERN.findall(line)
        rotation_matches = _ROTATION_PATTERN.findall(line)

        if not location_matches and not rotation_matches:
            errors.append(f"Line {line_number}: No keyframe payload found.")

        if location_matches and rotation_matches:
            errors.append(f"Line {line_number}: Line mixes translation and rotation payloads.")

        for payload_index, payload in enumerate(location_matches, start=1):
            _validate_payload(
                payload=payload,
                expected_size=4,
                label="Translation",
                line_number=line_number,
                payload_index=payload_index,
                errors=errors,
            )

        for payload_index, payload in enumerate(rotation_matches, start=1):
            _validate_payload(
                payload=payload,
                expected_size=5,
                label="Quaternion",
                line_number=line_number,
                payload_index=payload_index,
                errors=errors,
            )

    return {"valid": len(errors) == 0, "errors": errors}


def prepare_animation_text_for_execution(
    animation_text: str,
    repair_callback: Optional[AnimationRepairCallback] = None,
) -> str:
    validation = validate_animation_text(animation_text)
    if validation["valid"]:
        print("[Validator] Validation successful")
        return str(animation_text or "").strip()

    _log_validation_errors(validation["errors"])

    auto_repaired_text, repaired_count = auto_repair_animation_text(animation_text, validation["errors"])
    print(f"[Validator] Auto-repaired {repaired_count} errors")

    validation = validate_animation_text(auto_repaired_text)
    if validation["valid"]:
        print("[Validator] Validation successful")
        return auto_repaired_text

    _log_validation_errors(validation["errors"])

    if repair_callback is not None:
        print("[Validator] Requesting LLM repair")
        llm_repaired_text = repair_callback(auto_repaired_text, validation["errors"])
        llm_repaired_text, _ = auto_repair_animation_text(llm_repaired_text, validation["errors"])
        validation = validate_animation_text(llm_repaired_text)
        if validation["valid"]:
            print("[Validator] Validation successful")
            return llm_repaired_text
        _log_validation_errors(validation["errors"])

    raise ValueError("Animation validation failed:\n" + "\n".join(validation["errors"]))


def auto_repair_animation_text(animation_text: str, errors: Optional[Sequence[str]] = None) -> Tuple[str, int]:
    before_error_count = len(errors) if errors is not None else len(validate_animation_text(animation_text)["errors"])
    text = str(animation_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    repaired_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        payloads = _ANY_PAYLOAD_PATTERN.findall(line)
        if not payloads:
            continue

        target_path = _extract_animation_target(line)
        if target_path is None:
            repaired_lines.append(line)
            continue

        location_payloads = [payload for payload in payloads if payload.startswith("[")]
        rotation_payloads = [payload for payload in payloads if payload.startswith("(")]

        if location_payloads:
            repaired_lines.append(",".join([target_path] + location_payloads))
        if rotation_payloads:
            repaired_lines.append(",".join([target_path] + rotation_payloads))

    repaired_text = "\n".join(repaired_lines)
    after_error_count = len(validate_animation_text(repaired_text)["errors"])
    repaired_count = max(0, before_error_count - after_error_count)
    return repaired_text, repaired_count


def _normalize_animation_lines(animation_text: str) -> List[Tuple[int, str]]:
    return [
        (index, line.strip())
        for index, line in enumerate(str(animation_text or "").splitlines(), start=1)
        if line.strip()
    ]


def _extract_animation_target(line: str) -> Optional[str]:
    match = _FIRST_PAYLOAD_PATTERN.match(line.strip())
    if match is None:
        return None
    return match.group(1).rstrip(", ").strip()


def _validate_target_path(target_path: str, line_number: int, errors: List[str]) -> None:
    if not target_path:
        errors.append(f"Line {line_number}: Animation target path is empty.")
        return

    empty_parts = [part for part in target_path.split("/") if not part.strip()]
    if empty_parts:
        errors.append(f"Line {line_number}: Empty joint name in target path `{target_path}`.")


def _validate_payload(
    payload: str,
    expected_size: int,
    label: str,
    line_number: int,
    payload_index: int,
    errors: List[str],
) -> None:
    parts = [part.strip() for part in payload.split(",")]
    if len(parts) != expected_size:
        errors.append(
            f"Line {line_number}: {label} keyframe {payload_index} has {len(parts)} values; expected {expected_size}."
        )
        if len(parts) == expected_size - 1 or not parts or not parts[0]:
            errors.append(f"Line {line_number}: {label} keyframe {payload_index} is missing a timestamp.")

    if parts and not parts[0]:
        errors.append(f"Line {line_number}: {label} keyframe {payload_index} is missing a timestamp.")

    for value_index, part in enumerate(parts, start=1):
        if not part:
            errors.append(
                f"Line {line_number}: {label} keyframe {payload_index} value {value_index} is empty."
            )
            continue

        if _NUMBER_PATTERN.fullmatch(part) is None:
            errors.append(
                f"Line {line_number}: {label} keyframe {payload_index} value `{part}` is not numeric."
            )
            continue

        try:
            number = float(part)
        except ValueError:
            errors.append(
                f"Line {line_number}: {label} keyframe {payload_index} value `{part}` is not numeric."
            )
            continue

        if not math.isfinite(number):
            errors.append(
                f"Line {line_number}: {label} keyframe {payload_index} value `{part}` is not finite."
            )


def _log_validation_errors(errors: Sequence[str]) -> None:
    print(f"[Validator] Found {len(errors)} errors")
    for error in errors:
        print(f"[Validator] {error}")


@dataclass(frozen=True)
class KeyframePoint:
    time_sec: float
    values: Tuple[float, ...]


@dataclass(frozen=True)
class AnimationTrack:
    target_path: str
    channel: str
    keyframes: Tuple[KeyframePoint, ...]

    @property
    def target_name(self) -> str:
        return self.target_path.split("/")[-1]


@dataclass(frozen=True)
class ParsedAnimation:
    armature_name: str
    tracks: Tuple[AnimationTrack, ...]


class BlenderExecutor:
    """Parse `KeyFrameAgent` output and apply it to a Blender armature.

    Expected line format from `KeyFrameAgent`:
    - Root/object translation:
      `Armature,[0.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0]`
      where each bracket payload is `[time,x,y,z]`
    - Object/bone rotation:
      `Armature/Spine,(0.0,0.0,0.0,0.0,1.0),(1.0,0.1,0.0,0.0,0.99)`
      where each tuple payload is `(time,x,y,z,w)`

    Notes:
    - The keyframe-agent quaternion order is `x,y,z,w`.
    - Blender's `rotation_quaternion` expects `w,x,y,z`.
    """

    _FIRST_PAYLOAD_PATTERN = re.compile(r"^(.*?)(?=,\[|,\()")
    _LOCATION_PATTERN = re.compile(r"\[([^\]]+)\]")
    _ROTATION_PATTERN = re.compile(r"\(([^\)]+)\)")

    def __init__(
        self,
        bpy_module=None,
        fps: Optional[float] = None,
        frame_start: Optional[float] = None,
        animation_repairer: Optional[AnimationRepairCallback] = None,
    ) -> None:
        self.bpy = _bpy if bpy_module is None else bpy_module
        self.fps = fps
        self.frame_start = frame_start
        self.animation_repairer = animation_repairer

    @property
    def blender_available(self) -> bool:
        return self.bpy is not None

    def parse_output(self, output: Union[str, Sequence[str]]) -> ParsedAnimation:
        lines = self._normalize_lines(output)
        if not lines:
            raise ValueError("No animation lines were provided.")

        tracks = [self._parse_line(line) for line in lines]
        armature_name = tracks[0].target_path.split("/")[0]
        return ParsedAnimation(armature_name=armature_name, tracks=tuple(tracks))

    def execute(
        self,
        output: Union[str, Sequence[str], ParsedAnimation],
        armature_object=None,
        armature_name: Optional[str] = None,
        fps: Optional[float] = None,
        frame_start: Optional[float] = None,
        clear_existing_action: bool = False,
    ) -> ParsedAnimation:
        parsed = output if isinstance(output, ParsedAnimation) else self.parse_output(output)
        armature = self._resolve_armature_object(
            armature_object=armature_object,
            armature_name=armature_name or parsed.armature_name,
        )
        effective_fps = self._resolve_fps(fps)
        effective_frame_start = self._resolve_frame_start(frame_start)

        self._prepare_animation_data(armature, clear_existing_action=clear_existing_action)

        for track in parsed.tracks:
            target = self._resolve_target(armature, track)
            self._apply_track(
                target=target,
                track=track,
                fps=effective_fps,
                frame_start=effective_frame_start,
            )

        return parsed

    def execute_from_text(
        self,
        output_text: str,
        armature_object=None,
        armature_name: Optional[str] = None,
        fps: Optional[float] = None,
        frame_start: Optional[float] = None,
        clear_existing_action: bool = False,
    ) -> ParsedAnimation:
        output_text = prepare_animation_text_for_execution(
            output_text,
            repair_callback=self.animation_repairer,
        )
        return self.execute(
            output=output_text,
            armature_object=armature_object,
            armature_name=armature_name,
            fps=fps,
            frame_start=frame_start,
            clear_existing_action=clear_existing_action,
        )

    def _normalize_lines(self, output: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(output, str):
            raw_lines = output.splitlines()
        else:
            raw_lines = list(output)
        return [line.strip() for line in raw_lines if str(line).strip()]

    def _parse_line(self, line: str) -> AnimationTrack:
        target_path = self._extract_target_path(line)
        location_matches = self._LOCATION_PATTERN.findall(line)
        rotation_matches = self._ROTATION_PATTERN.findall(line)

        if location_matches and rotation_matches:
            raise ValueError(f"Line mixes translation and rotation payloads: {line}")
        if location_matches:
            keyframes = tuple(self._parse_keyframes(location_matches, expected_size=4, line=line))
            return AnimationTrack(target_path=target_path, channel="location", keyframes=keyframes)
        if rotation_matches:
            keyframes = tuple(self._parse_keyframes(rotation_matches, expected_size=5, line=line))
            return AnimationTrack(
                target_path=target_path,
                channel="rotation_quaternion",
                keyframes=keyframes,
            )

        raise ValueError(f"No keyframe payload found in line: {line}")

    def _extract_target_path(self, line: str) -> str:
        match = self._FIRST_PAYLOAD_PATTERN.match(line.strip())
        if match is None:
            raise ValueError(f"Could not parse animation target from line: {line}")
        target_path = match.group(1).rstrip(", ")
        if not target_path:
            raise ValueError(f"Animation target path is empty: {line}")
        return target_path

    def _parse_keyframes(
        self,
        matches: Iterable[str],
        expected_size: int,
        line: str,
    ) -> List[KeyframePoint]:
        parsed: List[KeyframePoint] = []
        for match in matches:
            parts = [part.strip() for part in match.split(",")]
            if len(parts) != expected_size:
                raise ValueError(
                    f"Expected {expected_size} values in payload `{match}` but got {len(parts)} in line: {line}"
                )
            values = tuple(float(part) for part in parts)
            parsed.append(KeyframePoint(time_sec=values[0], values=values[1:]))
        parsed.sort(key=lambda item: item.time_sec)
        return parsed

    def _resolve_armature_object(self, armature_object=None, armature_name: Optional[str] = None):
        if not self.blender_available:
            raise RuntimeError("`bpy` is not available. Run this code inside Blender or inject a bpy-compatible module.")

        if armature_object is not None:
            armature = armature_object
        else:
            if not armature_name:
                raise ValueError("An armature name or armature object must be provided.")
            armature = self.bpy.data.objects.get(armature_name)
            if armature is None:
                raise KeyError(f"Armature object `{armature_name}` was not found in the current Blender file.")

        if getattr(armature, "type", None) != "ARMATURE":
            raise TypeError(f"Object `{armature.name}` is not an armature.")
        return armature

    def _resolve_fps(self, fps: Optional[float]) -> float:
        if fps is not None:
            return float(fps)
        if self.fps is not None:
            return float(self.fps)
        if self.blender_available and getattr(self.bpy.context, "scene", None) is not None:
            scene = self.bpy.context.scene
            fps_value = float(scene.render.fps)
            fps_base = float(scene.render.fps_base) if float(scene.render.fps_base) != 0.0 else 1.0
            return fps_value / fps_base
        return 24.0

    def _resolve_frame_start(self, frame_start: Optional[float]) -> float:
        if frame_start is not None:
            return float(frame_start)
        if self.frame_start is not None:
            return float(self.frame_start)
        if self.blender_available and getattr(self.bpy.context, "scene", None) is not None:
            return float(self.bpy.context.scene.frame_start)
        return 1.0

    def _prepare_animation_data(self, armature, clear_existing_action: bool) -> None:
        armature.animation_data_create()
        if clear_existing_action or armature.animation_data.action is None:
            action_name = f"{armature.name}_Action"
            armature.animation_data.action = self.bpy.data.actions.new(name=action_name)

    def _resolve_target(self, armature, track: AnimationTrack):
        if "/" not in track.target_path:
            return armature

        bone_name = track.target_name
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            raise KeyError(
                f"Pose bone `{bone_name}` was not found on armature `{armature.name}` for track `{track.target_path}`."
            )
        return pose_bone

    def _apply_track(self, target, track: AnimationTrack, fps: float, frame_start: float) -> None:
        if track.channel == "rotation_quaternion":
            target.rotation_mode = "QUATERNION"
            for keyframe in track.keyframes:
                target.rotation_quaternion = self._xyzw_to_wxyz(keyframe.values)
                target.keyframe_insert(
                    data_path="rotation_quaternion",
                    frame=self._time_to_frame(keyframe.time_sec, fps, frame_start),
                    group=getattr(target, "name", "Object Transforms"),
                )
            return

        if track.channel == "location":
            root_bone = None
            if getattr(target, "type", None) == "ARMATURE" and getattr(target, "pose", None) is not None:
                root_bone = target.pose.bones.get("root")
                if root_bone is None:
                    root_bone = next(
                        (pose_bone for pose_bone in target.pose.bones if pose_bone.parent is None),
                        None,
                    )

            for keyframe in track.keyframes:
                if root_bone is not None:
                    if Vector is None:
                        raise RuntimeError("mathutils.Vector is unavailable; run this inside Blender.")
                    local_vec = Vector(keyframe.values)
                    world_vec = root_bone.matrix.to_3x3() @ local_vec
                    target.location = world_vec
                else:
                    target.location = keyframe.values
                target.keyframe_insert(
                    data_path="location",
                    frame=self._time_to_frame(keyframe.time_sec, fps, frame_start),
                    group=getattr(target, "name", "Object Transforms"),
                )
            return

        raise ValueError(f"Unsupported track channel: {track.channel}")

    def _time_to_frame(self, time_sec: float, fps: float, frame_start: float) -> float:
        return frame_start + (float(time_sec) * float(fps))

    def _xyzw_to_wxyz(self, quaternion_xyzw: Sequence[float]) -> QuaternionXYZW:
        if len(quaternion_xyzw) != 4:
            raise ValueError(f"Quaternion payload must have 4 values, got {len(quaternion_xyzw)}.")
        x, y, z, w = (float(value) for value in quaternion_xyzw)
        return (w, x, y, z)


if __name__ == "__main__":
    sample_output = """
    SMPLX-lh-male,(0.0,1.0,0.0,0.0,0.0),(2.0,1.0,0.0,0.0,0.0)
    SMPLX-lh-male/root,(0.0,0.0,0.0,-0.7,-0.7),(2.0,0.0,0.0,-0.7,-0.7)
    SMPLX-lh-male/root/pelvis,(0.0,1.0,0.0,0.0,0.0),(2.0,1.0,0.0,0.0,0.0)
    SMPLX-lh-male/root/pelvis/left_hip/left_knee,(0.0,0.99,0.12,0.0,0.0),(2.0,0.99,0.12,0.0,0.0)
    SMPLX-lh-male/root/pelvis/right_hip/right_knee,(0.0,0.99,0.12,0.0,0.0),(2.0,0.99,0.12,0.0,0.0)
    SMPLX-lh-male/root/pelvis/spine1,(0.0,1.0,0.0,0.0,0.0),(2.0,1.0,0.0,0.0,0.0)
    SMPLX-lh-male/root/pelvis/spine1/spine2,(0.0,1.0,0.0,0.0,0.0),(2.0,1.0,0.0,0.0,0.0)
    SMPLX-lh-male/root/pelvis/spine1/spine2/spine3,(0.0,1.0,0.0,0.0,0.0),(2.0,1.0,0.0,0.0,0.0)
    SMPLX-lh-male/root/pelvis/spine1/spine2/spine3/neck/head,(0.0,1.0,0.0,0.0,0.0),(2.0,1.0,0.0,0.0,0.0)
    """.strip()

    executor = BlenderExecutor()
    parsed = executor.parse_output(sample_output)
    pprint(asdict(parsed), sort_dicts=False)
