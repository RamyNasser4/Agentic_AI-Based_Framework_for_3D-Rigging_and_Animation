from __future__ import annotations

from dataclasses import asdict, dataclass
from pprint import pprint
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import re

try:
    import bpy as _bpy  # type: ignore
except ModuleNotFoundError:
    _bpy = None


QuaternionXYZW = Tuple[float, float, float, float]
VectorXYZ = Tuple[float, float, float]


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
    ) -> None:
        self.bpy = _bpy if bpy_module is None else bpy_module
        self.fps = fps
        self.frame_start = frame_start

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
            for keyframe in track.keyframes:
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
    metarig,[0.0,0.0,0.0,0.0],[1.0,0.5,0.0,0.0]  
    metarig/spine,(0.0,0.7,0.0,0.0,0.7),(0.5,0.75,0.0,0.0,0.65),(1.0,0.7,0.0,0.0,0.7)
    metarig/spine/spine.001,(0.0,0.7,0.0,0.0,0.7),(0.5,0.75,0.0,0.0,0.65),(1.0,0.7,0.0,0.0,0.7)
    metarig/spine/spine.001/spine.002/spine.003/spine.006,(0.0,0.0,0.0,0.0,1.0),(0.5,0.0,0.0,0.0,1.0),(1.0,0.0,0.0,0.0,1.0)
    metarig/spine/spine.001/spine.002/spine.003/spine.006/ear.L,(0.0,0.0,0.0,0.0,1.0),(0.5,0.0,0.0,0.1,0.99),(1.0,0.0,0.0,0.0,1.0)
    metarig/spine/spine.001/spine.002/spine.003/spine.006/ear.R,(0.0,0.0,0.0,0.0,1.0),(0.5,0.0,0.0,-0.1,0.99),(1.0,0.0,0.0,0.0,1.0)
    metarig/thigh.L,(0.0,0.2,0.0,0.0,0.98),(0.5,0.4,0.0,0.0,0.90),(1.0,0.2,0.0,0.0,0.98)
    metarig/thigh.L/shin.L,(0.0,0.0,0.0,0.0,1.0),(0.5,-0.2,0.0,0.0,0.90),(1.0,0.0,0.0,0.0,1.0)
    metarig/thigh.L/shin.L/foot.L,(0.0,0.0,0.0,0.0,1.0),(0.5,0.1,0.0,0.0,0.95),(1.0,0.0,0.0,0.0,1.0)     
    metarig/thigh.R,(0.0,0.0,0.0,0.0,0.98),(0.5,0.2,0.0,0.0,0.98),(1.0,0.0,0.0,0.0,0.98)
    metarig/thigh.R/shin.R,(0.0,0.0,0.0,0.0,1.0),(0.5,-0.2,0.0,0.0,0.90),(1.0,0.0,0.0,0.0,1.0)
    metarig/thigh.R/shin.R/foot.R,(0.0,0.0,0.0,0.0,1.0),(0.5,0.1,0.0,0.0,0.95),(1.0,0.0,0.0,0.0,1.0)     
    metarig/tail,(0.0,0.0,0.0,0.0,1.0),(0.5,0.0,0.1,0.0,0.98),(1.0,0.0,0.0,0.0,1.0)
    metarig/tail/tail.001,(0.0,0.0,0.0,0.0,1.0),(0.5,0.0,0.1,0.0,0.98),(1.0,0.0,0.0,0.0,1.0)
    metarig/tail/tail.001/tail.002,(0.0,0.0,0.0,0.0,1.0),(0.5,0.0,0.1,0.0,0.98),(1.0,0.0,0.0,0.0,1.0)    
    metarig/tail/tail.001/tail.002/tail.003,(0.0,0.0,0.0,0.0,1.0),(0.5,0.0,0.1,0.0,0.98),(1.0,0.0,0.0,0.0,1.0)
    """.strip()

    executor = BlenderExecutor()
    parsed = executor.parse_output(sample_output)
    pprint(asdict(parsed), sort_dicts=False)
