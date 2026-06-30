from __future__ import annotations

from typing import Dict, List, Tuple

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - only available inside Blender
    bpy = None


_LAST_BONE_EDGES: List[Tuple[str, str]] = []
_LAST_FRAME_NUMBERS: List[int] = []


def get_recorded_bone_edges() -> List[Tuple[str, str]]:
    return list(_LAST_BONE_EDGES)


def get_recorded_frame_numbers() -> List[int]:
    return list(_LAST_FRAME_NUMBERS)


def extract_skeleton_frames(armature_name: str) -> List[Dict[str, List[float]]]:
    if bpy is None:
        raise RuntimeError("skeleton_recorder must run inside Blender because bpy is unavailable.")

    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise KeyError(f"Armature `{armature_name}` was not found in the current Blender file.")
    if getattr(armature, "type", None) != "ARMATURE":
        raise TypeError(f"Object `{armature_name}` is not an armature.")

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    action = getattr(getattr(armature, "animation_data", None), "action", None)
    if action is not None:
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
    else:
        frame_start = int(scene.frame_start)
        frame_end = int(scene.frame_end)
    original_frame = int(scene.frame_current)

    frames: List[Dict[str, List[float]]] = []
    global _LAST_BONE_EDGES, _LAST_FRAME_NUMBERS
    _LAST_BONE_EDGES = [
        (pose_bone.parent.name, pose_bone.name)
        for pose_bone in armature.pose.bones
        if pose_bone.parent is not None
    ]
    _LAST_FRAME_NUMBERS = list(range(frame_start, frame_end + 1))

    try:
        for frame_index in range(frame_start, frame_end + 1):
            scene.frame_set(frame_index)
            bpy.context.view_layer.update()

            evaluated_armature = armature.evaluated_get(depsgraph)
            world_matrix = evaluated_armature.matrix_world.copy()

            frame_data: Dict[str, List[float]] = {}
            for pose_bone in evaluated_armature.pose.bones:
                head_world = world_matrix @ pose_bone.head
                frame_data[pose_bone.name] = [
                    float(head_world.x),
                    float(head_world.y),
                    float(head_world.z),
                ]

            frames.append(frame_data)
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()

    return frames
