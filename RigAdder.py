import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Vector, Quaternion


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from parser import parse_rig_data, rig_data, rig_data_whale


Vector3 = Tuple[float, float, float]
Quaternion4 = Tuple[float, float, float, float]


def _vector3(values: Tuple[float, ...], default: Vector3 = (0.0, 0.0, 0.0)) -> Vector:
    if not values or len(values) < 3:
        return Vector(default)
    return Vector((float(values[0]), float(values[1]), float(values[2])))


def _quat_xyzw(values: Tuple[float, ...], default: Quaternion4 = (0.0, 0.0, 0.0, 1.0)) -> Quaternion:
    if not values or len(values) < 4:
        x, y, z, w = default
    else:
        x, y, z, w = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    return Quaternion((w, x, y, z))


def _ensure_mode(mode: str) -> None:
    if bpy.context.mode != mode:
        bpy.ops.object.mode_set(mode=mode)


def _create_or_reset_armature(armature_name: str):
    existing = bpy.data.objects.get(armature_name)
    if existing and existing.type == "ARMATURE":
        bpy.context.view_layer.objects.active = existing
        existing.select_set(True)
        _ensure_mode("EDIT")
        edit_bones = existing.data.edit_bones
        for edit_bone in list(edit_bones):
            edit_bones.remove(edit_bone)
        _ensure_mode("OBJECT")
        return existing

    arm_data = bpy.data.armatures.new(armature_name)
    arm_obj = bpy.data.objects.new(armature_name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    return arm_obj


def _build_edit_bones(arm_obj, root_node: Dict) -> Dict[str, bpy.types.EditBone]:
    _ensure_mode("EDIT")
    edit_bones = arm_obj.data.edit_bones
    created: Dict[str, bpy.types.EditBone] = {}

    root_name = str(root_node.get("name", "ArmatureRoot"))
    root_pos = _vector3(tuple(root_node.get("position", (0.0, 0.0, 0.0))))

    root_bone = edit_bones.new(root_name)
    root_bone.head = root_pos
    root_bone.tail = root_pos + Vector((0.0, 0.1, 0.0))
    created[root_name] = root_bone

    def add_children(node: Dict, parent_bone_name: str, parent_head_world: Vector) -> None:
        children: List[Dict] = list(node.get("children", []))
        for child in children:
            child_name = str(child.get("name"))
            child_offset = _vector3(tuple(child.get("position", (0.0, 0.0, 0.0))))
            child_head_world = parent_head_world + child_offset

            child_bone = edit_bones.new(child_name)
            child_bone.head = child_head_world

            if child.get("children"):
                first_grandchild = child["children"][0]
                first_offset = _vector3(tuple(first_grandchild.get("position", (0.0, 0.1, 0.0))))
                tail_direction = first_offset if first_offset.length > 1e-8 else Vector((0.0, 0.1, 0.0))
            else:
                tail_direction = Vector((0.0, 0.1, 0.0))

            child_bone.tail = child_head_world + tail_direction
            child_bone.parent = edit_bones[parent_bone_name]
            child_bone.use_connect = False

            created[child_name] = child_bone
            add_children(child, child_name, child_head_world)

    add_children(root_node, root_name, root_pos)
    return created


def _apply_object_transform(arm_obj, root_node: Dict) -> None:
    arm_obj.location = _vector3(tuple(root_node.get("position", (0.0, 0.0, 0.0))))
    arm_obj.rotation_mode = "QUATERNION"
    arm_obj.rotation_quaternion = _quat_xyzw(tuple(root_node.get("rotation", (0.0, 0.0, 0.0, 1.0))))


def _apply_pose_rotations(arm_obj, root_node: Dict) -> None:
    _ensure_mode("POSE")

    def set_node_pose(node: Dict) -> None:
        name = str(node.get("name"))
        pose_bone = arm_obj.pose.bones.get(name)
        if pose_bone is not None:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = _quat_xyzw(tuple(node.get("rotation", (0.0, 0.0, 0.0, 1.0))))

        for child in list(node.get("children", [])):
            set_node_pose(child)

    set_node_pose(root_node)


def create_rig_from_rig_data(raw_text: str, armature_name: Optional[str] = None):
    data = parse_rig_data(raw_text)
    if "name" not in data:
        raise ValueError("Parsed rig data does not contain a root 'name'.")

    arm_name = armature_name or str(data["name"])
    arm_obj = _create_or_reset_armature(arm_name)

    _apply_object_transform(arm_obj, data)
    _build_edit_bones(arm_obj, data)
    _apply_pose_rotations(arm_obj, data)
    _ensure_mode("OBJECT")

    return arm_obj


def main() -> None:
    arm_obj = create_rig_from_rig_data(rig_data)
    print(f"Rig created: {arm_obj.name}")
    arm_obj_whale = create_rig_from_rig_data(rig_data_whale)
    print(f"Rig created: {arm_obj_whale.name}")


if __name__ == "__main__":
    main()
