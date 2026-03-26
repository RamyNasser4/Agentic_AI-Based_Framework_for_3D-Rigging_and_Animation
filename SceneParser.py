from __future__ import annotations

from typing import Iterable, List, Optional

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ImportError:  # pragma: no cover - only available inside Blender
    bpy = None
    Vector = None


class SceneParser:
    def __init__(self, context=None, precision: int = 4) -> None:
        self.context = context
        self.precision = precision

    def generate_object_json(
        self,
        selected_objects: Optional[Iterable[object]] = None,
        include_root_directions: bool = True,
    ) -> str:
        if bpy is None:
            raise RuntimeError("SceneParser must be run inside Blender because bpy is unavailable.")

        objects = self._resolve_objects(selected_objects)
        if not objects:
            raise ValueError("No objects were provided to generate_object_json().")

        object_json = ",".join(self._serialize_object(obj) for obj in objects)
        if not include_root_directions:
            return object_json

        return f"{object_json}\n{self._format_root_directions(objects[0])}"

    def generate_scene_info(self, selected_objects: Optional[Iterable[object]] = None) -> str:
        return self.generate_object_json(
            selected_objects=selected_objects,
            include_root_directions=True,
        )

    def _resolve_objects(self, selected_objects: Optional[Iterable[object]]) -> List[object]:
        if bpy is None:
            raise RuntimeError("SceneParser must be run inside Blender because bpy is unavailable.")

        if selected_objects is None:
            context = self.context or bpy.context
            return list(context.selected_objects)

        resolved_objects = []
        for item in selected_objects:
            if isinstance(item, str):
                obj = bpy.data.objects.get(item)
                if obj is None:
                    raise ValueError(f"Object {item!r} was not found in the current Blender file.")
                resolved_objects.append(obj)
                continue

            if hasattr(item, "name") and hasattr(item, "type"):
                resolved_objects.append(item)
                continue

            raise TypeError(
                "selected_objects must contain Blender object references or object names."
            )

        return resolved_objects

    def _serialize_object(self, obj: object) -> str:
        basis_matrix = obj.matrix_basis.copy()
        position = basis_matrix.to_translation()
        rotation = basis_matrix.to_quaternion()

        children = []
        if obj.type == "ARMATURE":
            children = [
                self._serialize_bone(pose_bone)
                for pose_bone in obj.pose.bones
                if pose_bone.parent is None
            ]
        elif getattr(obj, "children", None):
            children = [self._serialize_object(child) for child in obj.children]

        return self._serialize_node(obj.name, position, rotation, children)

    def _serialize_bone(self, pose_bone: object) -> str:
        basis_matrix = pose_bone.matrix_basis.copy()
        position = basis_matrix.to_translation()
        rotation = basis_matrix.to_quaternion()
        children = [self._serialize_bone(child) for child in pose_bone.children]

        return self._serialize_node(pose_bone.name, position, rotation, children)

    def _object_quaternion(self, obj: object, fallback_matrix) -> object:
        rotation_mode = getattr(obj, "rotation_mode", "QUATERNION")
        if rotation_mode == "QUATERNION":
            return obj.rotation_quaternion.copy()
        return fallback_matrix.to_quaternion()

    def _serialize_node(self, name: str, position, rotation, children: List[str]) -> str:
        node = (
            f"name:{name},"
            f"position:{self._format_tuple(position)},"
            f"rotation:{self._format_quaternion_tuple(rotation)}"
        )
        if children:
            node += f",children:[{','.join(children)}]"
        return node

    def _format_tuple(self, values) -> str:
        formatted_values = ",".join(self._format_float(value) for value in values)
        return f"({formatted_values})"

    def _format_quaternion_tuple(self, quaternion) -> str:
        formatted_values = ",".join(
            self._format_float(value)
            for value in (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        )
        return f"({formatted_values})"

    def _format_root_directions(self, obj: object) -> str:
        if Vector is None:
            raise RuntimeError("SceneParser must be run inside Blender because mathutils is unavailable.")

        root_bone = obj.pose.bones["root"]
        orientation = root_bone.matrix.to_3x3()  # world-space including rest pose
        forward = (orientation @ Vector((0.0, 1.0, 0.0))).normalized()
        right = (orientation @ Vector((1.0, 0.0, 0.0))).normalized()
        up = (orientation @ Vector((0.0, 0.0, 1.0))).normalized()

        return (
            f"Root forward direction: {self._format_direction_tuple(forward)}; "
            f"right direction: {self._format_direction_tuple(right)}; "
            f"up direction: {self._format_direction_tuple(up)}"
        )

    def _format_direction_tuple(self, values) -> str:
        formatted_values = ", ".join(self._format_float(value) for value in values)
        return f"({formatted_values})"

    def _format_float(self, value: float) -> str:
        if abs(value) < 10 ** (-self.precision):
            value = 0.0
        return f"{value:.{self.precision}f}"
