from __future__ import annotations

from typing import Iterable, List, Optional

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - only available inside Blender
    bpy = None


class SceneParser:
    def __init__(self, context=None, precision: int = 4) -> None:
        self.context = context
        self.precision = precision

    def generate_object_json(self, selected_objects: Optional[Iterable[object]] = None) -> str:
        if bpy is None:
            raise RuntimeError("SceneParser must be run inside Blender because bpy is unavailable.")

        objects = self._resolve_objects(selected_objects)
        if not objects:
            raise ValueError("No objects were provided to generate_object_json().")

        return ",".join(self._serialize_object(obj) for obj in objects)

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
        local_matrix = obj.matrix_local.copy()
        position = local_matrix.to_translation()
        rotation = self._object_quaternion(obj, local_matrix)

        children = []
        if obj.type == "ARMATURE":
            children = [self._serialize_bone(bone) for bone in obj.data.bones if bone.parent is None]
        elif getattr(obj, "children", None):
            children = [self._serialize_object(child) for child in obj.children]

        return self._serialize_node(obj.name, position, rotation, children)

    def _serialize_bone(self, bone: object) -> str:
        local_matrix = bone.matrix_local.copy()
        if bone.parent is not None:
            local_matrix = bone.parent.matrix_local.inverted() @ bone.matrix_local

        position = local_matrix.to_translation()
        rotation = local_matrix.to_quaternion()
        children = [self._serialize_bone(child) for child in bone.children]

        return self._serialize_node(bone.name, position, rotation, children)

    def _object_quaternion(self, obj: object, fallback_matrix) -> object:
        rotation_mode = getattr(obj, "rotation_mode", "QUATERNION")
        if rotation_mode == "QUATERNION":
            return obj.rotation_quaternion.copy()
        return fallback_matrix.to_quaternion()

    def _serialize_node(self, name: str, position, rotation, children: List[str]) -> str:
        node = (
            f"name:{name},"
            f"position:{self._format_tuple(position)},"
            f"rotation:{self._format_tuple(rotation)}"
        )
        if children:
            node += f",children:[{','.join(children)}]"
        return node

    def _format_tuple(self, values) -> str:
        formatted_values = ",".join(self._format_float(value) for value in values)
        return f"({formatted_values})"

    def _format_float(self, value: float) -> str:
        if abs(value) < 10 ** (-self.precision):
            value = 0.0
        return f"{value:.{self.precision}f}"
