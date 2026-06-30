from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

try:
    from .multimodal_utils import compress_image_for_llm
except ImportError:  # pragma: no cover - direct script fallback
    from multimodal_utils import compress_image_for_llm

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ImportError:  # pragma: no cover - Blender-only imports
    bpy = None
    Vector = None


@dataclass
class RenderedImage:
    label: str
    data_url: str
    camera_metadata: Dict[str, Any]


@dataclass
class RenderedFrame:
    frame_index: int
    collage: np.ndarray


VisualEvidence = Dict[str, Any]


class MeshRenderer:
    def __init__(self, context=None, scene=None) -> None:
        self.context = context
        self.scene = scene

    def render_views_single_frame(
        self,
        objects: Sequence[object],
        resolution: int = 768,
    ) -> List[RenderedImage]:
        self._require_blender()
        rendered_views = self._render_view_images_for_objects(
            selected_objects=list(objects or []),
            resolution=resolution,
        )
        return [
            RenderedImage(
                label=label,
                data_url=compress_image_for_llm(image),
                camera_metadata=camera_metadata,
            )
            for label, image, camera_metadata in rendered_views
        ]

    def render_collage_sequence(
        self,
        frame_objects: Sequence[Sequence[object]],
        resolution: int = 768,
    ) -> List[RenderedFrame]:
        self._require_blender()
        if not frame_objects:
            return []

        scene = self._scene()
        original_frame = int(scene.frame_current)
        frame_numbers = self._sample_frame_numbers(frame_objects)
        rendered_frames: List[RenderedFrame] = []

        try:
            for frame_number, objects in zip(frame_numbers, frame_objects):
                scene.frame_set(int(frame_number))
                bpy.context.view_layer.update()
                collage = self._render_collage_for_objects(
                    selected_objects=list(objects or []),
                    resolution=resolution,
                    frame_index=int(frame_number),
                )
                if collage is None:
                    continue
                rendered_frames.append(
                    RenderedFrame(
                        frame_index=int(frame_number),
                        collage=collage,
                    )
                )
        finally:
            scene.frame_set(original_frame)
            bpy.context.view_layer.update()

        return rendered_frames

    def _render_collage_for_objects(
        self,
        selected_objects: Sequence[object],
        resolution: int,
        frame_index: int,
    ) -> Optional[np.ndarray]:
        rendered_views = self._render_view_images_for_objects(selected_objects, resolution)
        if not rendered_views:
            return None
        view_images = {label: image for label, image, _metadata in rendered_views}
        return _create_multiview_collage(
            views=[
                ("front", view_images["front"]),
                ("right", view_images["right"]),
                ("top", view_images["top"]),
                ("perspective", view_images["perspective"]),
            ],
            frame_label=frame_index,
        )

    def _render_view_images_for_objects(
        self,
        selected_objects: Sequence[object],
        resolution: int,
    ) -> List[tuple[str, np.ndarray, Dict[str, Any]]]:
        render_objects = self._find_render_meshes(selected_objects)
        if not render_objects:
            return []

        center, span = self._compute_bounds(render_objects)
        distance = max(span * 2.75, 1.0)
        ortho_scale = max(span * 1.35, 1.0)
        view_specs = [
            ("front", Vector((0.0, -distance, 0.0)), True),
            ("right", Vector((distance, 0.0, 0.0)), True),
            ("top", Vector((0.0, 0.0, distance)), True),
            ("perspective", Vector((-distance, -distance, distance * 0.8)), False),
        ]

        scene = self._scene()
        original_camera = scene.camera
        original_resolution = (
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.film_transparent,
        )
        original_filepath = scene.render.filepath
        original_hide_render = {obj.name: obj.hide_render for obj in bpy.data.objects}
        camera = None
        camera_data = None
        light = None
        light_data = None
        renders: List[tuple[str, np.ndarray, Dict[str, Any]]] = []

        try:
            camera_data = bpy.data.cameras.new("MeshRendererCamera")
            camera = bpy.data.objects.new("MeshRendererCamera", camera_data)
            light_data = bpy.data.lights.new("MeshRendererLight", type="AREA")
            light = bpy.data.objects.new("MeshRendererLight", light_data)
            scene.collection.objects.link(camera)
            scene.collection.objects.link(light)
            light.data.energy = 450.0
            light.data.size = max(span * 1.5, 1.0)
            light.location = center + Vector((distance * 0.25, -distance * 0.35, distance))
            scene.camera = camera
            scene.render.resolution_x = int(resolution)
            scene.render.resolution_y = int(resolution)
            scene.render.film_transparent = False

            allowed = {obj.name for obj in render_objects}
            allowed.update({camera.name, light.name})
            for obj in bpy.data.objects:
                obj.hide_render = obj.name not in allowed

            with tempfile.TemporaryDirectory(prefix="mesh_renderer_") as temp_dir:
                for label, offset, orthographic in view_specs:
                    camera.location = center + offset
                    self._look_at(camera, center)
                    camera.data.type = "ORTHO" if orthographic else "PERSP"
                    camera.data.ortho_scale = ortho_scale
                    camera.data.lens = 70
                    scene.render.filepath = os.path.join(temp_dir, f"{label}.png")
                    bpy.ops.render.render(write_still=True)
                    image = cv2.imread(scene.render.filepath, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Failed to read mesh render `{scene.render.filepath}`.")
                    renders.append((label, image, self._camera_metadata(camera, label)))
        finally:
            scene.camera = original_camera
            scene.render.resolution_x = original_resolution[0]
            scene.render.resolution_y = original_resolution[1]
            scene.render.film_transparent = original_resolution[2]
            scene.render.filepath = original_filepath
            for obj in bpy.data.objects:
                if obj.name in original_hide_render:
                    obj.hide_render = original_hide_render[obj.name]
            if camera is not None:
                bpy.data.objects.remove(camera, do_unlink=True)
            if camera_data is not None:
                bpy.data.cameras.remove(camera_data, do_unlink=True)
            if light is not None:
                bpy.data.objects.remove(light, do_unlink=True)
            if light_data is not None:
                bpy.data.lights.remove(light_data, do_unlink=True)

        return renders

    def _require_blender(self) -> None:
        if bpy is None or Vector is None:
            raise RuntimeError("MeshRenderer must run inside Blender.")

    def _scene(self):
        if self.scene is not None:
            return self.scene
        context = self.context or bpy.context
        if hasattr(context, "render") and hasattr(context, "frame_set"):
            return context
        return context.scene

    def _sample_frame_numbers(self, frame_objects: Sequence[Sequence[object]]) -> List[int]:
        count = len(frame_objects)
        if count <= 0:
            return []

        start, end = self._frame_range_for_objects(
            [obj for objects in frame_objects for obj in objects]
        )
        if count == 1:
            return [start]
        if end <= start:
            return [start for _ in range(count)]

        step = (end - start) / float(count - 1)
        sampled = [int(round(start + (step * index))) for index in range(count)]
        deduped: List[int] = []
        for frame_number in sampled:
            if not deduped or deduped[-1] != frame_number:
                deduped.append(frame_number)
        while len(deduped) < count:
            deduped.append(deduped[-1] if deduped else start)
        return deduped[:count]

    def _frame_range_for_objects(self, objects: Sequence[object]) -> tuple[int, int]:
        starts: List[int] = []
        ends: List[int] = []

        for obj in objects:
            for candidate in self._iter_objects_with_actions(obj):
                action = getattr(getattr(candidate, "animation_data", None), "action", None)
                if action is None:
                    continue
                starts.append(int(action.frame_range[0]))
                ends.append(int(action.frame_range[1]))

        scene = self._scene()
        if not starts or not ends:
            return int(scene.frame_start), int(scene.frame_end)
        return min(starts), max(ends)

    def _iter_objects_with_actions(self, obj: object) -> Iterable[object]:
        if obj is None:
            return
        yield obj
        for child in getattr(obj, "children_recursive", []):
            yield child

    def _find_render_meshes(self, objects: Sequence[object]) -> List[object]:
        meshes: List[object] = []
        seen = set()

        def add_mesh(obj):
            if obj is not None and getattr(obj, "type", None) == "MESH" and obj.name not in seen:
                meshes.append(obj)
                seen.add(obj.name)

        selected_names = {obj.name for obj in objects if hasattr(obj, "name")}
        for obj in objects:
            add_mesh(obj)
            for child in getattr(obj, "children_recursive", []):
                add_mesh(child)

            if getattr(obj, "type", None) == "ARMATURE":
                for candidate in bpy.data.objects:
                    if getattr(candidate, "type", None) != "MESH":
                        continue
                    if getattr(candidate, "parent", None) == obj:
                        add_mesh(candidate)
                        continue
                    for modifier in getattr(candidate, "modifiers", []):
                        if (
                            getattr(modifier, "type", None) == "ARMATURE"
                            and getattr(modifier, "object", None) == obj
                        ):
                            add_mesh(candidate)
                            break

        if not meshes:
            for obj in self._scene().objects:
                if getattr(obj, "type", None) == "MESH" and getattr(obj, "parent", None):
                    if obj.parent.name in selected_names:
                        add_mesh(obj)

        return meshes

    def _compute_bounds(self, objects: Sequence[object]):
        points = []
        for obj in objects:
            for corner in obj.bound_box:
                points.append(obj.matrix_world @ Vector(corner))

        if not points:
            return Vector((0.0, 0.0, 0.0)), 1.0

        min_corner = Vector(
            (
                min(point.x for point in points),
                min(point.y for point in points),
                min(point.z for point in points),
            )
        )
        max_corner = Vector(
            (
                max(point.x for point in points),
                max(point.y for point in points),
                max(point.z for point in points),
            )
        )
        center = (min_corner + max_corner) * 0.5
        span_vector = max_corner - min_corner
        span = max(span_vector.x, span_vector.y, span_vector.z, 1.0)
        return center, float(span)

    def _look_at(self, camera, target) -> None:
        direction = target - camera.location
        if direction.length < 1e-6:
            direction = Vector((0.0, 0.0, -1.0))
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    def _camera_metadata(self, camera, label: str) -> Dict[str, Any]:
        rotation = camera.rotation_euler
        world_quaternion = camera.matrix_world.to_quaternion()
        return {
            "render_label": label,
            "location": self._vector_to_list(camera.location),
            "rotation_euler": [float(rotation.x), float(rotation.y), float(rotation.z)],
            "forward_direction": self._vector_to_list(world_quaternion @ Vector((0.0, 0.0, -1.0))),
            "up_direction": self._vector_to_list(world_quaternion @ Vector((0.0, 1.0, 0.0))),
            "right_direction": self._vector_to_list(world_quaternion @ Vector((1.0, 0.0, 0.0))),
            "projection": camera.data.type,
        }

    def _vector_to_list(self, vector) -> List[float]:
        return [round(float(vector.x), 6), round(float(vector.y), 6), round(float(vector.z), 6)]


def build_visual_evidence(
    prompt: str,
    mesh_collage_sequence: Sequence[RenderedFrame],
    skeleton_collage_sequence: Optional[Sequence[RenderedFrame]] = None,
) -> VisualEvidence:
    mesh_frames = list(mesh_collage_sequence or [])
    skeleton_frames = list(skeleton_collage_sequence or [])

    return {
        "prompt": str(prompt or ""),
        "mesh_collages": [frame.collage for frame in mesh_frames],
        "skeleton_collages": [frame.collage for frame in skeleton_frames],
        "mesh_frame_indices": [int(frame.frame_index) for frame in mesh_frames],
        "skeleton_frame_indices": [int(frame.frame_index) for frame in skeleton_frames],
    }


def _create_multiview_collage(
    views: Sequence[tuple[str, np.ndarray]],
    frame_label: int,
) -> np.ndarray:
    if len(views) != 4:
        raise ValueError("Mesh collage requires exactly four views.")

    normalized_views = []
    first_height, first_width = views[0][1].shape[:2]
    for label, image in views:
        if image.shape[:2] != (first_height, first_width):
            image = cv2.resize(image, (first_width, first_height), interpolation=cv2.INTER_AREA)
        normalized_views.append((label, image.copy()))

    header_height = 56
    collage = np.zeros(
        ((first_height * 2) + header_height, first_width * 2, 3),
        dtype=np.uint8,
    )
    positions = [
        (0, header_height),
        (first_width, header_height),
        (0, header_height + first_height),
        (first_width, header_height + first_height),
    ]

    for (label, image), (x, y) in zip(normalized_views, positions):
        _draw_label(image, label.upper())
        collage[y : y + first_height, x : x + first_width] = image

    cv2.putText(
        collage,
        f"Frame {frame_label}",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    divider_color = (80, 80, 80)
    cv2.line(
        collage,
        (first_width, header_height),
        (first_width, header_height + (first_height * 2)),
        divider_color,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        collage,
        (0, header_height + first_height),
        (first_width * 2, header_height + first_height),
        divider_color,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        collage,
        (0, header_height),
        (first_width * 2, header_height),
        divider_color,
        1,
        lineType=cv2.LINE_AA,
    )
    return collage


def _draw_label(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (8, 8), (220, 44), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        label,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
