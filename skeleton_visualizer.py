from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .mesh_renderer import RenderedFrame
except ImportError:  # pragma: no cover - direct script fallback
    from mesh_renderer import RenderedFrame

try:
    from .skeleton_recorder import get_recorded_bone_edges, get_recorded_frame_numbers
except ImportError:  # pragma: no cover - direct script fallback
    from skeleton_recorder import get_recorded_bone_edges, get_recorded_frame_numbers


FrameDict = Dict[str, Sequence[float]]

FRONT_VIEW = (0, 2)  # XZ
RIGHT_VIEW = (1, 2)  # YZ
TOP_VIEW = (0, 1)  # XY


class SkeletonVisualizer:
    def render_collage_sequence(self, frames: List[FrameDict]) -> List[RenderedFrame]:
        if not frames:
            return []

        sampled_indices = _sample_frame_indices(len(frames))
        sampled_frames = [frames[index] for index in sampled_indices]
        frame_numbers = get_recorded_frame_numbers()
        print(f"[Visualizer] Sampled {len(sampled_frames)} frames")

        front_scale = _compute_global_scale(sampled_frames, FRONT_VIEW)
        right_scale = _compute_global_scale(sampled_frames, RIGHT_VIEW)
        top_scale = _compute_global_scale(sampled_frames, TOP_VIEW)
        perspective_scale = _compute_global_perspective_scale(sampled_frames)
        edges = get_recorded_bone_edges()

        rendered_frames: List[RenderedFrame] = []
        for source_index, frame in zip(sampled_indices, sampled_frames):
            frame_label = (
                frame_numbers[source_index]
                if len(frame_numbers) > source_index
                else source_index
            )

            print("[Visualizer] Rendering FRONT view")
            front_image = _render_frame(
                frame=frame,
                view_name="FRONT",
                edges=edges,
                projection_axes=FRONT_VIEW,
                scale=front_scale,
            )

            print("[Visualizer] Rendering RIGHT view")
            right_image = _render_frame(
                frame=frame,
                view_name="RIGHT",
                edges=edges,
                projection_axes=RIGHT_VIEW,
                scale=right_scale,
            )

            print("[Visualizer] Rendering TOP view")
            top_image = _render_frame(
                frame=frame,
                view_name="TOP",
                edges=edges,
                projection_axes=TOP_VIEW,
                scale=top_scale,
            )

            print("[Visualizer] Rendering PERSPECTIVE view")
            perspective_image = _render_frame(
                frame=frame,
                view_name="PERSPECTIVE",
                edges=edges,
                projection_axes=None,
                scale=perspective_scale,
            )

            rendered_frames.append(
                RenderedFrame(
                    frame_index=int(frame_label),
                    collage=_create_multiview_collage(
                        views=[
                            front_image,
                            right_image,
                            top_image,
                            perspective_image,
                        ],
                        frame_label=int(frame_label),
                    ),
                )
            )
            print(f"[Visualizer] Created collage for frame {frame_label}")

        return rendered_frames


def render_skeleton_images(frames: List[FrameDict]) -> List[np.ndarray]:
    return [frame.collage for frame in SkeletonVisualizer().render_collage_sequence(frames)]


def _create_multiview_collage(
    views: Sequence[np.ndarray],
    frame_label: int,
) -> np.ndarray:
    if len(views) != 4:
        raise ValueError("Skeleton collage requires exactly four views.")

    view_height, view_width = views[0].shape[:2]
    header_height = 56
    collage = np.zeros(
        ((view_height * 2) + header_height, view_width * 2, 3),
        dtype=np.uint8,
    )

    placements = [
        (0, header_height),
        (view_width, header_height),
        (0, header_height + view_height),
        (view_width, header_height + view_height),
    ]
    for image, (x, y) in zip(views, placements):
        collage[y : y + view_height, x : x + view_width] = image

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
        (view_width, header_height),
        (view_width, header_height + (view_height * 2)),
        divider_color,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        collage,
        (0, header_height + view_height),
        (view_width * 2, header_height + view_height),
        divider_color,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        collage,
        (0, header_height),
        (view_width * 2, header_height),
        divider_color,
        1,
        lineType=cv2.LINE_AA,
    )

    return collage


def _sample_frame_indices(frame_count: int) -> List[int]:
    if frame_count <= 0:
        return []

    sampled = np.linspace(0, frame_count - 1, num=min(12, frame_count), dtype=int)
    return sorted({int(index) for index in sampled})


def _choose_projection_axes(frames: List[FrameDict]) -> Tuple[int, int]:
    points = _flatten_points(frames)
    if points.size == 0:
        return (0, 2)

    variances = np.var(points, axis=0)
    chosen = np.argsort(variances)[-2:]
    chosen = sorted(int(axis) for axis in chosen)
    return (chosen[0], chosen[1])


def _compute_global_scale(frames: List[FrameDict], projection_axes: Tuple[int, int]) -> float:
    points = _flatten_points(frames)
    if points.size == 0:
        return 1.0

    projected = points[:, projection_axes]
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    span = float(np.max(maxs - mins))
    return span if span > 1e-6 else 1.0


def _compute_global_perspective_scale(frames: List[FrameDict]) -> float:
    points = _flatten_points(frames)
    if points.size == 0:
        return 1.0

    projected = _project_perspective_points(points)
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    span = float(np.max(maxs - mins))
    return span if span > 1e-6 else 1.0


def _flatten_points(frames: List[FrameDict]) -> np.ndarray:
    all_points: List[np.ndarray] = []
    for frame in frames:
        for point in frame.values():
            if len(point) >= 3:
                all_points.append(np.asarray(point[:3], dtype=np.float32))

    if not all_points:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(all_points)


def _render_frame(
    frame: FrameDict,
    view_name: str,
    edges: List[Tuple[str, str]],
    projection_axes: Optional[Tuple[int, int]],
    scale: float,
) -> np.ndarray:
    canvas_size = 512
    margin = 48.0
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    points_2d: Dict[str, Tuple[int, int]] = {}
    if frame:
        positions = np.asarray([frame[name][:3] for name in frame], dtype=np.float32)
        if projection_axes is None:
            projected = _project_perspective_points(positions)
        else:
            projected = positions[:, projection_axes]
        center = projected.mean(axis=0)
        normalized = projected - center
        usable_size = canvas_size - (2.0 * margin)
        scale_factor = usable_size / max(scale, 1.0)

        for joint_name, coords in zip(frame.keys(), normalized):
            x = int(round((coords[0] * scale_factor) + (canvas_size / 2.0)))
            y = int(round((canvas_size / 2.0) - (coords[1] * scale_factor)))
            points_2d[joint_name] = (x, y)

    for parent_name, child_name in edges:
        parent_point = points_2d.get(parent_name)
        child_point = points_2d.get(child_name)
        if parent_point is None or child_point is None:
            continue
        cv2.line(canvas, parent_point, child_point, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    for point in points_2d.values():
        cv2.circle(canvas, point, 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        view_name,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    return canvas


def _project_perspective_points(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    projected_x = (x - y) * 0.7071
    projected_y = (z * 0.9) - ((x + y) * 0.35)
    return np.stack([projected_x, projected_y], axis=1)
