from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from .skeleton_recorder import get_recorded_bone_edges, get_recorded_frame_numbers
except ImportError:  # pragma: no cover - direct script fallback
    from skeleton_recorder import get_recorded_bone_edges, get_recorded_frame_numbers


FrameDict = Dict[str, Sequence[float]]

FRONT_VIEW = (0, 2)  # XZ
SIDE_VIEW = (1, 2)  # YZ
TOP_VIEW = (0, 1)  # XY


def render_skeleton_images(frames: List[FrameDict]) -> List[np.ndarray]:
    if not frames:
        return []

    sampled_indices = _sample_frame_indices(len(frames))
    sampled_frames = [frames[index] for index in sampled_indices]
    frame_numbers = get_recorded_frame_numbers()
    print(f"[Visualizer] Sampled {len(sampled_frames)} frames")

    front_scale = _compute_global_scale(sampled_frames, FRONT_VIEW)
    side_scale = _compute_global_scale(sampled_frames, SIDE_VIEW)
    top_scale = _compute_global_scale(sampled_frames, TOP_VIEW)
    edges = get_recorded_bone_edges()

    images: List[np.ndarray] = []
    for source_index, frame in zip(sampled_indices, sampled_frames):
        frame_label = frame_numbers[source_index] if len(frame_numbers) > source_index else source_index

        print("[Visualizer] Rendering FRONT view")
        front_image = _render_frame(
            frame=frame,
            frame_label=frame_label,
            view_name="FRONT",
            edges=edges,
            projection_axes=FRONT_VIEW,
            scale=front_scale,
        )

        print("[Visualizer] Rendering SIDE view")
        side_image = _render_frame(
            frame=frame,
            frame_label=frame_label,
            view_name="SIDE",
            edges=edges,
            projection_axes=SIDE_VIEW,
            scale=side_scale,
        )

        print("[Visualizer] Rendering TOP view")
        top_image = _render_frame(
            frame=frame,
            frame_label=frame_label,
            view_name="TOP",
            edges=edges,
            projection_axes=TOP_VIEW,
            scale=top_scale,
        )

        images.append(
            _create_multiview_collage(
                front_image=front_image,
                side_image=side_image,
                top_image=top_image,
                frame_label=frame_label,
            )
        )
        print(f"[Visualizer] Created collage for frame {frame_label}")

    return images


def _create_multiview_collage(
    front_image: np.ndarray,
    side_image: np.ndarray,
    top_image: np.ndarray,
    frame_label: int,
) -> np.ndarray:
    view_height, view_width = front_image.shape[:2]
    header_height = 56
    collage = np.zeros(
        ((view_height * 2) + header_height, view_width * 2, 3),
        dtype=np.uint8,
    )

    collage[header_height : header_height + view_height, 0:view_width] = front_image
    collage[header_height : header_height + view_height, view_width : view_width * 2] = side_image
    collage[
        header_height + view_height : header_height + (view_height * 2),
        0:view_width,
    ] = top_image

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
    frame_label: int,
    view_name: str,
    edges: List[Tuple[str, str]],
    projection_axes: Tuple[int, int],
    scale: float,
) -> np.ndarray:
    canvas_size = 512
    margin = 48.0
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    points_2d: Dict[str, Tuple[int, int]] = {}
    if frame:
        positions = np.asarray([frame[name][:3] for name in frame], dtype=np.float32)
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
