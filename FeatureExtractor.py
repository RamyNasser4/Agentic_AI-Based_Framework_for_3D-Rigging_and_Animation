from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from paramUtil import t2m_kinematic_chain, t2m_raw_offsets, t2m_tgt_skel_id
from utils import bvh, quat


FEATURE_DIM = 263
FEET_THRE = 0.002
L_IDX1, L_IDX2 = 5, 8
FID_R, FID_L = [8, 11], [7, 10]
FACE_JOINT_INDX = [2, 1, 17, 16]
N_RAW_OFFSETS = np.asarray(t2m_raw_offsets, dtype=np.float32)
KINEMATIC_CHAIN = t2m_kinematic_chain
HUMANML3D_JOINT_ORDER = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
JOINT_ALIASES = {
    "pelvis": ("pelvis", "hips", "hip", "root"),
    "left_hip": ("left_hip", "lefthip", "lhip", "leftupleg"),
    "right_hip": ("right_hip", "righthip", "rhip", "rightupleg"),
    "spine1": ("spine1", "spine", "lowerback"),
    "left_knee": ("left_knee", "leftknee", "leftleg"),
    "right_knee": ("right_knee", "rightknee", "rightleg"),
    "spine2": ("spine2", "spine01", "midspine"),
    "left_ankle": ("left_ankle", "leftankle"),
    "right_ankle": ("right_ankle", "rightankle"),
    "spine3": ("spine3", "spine02", "chest", "upperchest"),
    "left_foot": ("left_foot", "leftfoot", "lefttoebase"),
    "right_foot": ("right_foot", "rightfoot", "righttoebase"),
    "neck": ("neck", "neck1"),
    "left_collar": ("left_collar", "leftclavicle", "leftcollar"),
    "right_collar": ("right_collar", "rightclavicle", "rightcollar"),
    "head": ("head",),
    "left_shoulder": ("left_shoulder", "leftshoulder", "leftarm"),
    "right_shoulder": ("right_shoulder", "rightshoulder", "rightarm"),
    "left_elbow": ("left_elbow", "leftforearm"),
    "right_elbow": ("right_elbow", "rightforearm"),
    "left_wrist": ("left_wrist", "lefthand"),
    "right_wrist": ("right_wrist", "righthand"),
}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _safe_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return v / norm


def _qinv_np(q: np.ndarray) -> np.ndarray:
    mask = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    return q * mask


def _qmul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return quat.mul(q, r).astype(np.float32, copy=False)


def _qrot_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat.mul_vec(q, v).astype(np.float32, copy=False)


def _qbetween_np(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    return quat.normalize(quat.between(v0, v1)).astype(np.float32, copy=False)


def _qfix(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def _quaternion_to_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)

    matrix = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3)).astype(np.float32, copy=False)


def _quaternion_to_cont6d_np(quaternions: np.ndarray) -> np.ndarray:
    rotation_mat = _quaternion_to_matrix_np(quaternions)
    return np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1).astype(
        np.float32,
        copy=False,
    )


class NumpySkeleton:
    def __init__(self, offset: np.ndarray, kinematic_tree: list[list[int]]):
        self._raw_offset_np = np.asarray(offset, dtype=np.float32)
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset_np)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for joint_idx in range(1, len(chain)):
                self._parents[chain[joint_idx]] = chain[joint_idx - 1]

    def get_offsets_joints(self, joints: np.ndarray) -> np.ndarray:
        offsets = self._raw_offset_np.copy()
        for joint_idx in range(1, self._raw_offset_np.shape[0]):
            parent_idx = self._parents[joint_idx]
            offsets[joint_idx] = (
                np.linalg.norm(joints[joint_idx] - joints[parent_idx]) * offsets[joint_idx]
            )
        self._offset = offsets.astype(np.float32, copy=False)
        return self._offset

    def set_offset(self, offsets: np.ndarray) -> None:
        self._offset = np.asarray(offsets, dtype=np.float32)

    def inverse_kinematics_np(
        self,
        joints: np.ndarray,
        face_joint_idx: list[int],
        smooth_forward: bool = False,
    ) -> np.ndarray:
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = _safe_normalize(across)

        forward = np.cross(np.array([[0, 1, 0]], dtype=np.float32), across, axis=-1)
        if smooth_forward:
            forward = gaussian_filter1d(forward, 20, axis=0, mode="nearest")
        forward = _safe_normalize(forward)

        target = np.array([[0, 0, 1]], dtype=np.float32).repeat(len(forward), axis=0)
        root_quat = _qbetween_np(forward, target)

        quat_params = np.zeros(joints.shape[:-1] + (4,), dtype=np.float32)
        root_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        quat_params[:, 0] = root_quat

        for chain in self._kinematic_tree:
            rotation = root_quat
            for joint_idx in range(len(chain) - 1):
                u = self._raw_offset_np[chain[joint_idx + 1]][np.newaxis, ...].repeat(
                    len(joints),
                    axis=0,
                )
                v = joints[:, chain[joint_idx + 1]] - joints[:, chain[joint_idx]]
                v = _safe_normalize(v)
                rot_u_v = _qbetween_np(u, v)
                local_rotation = _qmul_np(_qinv_np(rotation), rot_u_v)
                quat_params[:, chain[joint_idx + 1], :] = local_rotation
                rotation = _qmul_np(rotation, local_rotation)

        return quat_params

    def forward_kinematics_np(
        self,
        quat_params: np.ndarray,
        root_pos: np.ndarray,
        skel_joints: np.ndarray | None = None,
        do_root_R: bool = True,
    ) -> np.ndarray:
        if skel_joints is not None:
            offsets = self.get_offsets_joints(skel_joints[0] if skel_joints.ndim == 3 else skel_joints)
        else:
            offsets = self._offset
        if offsets is None:
            offsets = self._raw_offset_np

        joints = np.zeros(quat_params.shape[:-1] + (3,), dtype=np.float32)
        joints[:, 0] = root_pos
        offsets_batch = np.repeat(offsets[np.newaxis, ...], quat_params.shape[0], axis=0)
        identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32).repeat(len(quat_params), axis=0)

        for chain in self._kinematic_tree:
            rotation = quat_params[:, 0] if do_root_R else identity
            for joint_idx in range(1, len(chain)):
                rotation = _qmul_np(rotation, quat_params[:, chain[joint_idx]])
                offset_vec = offsets_batch[:, chain[joint_idx]]
                joints[:, chain[joint_idx]] = _qrot_np(rotation, offset_vec) + joints[:, chain[joint_idx - 1]]

        return joints


def _sanitize_positions(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions with shape (T, J, 3), got {positions.shape}.")

    if positions.size == 0 or np.isfinite(positions).all():
        return positions

    flat = positions.reshape(positions.shape[0], -1)
    frame_ids = np.arange(flat.shape[0], dtype=np.float32)
    for col in range(flat.shape[1]):
        values = flat[:, col]
        valid = np.isfinite(values)
        if valid.all():
            continue
        if not valid.any():
            flat[:, col] = 0.0
            continue
        flat[~valid, col] = np.interp(frame_ids[~valid], frame_ids[valid], values[valid])
    return flat.reshape(positions.shape)


@lru_cache(maxsize=1)
def _get_target_offsets() -> np.ndarray:
    example_path = (
        Path(__file__).resolve().parent
        / "HumanML3D"
        / "humanml"
        / "joints"
        / f"{t2m_tgt_skel_id}.npy"
    )
    if not example_path.exists():
        raise FileNotFoundError(f"HumanML3D example joints file not found: {example_path}")

    example_data = np.load(example_path)
    tgt_skel = NumpySkeleton(N_RAW_OFFSETS, KINEMATIC_CHAIN)
    return tgt_skel.get_offsets_joints(np.asarray(example_data[0, :22], dtype=np.float32))


def _get_humanml3d_indices(names: list[str]) -> list[int]:
    normalized_to_index = {}
    for idx, name in enumerate(names):
        normalized_to_index.setdefault(_normalize_name(name), idx)

    indices = []
    missing = []
    for joint_name in HUMANML3D_JOINT_ORDER:
        candidates = JOINT_ALIASES[joint_name]
        joint_idx = None
        for candidate in candidates:
            joint_idx = normalized_to_index.get(_normalize_name(candidate))
            if joint_idx is not None:
                break
        if joint_idx is None:
            missing.append(joint_name)
        else:
            indices.append(joint_idx)

    if missing:
        raise ValueError(
            "Could not map BVH joints to HumanML3D 22-joint order. "
            f"Missing joints: {missing}. Available joints: {names}"
        )
    return indices


def _load_bvh_global_positions(bvh_path: str) -> tuple[np.ndarray, list[str]]:
    motion = bvh.load(bvh_path)
    local_rotations = quat.from_euler(
        np.deg2rad(np.asarray(motion["rotations"], dtype=np.float32)),
        order=motion["order"],
    )
    local_positions = np.asarray(motion["positions"], dtype=np.float32)
    _, global_positions = quat.fk(local_rotations, local_positions, motion["parents"])
    return np.asarray(global_positions, dtype=np.float32), motion["names"]


def uniform_skeleton(positions: np.ndarray, target_offset: np.ndarray) -> np.ndarray:
    src_skel = NumpySkeleton(N_RAW_OFFSETS, KINEMATIC_CHAIN)
    src_offset = src_skel.get_offsets_joints(np.asarray(positions[0], dtype=np.float32))
    tgt_offset = np.asarray(target_offset, dtype=np.float32)

    src_leg_len = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg_len = np.abs(tgt_offset[L_IDX1]).max() + np.abs(tgt_offset[L_IDX2]).max()
    scale_rt = 1.0 if not np.isfinite(src_leg_len) or src_leg_len < 1e-8 else (tgt_leg_len / src_leg_len)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_INDX)
    src_skel.set_offset(target_offset)
    return src_skel.forward_kinematics_np(quat_params, tgt_root_pos)


def process_file(positions: np.ndarray, feet_thre: float):
    if len(positions) < 2:
        empty = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        positions = np.asarray(positions, dtype=np.float32)
        return empty, positions.copy(), positions.copy(), np.zeros((0, 2), dtype=np.float32)

    positions = uniform_skeleton(positions, _get_target_offsets())

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1], dtype=np.float32)
    positions = positions - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = _safe_normalize(across)

    forward_init = np.cross(np.array([[0, 1, 0]], dtype=np.float32), across, axis=-1)
    forward_init = _safe_normalize(forward_init)

    target = np.array([[0, 0, 1]], dtype=np.float32)
    root_quat_init = _qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,), dtype=np.float32) * root_quat_init

    positions = _qrot_np(root_quat_init, positions)
    global_positions = positions.copy()

    def foot_detect(positions_: np.ndarray, thres: float):
        velfactor = np.array([thres, thres], dtype=np.float32)

        feet_l_x = (positions_[1:, FID_L, 0] - positions_[:-1, FID_L, 0]) ** 2
        feet_l_y = (positions_[1:, FID_L, 1] - positions_[:-1, FID_L, 1]) ** 2
        feet_l_z = (positions_[1:, FID_L, 2] - positions_[:-1, FID_L, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions_[1:, FID_R, 0] - positions_[:-1, FID_R, 0]) ** 2
        feet_r_y = (positions_[1:, FID_R, 1] - positions_[:-1, FID_R, 1]) ** 2
        feet_r_z = (positions_[1:, FID_R, 2] - positions_[:-1, FID_R, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)
    r_rot = None

    def get_rifke(positions_: np.ndarray) -> np.ndarray:
        positions_ = positions_.copy()
        positions_[..., 0] -= positions_[:, 0:1, 0]
        positions_[..., 2] -= positions_[:, 0:1, 2]
        positions_ = _qrot_np(np.repeat(r_rot[:, None], positions_.shape[1], axis=1), positions_)
        return positions_

    def get_cont6d_params(positions_: np.ndarray):
        skel = NumpySkeleton(N_RAW_OFFSETS, KINEMATIC_CHAIN)
        quat_params = skel.inverse_kinematics_np(positions_, FACE_JOINT_INDX, smooth_forward=True)
        cont_6d_params = _quaternion_to_cont6d_np(quat_params)
        r_rot_local = quat_params[:, 0].copy()
        velocity = (positions_[1:, 0] - positions_[:-1, 0]).copy()
        velocity = _qrot_np(r_rot_local[1:], velocity)
        r_velocity = _qmul_np(r_rot_local[1:], _qinv_np(r_rot_local[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot_local

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    local_vel = _qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def bvh_to_humanml3d_features(bvh_path: str) -> np.ndarray:
    """
    Input:
        bvh_path: path to a BVH file

    Output:
        features: numpy array of shape (T - 1, 263)
    """
    bvh_file = Path(bvh_path)
    if not bvh_file.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh_file}")

    global_positions, names = _load_bvh_global_positions(str(bvh_file))
    humanml_indices = _get_humanml3d_indices(names)
    positions = global_positions[:, humanml_indices, :]
    positions = _sanitize_positions(positions)

    if positions.shape[0] < 2:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    data_263, _, _, _ = process_file(positions, feet_thre=FEET_THRE)
    data_263 = np.asarray(data_263, dtype=np.float32)

    if data_263.ndim != 2 or data_263.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected features with shape (T, {FEATURE_DIM}), got {data_263.shape}.")

    if not np.isfinite(data_263).all():
        data_263 = np.nan_to_num(data_263, nan=0.0, posinf=0.0, neginf=0.0)

    return data_263.astype(np.float32, copy=False)


def normalize_features(features, mean, std):
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    normalized = (features - mean) / std
    return normalized.astype(np.float32, copy=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from a BVH file for HumanML3D.")
    parser.add_argument("bvh_path", type=str, help="Path to the input BVH file.")
    parser.add_argument("output_path", type=str, help="Path to save the extracted features (numpy .npy file).")
    args = parser.parse_args()

    features = bvh_to_humanml3d_features(args.bvh_path)
    np.save(args.output_path, features)
    print(f"Extracted features saved to {args.output_path}")
