"""Transformasi Raw51 -> Body110 yang konsisten dengan notebook training final."""

from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 17
RAW_FEATURES_PER_KEYPOINT = 3
MODEL_FEATURES_PER_KEYPOINT = 5
RAW_FEATURE_DIM = 51
BASE_MOTION_FEATURE_DIM = 85
ENGINEERED_FEATURE_DIM = 25
MODEL_FEATURE_DIM = 110

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

BODY_RELATIVE_CLIP = 4.0
MIN_BODY_SCALE = 0.03

ENGINEERED_FEATURE_NAMES = [
    "left_elbow_angle", "right_elbow_angle",
    "left_elbow_angular_velocity", "right_elbow_angular_velocity",
    "left_wrist_to_shoulder", "right_wrist_to_shoulder",
    "left_arm_extension_velocity", "right_arm_extension_velocity",
    "left_wrist_relative_speed", "right_wrist_relative_speed",
    "left_wrist_relative_horizontal_speed", "right_wrist_relative_horizontal_speed",
    "left_wrist_relative_vertical_speed", "right_wrist_relative_vertical_speed",
    "wrist_distance", "wrist_distance_change", "hand_speed_difference",
    "hand_motion_symmetry", "extension_opposition", "extension_synchrony",
    "hip_center_speed", "shoulder_center_speed",
    "left_ankle_relative_speed", "right_ankle_relative_speed", "stride_width",
]


def _safe_angle(a, b, c, va, vb, vc):
    ba, bc = a - b, c - b
    nba = np.linalg.norm(ba, axis=-1)
    nbc = np.linalg.norm(bc, axis=-1)
    valid = va & vb & vc & (nba > 1e-6) & (nbc > 1e-6)
    cosine = np.sum(ba * bc, axis=-1) / np.maximum(nba * nbc, 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
    return np.where(valid, angle, 0.0).astype(np.float32)


def _safe_distance(a, b, va, vb):
    return np.where(va & vb, np.linalg.norm(a - b, axis=-1), 0.0).astype(np.float32)


def _safe_speed(velocity, valid):
    return np.where(valid, np.linalg.norm(velocity, axis=-1), 0.0).astype(np.float32)


def _temporal_signed_difference(values, valid):
    output = np.zeros_like(values, dtype=np.float32)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    output[:, 1:] = np.where(pair_valid, values[:, 1:] - values[:, :-1], 0.0)
    return output


def _temporal_absolute_difference(values, valid):
    return np.abs(_temporal_signed_difference(values, valid)).astype(np.float32)


def _motion_symmetry(left_velocity, right_velocity, valid_left, valid_right):
    left_norm = np.linalg.norm(left_velocity, axis=-1)
    right_norm = np.linalg.norm(right_velocity, axis=-1)
    valid = valid_left & valid_right & (left_norm > 1e-6) & (right_norm > 1e-6)
    cosine = np.sum(left_velocity * right_velocity, axis=-1) / np.maximum(
        left_norm * right_norm, 1e-6
    )
    return np.where(valid, (np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0, 0.0).astype(
        np.float32
    )


def _calculate_body_center_and_scale(xy, valid):
    lhv, rhv = valid[..., LEFT_HIP], valid[..., RIGHT_HIP]
    lsv, rsv = valid[..., LEFT_SHOULDER], valid[..., RIGHT_SHOULDER]
    both_hips, both_shoulders = lhv & rhv, lsv & rsv

    hip_center = (xy[..., LEFT_HIP, :] + xy[..., RIGHT_HIP, :]) / 2.0
    shoulder_center = (
        xy[..., LEFT_SHOULDER, :] + xy[..., RIGHT_SHOULDER, :]
    ) / 2.0
    valid_count = np.maximum(valid.sum(axis=-1, keepdims=True), 1)
    valid_mean = (xy * valid[..., None]).sum(axis=-2) / valid_count
    center = np.where(
        both_hips[..., None],
        hip_center,
        np.where(both_shoulders[..., None], shoulder_center, valid_mean),
    )

    torso_length = np.linalg.norm(shoulder_center - hip_center, axis=-1)
    shoulder_width = np.linalg.norm(
        xy[..., LEFT_SHOULDER, :] - xy[..., RIGHT_SHOULDER, :], axis=-1
    )
    hip_width = np.linalg.norm(
        xy[..., LEFT_HIP, :] - xy[..., RIGHT_HIP, :], axis=-1
    )
    scale = np.where(
        both_hips & both_shoulders & (torso_length > MIN_BODY_SCALE),
        torso_length,
        np.where(
            both_shoulders & (shoulder_width > MIN_BODY_SCALE),
            shoulder_width,
            np.where(
                both_hips & (hip_width > MIN_BODY_SCALE), hip_width, MIN_BODY_SCALE
            ),
        ),
    )
    return center.astype(np.float32), np.maximum(scale, MIN_BODY_SCALE).astype(np.float32)


def build_base_motion_features(raw: np.ndarray) -> np.ndarray:
    """(N,T,51) -> (N,T,85): relative_x, relative_y, conf, frame_dx, frame_dy."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] != RAW_FEATURE_DIM:
        raise ValueError(f"Shape Raw51 salah: {raw.shape}")

    n, t, _ = raw.shape
    pose = raw.reshape(n, t, NUM_KEYPOINTS, RAW_FEATURES_PER_KEYPOINT)
    frame_xy, conf = pose[..., :2], pose[..., 2:3]
    valid = conf[..., 0] > 0
    center, scale = _calculate_body_center_and_scale(frame_xy, valid)
    relative_xy = (frame_xy - center[..., None, :]) / scale[..., None, None]
    relative_xy = np.clip(relative_xy, -BODY_RELATIVE_CLIP, BODY_RELATIVE_CLIP)
    relative_xy[~valid] = 0.0

    displacement = np.zeros_like(frame_xy, dtype=np.float32)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    displacement[:, 1:] = (frame_xy[:, 1:] - frame_xy[:, :-1]) * pair_valid[..., None]
    output = np.concatenate([relative_xy, conf, displacement], axis=-1).reshape(
        n, t, BASE_MOTION_FEATURE_DIM
    )
    if not np.isfinite(output).all():
        raise RuntimeError("Body85 mengandung NaN/Inf")
    return output.astype(np.float32)


def append_engineered_features(base_motion: np.ndarray) -> np.ndarray:
    """(N,T,85) -> (N,T,110) dengan 25 fitur biomekanik."""
    base_motion = np.asarray(base_motion, dtype=np.float32)
    if base_motion.ndim != 3 or base_motion.shape[-1] != BASE_MOTION_FEATURE_DIM:
        raise ValueError(f"Shape Body85 salah: {base_motion.shape}")

    pose = base_motion.reshape(
        *base_motion.shape[:2], NUM_KEYPOINTS, MODEL_FEATURES_PER_KEYPOINT
    )
    xy, conf, velocity = pose[..., :2], pose[..., 2], pose[..., 3:5]
    valid = conf > 0
    both_hips = valid[..., LEFT_HIP] & valid[..., RIGHT_HIP]
    both_shoulders = valid[..., LEFT_SHOULDER] & valid[..., RIGHT_SHOULDER]

    left_arm_valid = (
        valid[..., LEFT_SHOULDER] & valid[..., LEFT_ELBOW] & valid[..., LEFT_WRIST]
    )
    right_arm_valid = (
        valid[..., RIGHT_SHOULDER]
        & valid[..., RIGHT_ELBOW]
        & valid[..., RIGHT_WRIST]
    )
    lea = _safe_angle(
        xy[..., LEFT_SHOULDER, :], xy[..., LEFT_ELBOW, :], xy[..., LEFT_WRIST, :],
        valid[..., LEFT_SHOULDER], valid[..., LEFT_ELBOW], valid[..., LEFT_WRIST],
    )
    rea = _safe_angle(
        xy[..., RIGHT_SHOULDER, :], xy[..., RIGHT_ELBOW, :], xy[..., RIGHT_WRIST, :],
        valid[..., RIGHT_SHOULDER], valid[..., RIGHT_ELBOW], valid[..., RIGHT_WRIST],
    )
    leav = _temporal_absolute_difference(lea, left_arm_valid)
    reav = _temporal_absolute_difference(rea, right_arm_valid)

    lwsv = valid[..., LEFT_WRIST] & valid[..., LEFT_SHOULDER]
    rwsv = valid[..., RIGHT_WRIST] & valid[..., RIGHT_SHOULDER]
    lwts = _safe_distance(
        xy[..., LEFT_WRIST, :], xy[..., LEFT_SHOULDER, :],
        valid[..., LEFT_WRIST], valid[..., LEFT_SHOULDER],
    )
    rwts = _safe_distance(
        xy[..., RIGHT_WRIST, :], xy[..., RIGHT_SHOULDER, :],
        valid[..., RIGHT_WRIST], valid[..., RIGHT_SHOULDER],
    )
    laev = _temporal_signed_difference(lwts, lwsv)
    raev = _temporal_signed_difference(rwts, rwsv)

    lwrv = velocity[..., LEFT_WRIST, :] - velocity[..., LEFT_SHOULDER, :]
    rwrv = velocity[..., RIGHT_WRIST, :] - velocity[..., RIGHT_SHOULDER, :]
    lwrs, rwrs = _safe_speed(lwrv, lwsv), _safe_speed(rwrv, rwsv)
    lwrh = np.where(lwsv, np.abs(lwrv[..., 0]), 0.0).astype(np.float32)
    rwrh = np.where(rwsv, np.abs(rwrv[..., 0]), 0.0).astype(np.float32)
    lwrv_speed = np.where(lwsv, np.abs(lwrv[..., 1]), 0.0).astype(np.float32)
    rwrv_speed = np.where(rwsv, np.abs(rwrv[..., 1]), 0.0).astype(np.float32)

    both_wrists = valid[..., LEFT_WRIST] & valid[..., RIGHT_WRIST]
    wrist_distance = _safe_distance(
        xy[..., LEFT_WRIST, :], xy[..., RIGHT_WRIST, :],
        valid[..., LEFT_WRIST], valid[..., RIGHT_WRIST],
    )
    wrist_change = _temporal_absolute_difference(wrist_distance, both_wrists)
    speed_difference = np.where(both_wrists, np.abs(lwrs - rwrs), 0.0).astype(np.float32)
    symmetry = _motion_symmetry(lwrv, rwrv, lwsv, rwsv)
    both_extension = lwsv & rwsv
    opposition = np.where(both_extension, np.abs(laev - raev), 0.0).astype(np.float32)
    synchrony = np.where(both_extension, np.abs(laev + raev), 0.0).astype(np.float32)

    hip_velocity = (velocity[..., LEFT_HIP, :] + velocity[..., RIGHT_HIP, :]) / 2.0
    shoulder_velocity = (
        velocity[..., LEFT_SHOULDER, :] + velocity[..., RIGHT_SHOULDER, :]
    ) / 2.0
    hip_speed = _safe_speed(hip_velocity, both_hips)
    shoulder_speed = _safe_speed(shoulder_velocity, both_shoulders)
    left_ankle_speed = _safe_speed(
        velocity[..., LEFT_ANKLE, :] - hip_velocity,
        valid[..., LEFT_ANKLE] & both_hips,
    )
    right_ankle_speed = _safe_speed(
        velocity[..., RIGHT_ANKLE, :] - hip_velocity,
        valid[..., RIGHT_ANKLE] & both_hips,
    )
    stride_width = _safe_distance(
        xy[..., LEFT_ANKLE, :], xy[..., RIGHT_ANKLE, :],
        valid[..., LEFT_ANKLE], valid[..., RIGHT_ANKLE],
    )

    engineered = np.stack(
        [
            lea, rea, leav, reav, lwts, rwts, laev, raev,
            lwrs, rwrs, lwrh, rwrh, lwrv_speed, rwrv_speed,
            wrist_distance, wrist_change, speed_difference, symmetry,
            opposition, synchrony, hip_speed, shoulder_speed,
            left_ankle_speed, right_ankle_speed, stride_width,
        ],
        axis=-1,
    ).astype(np.float32)
    output = np.concatenate([base_motion, engineered], axis=-1).astype(np.float32)
    if output.shape[-1] != MODEL_FEATURE_DIM or not np.isfinite(output).all():
        raise RuntimeError(f"Body110 tidak valid: {output.shape}")
    return output


def prepare_sequence(raw_sequence, valid_mask, feature_mean, feature_std):
    """Menyiapkan Raw51 (30,51) dan mask (30,) menjadi input ONNX."""
    raw = np.asarray(raw_sequence, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    mean = np.asarray(feature_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(feature_std, dtype=np.float32).reshape(-1)
    if raw.shape != (30, RAW_FEATURE_DIM) or mask.shape != (30,):
        raise ValueError(f"Input sequence salah: raw={raw.shape}, mask={mask.shape}")
    if mean.shape != (MODEL_FEATURE_DIM,) or std.shape != (MODEL_FEATURE_DIM,):
        raise ValueError(f"Scaler harus (110,), diperoleh {mean.shape}/{std.shape}")

    base = build_base_motion_features(raw[None, ...])
    features = append_engineered_features(base)[0]
    base4 = base[0].reshape(30, NUM_KEYPOINTS, MODEL_FEATURES_PER_KEYPOINT)
    keypoint_valid = base4[..., 2] > 0
    transformed = (features - mean[None, :]) / np.maximum(std[None, :], 1e-6)
    transformed_base = transformed[:, :BASE_MOTION_FEATURE_DIM].reshape(
        30, NUM_KEYPOINTS, MODEL_FEATURES_PER_KEYPOINT
    )
    transformed_base[~keypoint_valid] = 0.0
    transformed[:, :BASE_MOTION_FEATURE_DIM] = transformed_base.reshape(
        30, BASE_MOTION_FEATURE_DIM
    )
    transformed[~mask] = 0.0
    if not np.isfinite(transformed).all():
        raise RuntimeError("Input ONNX mengandung NaN/Inf")
    return transformed[None, ...].astype(np.float32), mask[None, ...].astype(np.bool_)

