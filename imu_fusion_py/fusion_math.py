"""Basic docstring for my module."""

from typing import Optional

import numpy as np
from lie_groups_py.se3 import SE3
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import EULER_ORDER, GRAVITY, NELDER_MEAD_METHOD
from imu_fusion_py.math_utils import matrix_exponential, skew_matrix


def predict_rotation(
    orientation: np.ndarray, angular_velocity: np.ndarray, dt: float
) -> np.ndarray:
    """Apply angular velocity vector to a rotation matrix.

    :param orientation: A 3x3 rotation matrix.
    :param angular_velocity: Angular velocity vector represented as a numpy array.
    :param dt: Time interval in seconds.
    :return: Updated rotation matrix and new angular velocity vector.
    """
    if np.shape(orientation) == (3, 3):
        return orientation @ omega_exp(angular_velocity=angular_velocity, dt=dt)
    elif np.shape(orientation) == (4, 1):
        return omega_quat(angular_velocity=angular_velocity, dt=dt) @ orientation
    else:
        raise ValueError("Orientation must be a 3x3 matrix or a 4x1 quaternion.")


def pitch_roll_from_acceleration(
    acceleration_vec: np.ndarray,
    pitch_roll_init: Optional[np.ndarray] = None,
    method: str | None = NELDER_MEAD_METHOD,
) -> tuple[np.floating, np.floating, np.floating]:
    """Find the best pitch and roll angles that align with the gravity vector.

    The yaw angle is unobservable and will be ignored. Please see the README.md
    :param acceleration_vec: acceleration values in m/s^2
    :param pitch_roll_init: Initial guess for the rotation matrix (default: zeros)
    :param method: Optimization method (default: "nelder-mead")
    :return: Rotation matrix that best aligns with gravity
    """
    if pitch_roll_init is None:
        pitch_roll_init = np.zeros(2)

    if method is None:  # simple method without optimization
        x, y, z = np.reshape(acceleration_vec, (3,))
        roll = np.atan2(y, z)
        pitch = np.atan2(-x, np.sqrt(y**2 + z**2))
        error = 0.0

    else:
        residual = minimize(
            fun=pitch_roll_alignment_error,
            x0=pitch_roll_init,
            method=method,
            args=acceleration_vec,
            tol=1e-3,
            options={"disp": False},
        )
        pitch, roll = residual.x[0], residual.x[1]
        error = residual.fun
    return pitch, roll, error


def pitch_roll_alignment_error(
    pitch_roll_angles: np.ndarray, acceleration_vector: np.ndarray
) -> float:
    """Find the orientation that would best align with the gravity vector.

    :param pitch_roll_angles: Roll, pitch, and yaw angles in degrees
    :param acceleration_vector: Gravity vector
    :return: Error between the gravity vector and the projected vector in the m/s^2
    """
    # yaw pitch roll = alpha beta gamma
    beta, gamma = pitch_roll_angles
    last_row = np.array(
        [
            [-np.sin(beta)],
            [np.cos(beta) * np.sin(gamma)],
            [np.cos(beta) * np.cos(gamma)],
        ]
    )
    error = np.linalg.norm(acceleration_vector - GRAVITY * last_row)
    return float(error)


def apply_linear_acceleration(
    pose: SE3,
    vel: np.ndarray,
    accel_meas: np.ndarray,
    dt: float,
) -> tuple[SE3, np.ndarray]:
    """Apply linear velocity vector to a rotation matrix, position, and velocity.

    :param pose: Current SE3 pose
    :param vel: Current velocity vector represented as a numpy array.
    :param accel_meas: Linear acceleration vector represented as a numpy array.
    :param dt: Time interval in seconds.
    :return: Updated position and velocity vectors.
    """
    accel = accel_meas - GRAVITY * pose.rot @ np.array([[0.0], [0.0], [1.0]])
    pose.trans += vel * dt + 0.5 * accel * dt**2
    vel += accel * dt
    return pose, vel


def rotation_matrix_from_yaw_pitch_roll(ypr: np.ndarray) -> np.ndarray:
    """Calculate the rotation matrix from yaw, pitch, and roll angles.

    :param ypr: yaw, pitch, and roll angles in radians
    :return: Rotation matrix
    """
    return Rot.from_euler(seq=EULER_ORDER, angles=ypr).as_matrix()


def omega_quat(angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    """Convert angular velocity to quaternion modifier.

    :param angular_velocity: Angular velocity vector represented as a numpy array
    :param dt: Time interval in seconds
    :return: Omega matrix
    """
    om_x, om_y, om_z = np.reshape(angular_velocity, (3,))
    skew = np.array(
        [
            [2 / dt, -om_x, -om_y, -om_z],
            [om_x, 2 / dt, om_z, -om_y],
            [om_y, -om_z, 2 / dt, om_x],
            [om_z, om_y, -om_x, 2 / dt],
        ]
    )

    return dt / 2 * skew


def omega_exp(angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    """Convert angular velocity to rotation matrix modifier using exponential map.

    :param angular_velocity: Angular velocity vector represented as a numpy array
    :param dt: Time interval in seconds
    :return: Rotation matrix
    """
    rot = matrix_exponential(skew_matrix(angular_velocity), dt=dt)
    return rot


def quat2rot(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to a rotation matrix.

    :param quat: Quaternion represented as a numpy array
    :return: Rotation matrix
    """
    quat_flat = np.reshape(quat, (4,))
    return Rot.from_quat(quat_flat, scalar_first=True).as_matrix()
