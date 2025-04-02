"""Basic docstring for my module."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import EULER_ORDER, GRAVITY, METHOD
from imu_fusion_py.math_utils import matrix_exponential, skew_matrix


def apply_angular_velocity(
    matrix: np.ndarray, omegas: np.ndarray, dt: float
) -> np.ndarray:
    """Apply angular velocity vector to a rotation matrix.

    :param matrix: A 3x3 rotation matrix.
    :param omegas: Angular velocity vector represented as a numpy array.
    :param dt: Time interval in seconds.
    :return: Updated rotation matrix and new angular velocity vector.
    """
    omega_exp = matrix_exponential(skew_matrix(omegas), t=dt)
    return matrix @ omega_exp


def pitch_roll_from_acceleration(
    acceleration_vec: np.ndarray,
    pitch_roll_init: Optional[np.ndarray] = None,
    method: str = METHOD,
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
    pos: np.ndarray,
    vel: np.ndarray,
    accel_meas: np.ndarray,
    rot: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply linear velocity vector to a rotation matrix, position, and velocity.

    :param pos: Current position vector represented as a numpy array.
    :param vel: Current velocity vector represented as a numpy array.
    :param rot: Current rotation matrix.
    :param accel_meas: Linear acceleration vector represented as a numpy array.
    :param dt: Time interval in seconds.
    :return: Updated position and velocity vectors.
    """
    accel = accel_meas - GRAVITY * rot @ np.array([[0.0], [0.0], [1.0]])
    pos += vel * dt + 0.5 * accel * dt**2
    vel += accel * dt
    return pos, vel


def yaw_pitch_roll_to_rotation_matrix(ypr: np.ndarray) -> np.ndarray:
    """Calculate the rotation matrix from yaw, pitch, and roll angles.

    :param ypr: yaw, pitch, and roll angles in radians
    :return: Rotation matrix
    """
    return Rot.from_euler(seq=EULER_ORDER, angles=ypr).as_matrix()


if __name__ == "__main__":  # pragma: no cover
    acc = np.array([[GRAVITY / 3], [0.0], [GRAVITY / 3]])
    pitch, roll, err = pitch_roll_from_acceleration(acceleration_vec=acc)
    logger.info(
        f"Pitch: {np.rad2deg(pitch):.3f} degrees, "
        f"Roll: {np.rad2deg(roll):.3f} degrees, "
        f"Error: {err:.3f} m/s^2"
    )
