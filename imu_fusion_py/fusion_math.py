"""Basic docstring for my module."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from imu_fusion_py.config.definitions import GRAVITY, METHOD


def pitch_roll_from_acceleration(
    acceleration_vec: np.ndarray, x0: Optional[np.ndarray] = None, method: str = METHOD
) -> tuple[np.floating, np.floating, np.floating]:
    """Find the best pitch and roll angles that align with the gravity vector.

    The yaw angle is unobservable and will be ignored. Please see the README.md
    :param acceleration_vec: acceleration values in m/s^2
    :param x0: Initial guess for the rotation matrix (default: zeros)
    :param method: Optimization method (default: "nelder-mead")
    :return: Rotation matrix that best aligns with gravity
    """
    if x0 is None:
        x0 = np.zeros(2)
    residual = minimize(
        fun=pitch_roll_alignment_error,
        x0=x0,
        method=method,
        args=acceleration_vec,
        tol=1e-3,
        options={"disp": True},
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


def main():
    """Run a simple test case."""
    acc = np.array([[GRAVITY / 3], [0.0], [GRAVITY / 3]])
    pitch, roll, error = pitch_roll_from_acceleration(acceleration_vec=acc)
    logger.info(
        f"Pitch: {np.rad2deg(pitch):.3f} degrees, "
        f"Roll: {np.rad2deg(roll):.3f} degrees, "
        f"Error: {error:.3f} m/s^2"
    )


if __name__ == "__main__":
    main()
