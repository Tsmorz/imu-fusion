"""Basic docstring for my module."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from lie_groups_py.se3 import SE3
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import (
    EULER_ORDER,
    FIG_SIZE,
    GRAVITY,
    LEGEND_LOC,
    NELDER_MEAD_METHOD,
)
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


def accel2pitch_roll(
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
        pitch = -np.arcsin(x / np.linalg.norm([x, y, z]))
        error = 0.0

        if np.isnan(roll) or np.isnan(pitch):
            raise ValueError("At least one of the acceleration values is NaN.")
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


def predict_with_acceleration(
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
    grav_vec = GRAVITY * pose.rot @ np.array([[0.0], [0.0], [1.0]])
    accel = accel_meas - grav_vec
    xyz = pose.trans + vel * dt + 0.5 * accel * dt**2
    pose.trans = xyz
    vel += accel * dt
    return pose, vel


def yaw_pitch_roll2rot(ypr: np.ndarray) -> np.ndarray:
    """Calculate the rotation matrix from yaw, pitch, and roll angles.

    :param ypr: yaw, pitch, and roll angles in radians
    :return: Rotation matrix
    """
    return Rot.from_euler(seq=EULER_ORDER, angles=ypr).as_matrix()


def omega_quat(
    angular_velocity: np.ndarray, dt: float, scalar_first: bool = False
) -> np.ndarray:
    """Convert angular velocity to quaternion modifier.

    :param angular_velocity: Angular velocity vector represented as a numpy array
    :param dt: Time interval in seconds
    :param scalar_first: If true, the scalar component of the quaternion is placed first
    :return: Omega matrix
    """
    om_x, om_y, om_z = np.reshape(angular_velocity, (3,))
    if scalar_first:
        skew = np.array(
            [
                [2 / dt, -om_x, -om_y, -om_z],
                [om_x, 2 / dt, om_z, -om_y],
                [om_y, -om_z, 2 / dt, om_x],
                [om_z, om_y, -om_x, 2 / dt],
            ]
        )
    else:
        skew = np.array(
            [
                [2 / dt, om_z, -om_y, om_x],
                [-om_z, 2 / dt, om_x, om_y],
                [om_y, -om_x, 2 / dt, om_z],
                [-om_x, -om_y, -om_z, 2 / dt],
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


def quat2rot(quat: np.ndarray, scalar_first: bool = False) -> np.ndarray:
    """Convert quaternion to a rotation matrix.

    :param quat: Quaternion represented as a numpy array
    :param scalar_first: If true, the scalar component of the quaternion is placed first
    :return: Rotation matrix
    """
    quat_flat = np.reshape(quat, (4,))
    return Rot.from_quat(quat_flat, scalar_first=scalar_first).as_matrix()


def rot2quat(rot: np.ndarray, scalar_first: bool = False) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.

    :param rot: rotation matrix represented as a numpy array
    :param scalar_first: If true, the scalar component of the quaternion is placed first
    :return: quaternion
    """
    quat = Rot.from_matrix(rot).as_quat(scalar_first=scalar_first)
    return np.reshape(quat, (4, 1))


def initialize_imu(accel: np.ndarray) -> tuple[SE3, np.ndarray]:
    """Initialize the IMU with a given acceleration vector.

    :param accel: Gravity vector represented as a numpy array
    :return: Initial position and velocity vectors.
    """
    pitch, roll, _ = accel2pitch_roll(accel, method=None)
    yaw_pitch_roll_init = np.array([0.0, pitch, roll])
    rot = yaw_pitch_roll2rot(ypr=yaw_pitch_roll_init)
    pose = SE3(xyz=np.zeros(3), rot=rot)
    vel = np.zeros((3, 1))
    return pose, vel


def extract_euler_angles(history: list[SE3]) -> tuple[list, list, list]:
    """Extract the euler angles from the pose history.

    :param history: List of SE3 matrices
    :return: Lists of yaw, pitch, and roll angles in degrees.
    """
    yaw, pitch, roll = [], [], []
    for pose in history:
        yaw_pitch_roll = Rot.from_matrix(pose.rot).as_euler(seq=EULER_ORDER)
        yaw.append(yaw_pitch_roll[0])
        pitch.append(yaw_pitch_roll[1])
        roll.append(yaw_pitch_roll[2])
    return yaw, pitch, roll


def update_plot(
    ax: plt.axis, pose: SE3, pause_time: float = 0.01
) -> None:  # pragma: no cover
    """Update the plot animation with the new pose.

    :param ax: The axis to update.
    :param pose: The current SE3 pose.
    :param pause_time: The time to pause after updating the plot.
    """
    ax.clear()
    pose.plot(ax=ax)
    plt.axis("equal")
    ax.set_xlim([pose.trans[0] - 2, pose.trans[0] + 2])
    ax.set_ylim([pose.trans[1] - 2, pose.trans[1] + 2])
    ax.set_zlim([pose.trans[2] - 2, pose.trans[2] + 2])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.pause(pause_time)


def plot_euler_angles(state_history: list[list[SE3]]) -> None:  # pragma: no cover
    """Plot the euler angles from the pose history.

    :param state_history: List of lists of SE3 matrices representing the pose history.
    """
    plt.figure(figsize=FIG_SIZE)
    for state_list in state_history:
        ypr = extract_euler_angles(state_list)
        plt.plot(np.rad2deg(ypr[0]), label="yaw", color="blue", alpha=0.8)
        plt.plot(np.rad2deg(ypr[1]), label="pitch", color="orange", alpha=0.8)
        plt.plot(np.rad2deg(ypr[2]), label="roll", color="green", alpha=0.8)
    plt.grid(True)
    plt.legend(loc=LEGEND_LOC)
    plt.title("Euler Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Euler angle (degrees)")
    plt.show()
