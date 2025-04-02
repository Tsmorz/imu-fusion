"""Basic docstring for my module."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import EULER_ORDER, GRAVITY
from imu_fusion_py.fusion_math import (
    apply_angular_velocity,
    apply_linear_acceleration,
    pitch_roll_from_acceleration,
    rotation_matrix_from_yaw_pitch_roll,
)


@pytest.mark.parametrize(
    "gravity_alignment_vector",
    [
        np.array([[1.0], [0.0], [0.0]]),
        np.array([[0.0], [1.0], [0.0]]),
        np.array([[0.0], [0.0], [1.0]]),
        np.array([[0.0], [np.sqrt(2) / 2], [np.sqrt(2) / 2]]),
    ],
)
def test_pitch_roll_from_acceleration(gravity_alignment_vector) -> None:
    """Assert that align_to_gravity function returns a rotation matrix."""
    # Arrange
    acc_vector = GRAVITY * gravity_alignment_vector

    # Act
    pitch, roll, err = pitch_roll_from_acceleration(acceleration_vec=acc_vector)

    # Assert
    rot = Rot.from_euler(seq=EULER_ORDER, angles=[0.0, pitch, roll]).as_matrix()
    last_row = np.reshape(rot[2, :], (3, 1))
    np.testing.assert_array_almost_equal(last_row, gravity_alignment_vector, decimal=3)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_apply_angular_velocity(dt: float) -> None:
    """Test the apply_angular_velocity function."""
    # Arrange
    rot = np.eye(3)
    omegas = np.array([[0.0, 0.0, 2 * np.pi]])

    # Act
    for _i in range(int(1 / dt)):
        rot = apply_angular_velocity(rot, omegas, dt)

    # Assert
    np.testing.assert_array_almost_equal(rot, np.eye(3), decimal=3)


def test_yaw_pitch_roll_to_rotation_matrix() -> None:
    """Test the yaw_pitch_roll_to_rotation_matrix function."""
    # Arrange
    yaw = 0.0
    pitch = 0.0
    roll = 0.0

    # Act
    rot = rotation_matrix_from_yaw_pitch_roll(np.array([yaw, pitch, roll]))

    # Assert
    np.testing.assert_array_almost_equal(rot, np.eye(3), decimal=3)


def test_apply_linear_acceleration():
    """Test the apply_linear_acceleration function."""
    # Arrange
    pos = np.array([[0.0], [0.0], [0.0]])
    vel = np.array([[0.0], [0.0], [0.0]])
    rot = np.eye(3)
    accel = np.array([[0.0], [0.0], [GRAVITY]])
    dt = 0.01

    # Act
    for _i in range(int(1 / dt)):
        pos, vel = apply_linear_acceleration(
            pos=pos, vel=vel, rot=rot, accel_meas=accel, dt=dt
        )

    # Assert
    np.testing.assert_array_almost_equal(
        pos, np.array([[0.0], [0.0], [0.0]]), decimal=3
    )
