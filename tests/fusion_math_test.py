"""Basic docstring for my module."""

import numpy as np
import pytest
from lie_groups_py.se3 import SE3
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import EULER_ORDER, GRAVITY
from imu_fusion_py.fusion_math import (
    apply_linear_acceleration,
    pitch_roll_from_acceleration,
    predict_rotation,
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
def test_pitch_roll_from_acceleration_optimize(gravity_alignment_vector) -> None:
    """Assert that align_to_gravity function returns a rotation matrix."""
    # Arrange
    acc_vector = GRAVITY * gravity_alignment_vector

    # Act
    pitch, roll, err = pitch_roll_from_acceleration(acceleration_vec=acc_vector)

    # Assert
    rot = Rot.from_euler(seq=EULER_ORDER, angles=[0.0, pitch, roll]).as_matrix()
    last_row = np.reshape(rot[2, :], (3, 1))
    np.testing.assert_array_almost_equal(last_row, gravity_alignment_vector, decimal=3)


@pytest.mark.parametrize(
    "gravity_alignment_vector",
    [
        np.array([[1.0], [0.0], [0.0]]),
        np.array([[0.0], [1.0], [0.0]]),
        np.array([[0.0], [0.0], [1.0]]),
        np.array([[0.0], [np.sqrt(2) / 2], [np.sqrt(2) / 2]]),
    ],
)
def test_pitch_roll_from_acceleration_naive(gravity_alignment_vector) -> None:
    """Assert that align_to_gravity function returns a rotation matrix."""
    # Arrange
    acc_vector = GRAVITY * gravity_alignment_vector

    # Act
    pitch, roll, err = pitch_roll_from_acceleration(
        acceleration_vec=acc_vector, method=None
    )

    # Assert
    rot = Rot.from_euler(seq=EULER_ORDER, angles=[0.0, pitch, roll]).as_matrix()
    last_row = np.reshape(rot[2, :], (3, 1))
    np.testing.assert_array_almost_equal(last_row, gravity_alignment_vector, decimal=3)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_predict_rotation(dt: float) -> None:
    """Test the apply_angular_velocity function."""
    # Arrange
    rot = np.eye(3)
    omegas = np.array([[0.0, 0.0, 2 * np.pi]])

    # Act
    for _i in range(int(1 / dt)):
        rot = predict_rotation(rot, omegas, dt)

    # Assert
    np.testing.assert_array_almost_equal(rot, np.eye(3), decimal=3)


def test_predict_rotation_methods_equal() -> None:
    """Test that the apply_angular_velocity function methods are equal."""
    # Arrange
    rot = np.eye(3)
    quat = np.array([[0.0], [0.0], [0.0], [1.0]])
    omegas = np.array([[np.pi, np.pi, np.pi]])
    dt = 0.01

    # Act
    for _i in range(100):
        rot = predict_rotation(orientation=rot, angular_velocity=omegas, dt=dt)
        quat = predict_rotation(orientation=quat, angular_velocity=omegas, dt=dt)

    # Assert
    rot_from_quat = Rot.from_quat(np.reshape(quat, (4,)))
    np.testing.assert_array_almost_equal(rot, rot_from_quat.as_matrix(), decimal=3)


def test_predict_rotation_invalid_input():
    """Test the predict_rotation function with invalid inputs."""
    # Arrange
    rot = np.eye(4)
    omegas = np.array([[0.0, 0.0, 2 * np.pi]])
    dt = 0.01

    # Act and assert
    with pytest.raises(ValueError):
        predict_rotation(rot, omegas, dt)


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
    pose = SE3(xyz=np.zeros(3), rot=np.eye(3))
    vel = np.array([[0.0], [0.0], [0.0]])
    accel = np.array([[0.0], [0.0], [GRAVITY]])
    dt = 0.01

    # Act
    for _i in range(int(1 / dt)):
        pose, vel = apply_linear_acceleration(
            pose=pose, vel=vel, accel_meas=accel, dt=dt
        )

    # Assert
    np.testing.assert_array_almost_equal(
        pose.trans, np.array([[0.0], [0.0], [0.0]]), decimal=3
    )
