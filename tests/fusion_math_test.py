"""Basic docstring for my module."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import EULER_ORDER, GRAVITY
from imu_fusion_py.fusion_math import pitch_roll_from_acceleration


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
