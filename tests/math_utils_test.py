"""Basic docstring for my module."""

import numpy as np
import pytest

from imu_fusion_py.config.definitions import GRAVITY
from imu_fusion_py.math_utils import align_to_acceleration


@pytest.mark.parametrize(
    "gravity_alignment_vector",
    [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])],
)
def test_align_to_gravity(gravity_alignment_vector) -> None:
    """Assert that align_to_gravity function returns a rotation matrix."""
    # Arrange
    acc_vector = GRAVITY * gravity_alignment_vector

    # Act
    rot = align_to_acceleration(acceleration_vec=acc_vector)

    # Assert
    np.testing.assert_array_almost_equal(rot[2, :], gravity_alignment_vector)
