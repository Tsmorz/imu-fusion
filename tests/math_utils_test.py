"""Basic docstring for my module."""

import numpy as np
import pytest

from imu_fusion_py.config.definitions import GRAVITY
from imu_fusion_py.math_utils import align_to_acceleration, skew_matrix


@pytest.mark.parametrize(
    "gravity_alignment_vector",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3]),
    ],
)
def test_align_to_gravity(gravity_alignment_vector) -> None:
    """Assert that align_to_gravity function returns a rotation matrix."""
    # Arrange
    acc_vector = GRAVITY * gravity_alignment_vector

    # Act
    rot = align_to_acceleration(acceleration_vec=acc_vector)

    # Assert
    np.testing.assert_array_almost_equal(rot[2, :], gravity_alignment_vector, decimal=3)


def test_skew_matrix() -> None:
    """Assert that skew_matrix function returns a skew symmetric matrix."""
    # Arrange
    vector = np.array([[1.0], [2.0], [3.0]])

    # Act
    sk = skew_matrix(vector)

    # Assert
    np.testing.assert_array_equal(
        sk,
        np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]]),
    )


def test_skew_matrix_invalid_input() -> None:
    """Assert that skew_matrix raises a ValueError for invalid input."""
    # Arrange
    vector = np.array([1.0, 2.0])

    # Act and Assert
    with pytest.raises(ValueError) as e_info:
        skew_matrix(vector)

    # Assert
    assert str(e_info.value) == "Input vector must have a dimension of 3."
