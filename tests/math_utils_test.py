"""Basic docstring for my module."""

import numpy as np
import pytest

from imu_fusion_py.config.definitions import GRAVITY
from imu_fusion_py.math_utils import (
    align_to_acceleration,
    matrix_exponential,
    skew_matrix,
    symmetrize_matrix,
)


@pytest.mark.parametrize(
    "gravity_alignment_vector",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
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


def test_matrix_exponential_invalid_input() -> None:
    """Assert that matrix_exponential raises a ValueError for invalid input."""
    # Arrange
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Act and Assert
    with pytest.raises(ValueError) as e_info:
        matrix_exponential(matrix)

    # Assert
    err_msg = "Input matrix must be square. Matrix has dimensions: 2x3."
    assert str(e_info.value) == err_msg


def test_matrix_exponential_identity() -> None:
    """Assert that matrix_exponential returns the identity matrix for t=0."""
    # Arrange
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t = 0.0

    # Act
    exp_matrix = matrix_exponential(matrix, t)

    # Assert
    np.testing.assert_array_almost_equal(exp_matrix, matrix, decimal=3)


def test_matrix_exponential_zeros() -> None:
    """Assert that matrix_exponential returns the identity matrix for t=0."""
    # Arrange
    matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    t = 1.0

    # Act
    exp_matrix = matrix_exponential(matrix, t)

    # Assert
    np.testing.assert_array_almost_equal(exp_matrix, np.eye(3), decimal=3)


@pytest.mark.parametrize(
    "dt",
    [0.001, 0.01, 0.1, 1.0],
)
def test_matrix_exponential_non_diagonalizable(dt) -> None:
    """Assert that matrix_exponential returns the identity matrix for t=0."""
    # Arrange
    matrix = np.array([[0, 1], [0, 0]])
    expected_matrix = np.array([[1.0, dt], [0.0, 1.0]])

    # Act
    matrix_exp = matrix_exponential(matrix=matrix, t=dt)

    # Assert
    np.testing.assert_array_almost_equal(matrix_exp, expected_matrix, decimal=2)


def test_symmetrize_matrix() -> None:
    """Assert that symmetrize_matrix returns a symmetric matrix."""
    # Arrange
    matrix = np.array([[0.5, 1.0, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.5]])

    # Act
    sym_matrix = symmetrize_matrix(matrix)

    # Assert
    assert np.all(sym_matrix == 0.5)


def test_symmetrize_matrix_invalid() -> None:
    """Assert that symmetrize_matrix raises an error for non-square matrices."""
    # Arrange
    matrix = np.array([[0.5, 1.0], [0.0, 0.5], [0.0, 0.0]])

    # Act and Assert
    with pytest.raises(ValueError) as e_info:
        symmetrize_matrix(matrix)

    # Assert
    err_msg = "Input matrix must be square. Matrix has dimensions: 3x2."
    assert str(e_info.value) == err_msg
