"""Add a doc string to my files."""

import numpy as np
import scipy
from loguru import logger
from sympy import Matrix


def skew_matrix(vector: np.ndarray) -> np.ndarray:
    """Calculate the skew symmetric matrix from a given vector.

    :param vector: A 3D vector represented as a numpy array.
    :return: The skew symmetric matrix of the given vector.
    """
    dim = len(np.shape(vector))
    if dim == 2:
        vector = np.reshape(vector, (3,))
    if len(vector) != 3:
        raise ValueError("Input vector must have a dimension of 3.")

    sk = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    return sk


def matrix_exponential(matrix: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Calculate the matrix exponential of a given matrix.

    :param matrix: A square matrix represented as a numpy array.
    :param t: The time parameter.
    :return: The matrix exponential of the given matrix.
    """
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        dim = matrix.shape
        msg = f"Input matrix must be square. Matrix has dimensions: {dim[0]}x{dim[1]}."
        logger.error(msg)
        raise ValueError(msg)

    mat = Matrix(matrix)
    if mat.is_diagonalizable():
        eig_val, eig_vec = np.linalg.eig(matrix)
        diagonal = np.diag(np.exp(eig_val * t))
        matrix_exp = eig_vec @ diagonal @ np.linalg.inv(eig_vec)
    else:
        logger.warning(
            "Non-diagonalizable matrix encountered. Using Jordan canonical form."
        )
        P, J = mat.jordan_form()
        P, J = np.array(P).astype(np.float64), np.array(J).astype(np.float64)
        J = scipy.linalg.expm(t * J)
        matrix_exp = P @ J @ np.linalg.inv(P)
    return matrix_exp.real


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix.

    :param matrix: A square matrix represented as a numpy array.
    """
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        dim = matrix.shape
        msg = f"Input matrix must be square. Matrix has dimensions: {dim[0]}x{dim[1]}."
        logger.error(msg)
        raise ValueError(msg)

    return (matrix + matrix.T) / 2
