"""Add a doc string to my files."""

from typing import Optional

import numpy as np
import scipy
from loguru import logger
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from sympy import Matrix

from imu_fusion_py.config.definitions import EULER_ORDER, GRAVITY, METHOD


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


def align_to_acceleration(
    acceleration_vec: np.ndarray, x0: Optional[np.ndarray] = None, method: str = METHOD
) -> np.ndarray:
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
        fun=orientation_error,
        x0=x0,
        method=method,
        args=acceleration_vec,
        tol=1e-3,
        options={"disp": True},
    )
    yaw_pitch_roll = [0.0, residual.x[0], residual.x[1]]
    return Rot.from_euler(seq=EULER_ORDER, angles=yaw_pitch_roll).as_matrix()


def orientation_error(pitch_roll_angles: np.ndarray, g_vector: np.ndarray) -> float:
    """Find the orientation that would best align with the gravity vector.

    :param pitch_roll_angles: Roll, pitch, and yaw angles in degrees
    :param g_vector: Gravity vector
    :return: Error between the gravity vector and the projected vector in the m/s^2
    """
    angles = np.hstack(([0.0], pitch_roll_angles))
    rot = Rot.from_euler(seq=EULER_ORDER, angles=angles, degrees=False).as_matrix()
    error = np.linalg.norm(np.reshape(g_vector, (3,)) - GRAVITY * rot[2, :])
    return float(error)


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


def apply_linear_acceleration(
    pos: np.ndarray,
    vel: np.ndarray,
    rot: np.ndarray,
    accel: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply linear velocity vector to a rotation matrix, position, and velocity.

    :param pos: Current position vector represented as a numpy array.
    :param vel: Current velocity vector represented as a numpy array.
    :param rot: Current rotation matrix.
    :param accel: Linear acceleration vector represented as a numpy array.
    :param dt: Time interval in seconds.
    :return: Updated position and velocity vectors.
    """
    residual = accel - GRAVITY * rot @ np.array([[0], [0], [1]])
    vel += residual * dt
    pos += vel * dt
    return pos, vel, rot
