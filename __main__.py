"""Doc string for my module."""

import argparse
import os

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import (
    EULER_ORDER,
    GRAVITY,
    IMU_DATA_FILENAME,
)
from imu_fusion_py.fusion_math import apply_angular_velocity
from imu_fusion_py.imu_parser import ImuParser
from imu_fusion_py.math_utils import align_to_acceleration


def main(show_plot: bool = False) -> None:
    """Run main pipeline."""
    home = os.path.expanduser("~")
    imu_filepath = os.path.join(home, "Desktop", IMU_DATA_FILENAME)
    imu_data = ImuParser().parse_filepath(imu_filepath)

    if show_plot:
        imu_data.plot()

    data = imu_data.get_idx(0)
    rot = align_to_acceleration(acceleration_vec=data[0:3], x0=np.zeros(3))

    rot_from_gyr = rot
    data = imu_data.get_idx(0)
    gyr_bias = data[3:6]
    x0 = np.zeros(3)
    dt = 0.01
    pos = np.zeros((3, 1))
    vel = np.zeros((3, 1))
    for ii, t in enumerate(imu_data.time):
        data = imu_data.get_idx(ii)
        acc = data[0:3]
        gyr = data[3:6] - gyr_bias

        rot_from_acc = align_to_acceleration(
            acceleration_vec=acc, x0=x0, method="Powell"
        )
        x0 = Rot.from_matrix(rot_from_acc).as_euler(EULER_ORDER, degrees=False)

        g_body_frame = GRAVITY * rot_from_acc @ np.array([[0], [0], [1]])
        g_body_frame[1] = -g_body_frame[1]
        residual = acc - g_body_frame

        acc = residual
        vel += acc * dt
        pos += vel * dt + acc * dt**2
        logger.info(f"Residual: {residual.T} m/s**2")

        rot_from_gyr = apply_angular_velocity(matrix=rot_from_gyr, omegas=gyr, dt=dt)

        logger.warning(f"Comparing Acc and Gyr at t={t:.3f} sec")
        ypr = Rot.from_matrix(rot_from_acc).as_euler(EULER_ORDER, degrees=True)
        logger.info(f"Acc: {ypr}")
        ypr = Rot.from_matrix(rot_from_gyr).as_euler(EULER_ORDER, degrees=True)
        logger.info(f"Gyr: {ypr}")


if __name__ == "__main__":  # pragma: no cover
    """Run the main program with this function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--plot",
        required=False,
        action="store_true",
        help="Plot the raw IMU data",
    )
    args = parser.parse_args()

    main(show_plot=args.plot)
