"""Doc string for my module."""

import argparse
import os

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as Rot

from imu_fusion_py.config.definitions import (
    EULER_ORDER,
    IMU_DATA_FILENAME,
)
from imu_fusion_py.fusion_math import (
    apply_angular_velocity,
    apply_linear_acceleration,
    pitch_roll_from_acceleration,
    yaw_pitch_roll_to_rotation_matrix,
)
from imu_fusion_py.imu_data import ImuIterator
from imu_fusion_py.imu_parser import ImuParser


def main(show_plot: bool = False) -> None:
    """Run main pipeline."""
    home = os.path.expanduser("~")
    imu_filepath = os.path.join(home, "Desktop", IMU_DATA_FILENAME)
    imu_data = ImuParser().parse_filepath(imu_filepath)

    if show_plot:
        imu_data.plot()

    gyr_bias = imu_data.get_idx(0)[3:6]
    pitch_roll_init = np.zeros(2)
    pos, vel = np.zeros((3, 1)), np.zeros((3, 1))
    for measurement in ImuIterator(imu_data):
        acc = measurement[0:3]
        gyr = measurement[3:6] - gyr_bias
        t = measurement[9, 0]
        dt = 0.01

        yaw = 0.0
        pitch, roll, _ = pitch_roll_from_acceleration(
            acceleration_vec=acc, pitch_roll_init=pitch_roll_init
        )
        yaw_pitch_roll_init = np.array([yaw, pitch, roll])
        rot = yaw_pitch_roll_to_rotation_matrix(ypr=yaw_pitch_roll_init)

        pos, vel = apply_linear_acceleration(
            pos=pos, vel=vel, rot=rot, accel_meas=acc, dt=dt
        )
        rot_from_gyr = apply_angular_velocity(matrix=rot, omegas=gyr, dt=dt)

        # Print the results
        logger.warning(f"Comparing Acc and Gyr at t={t:.3f} sec")

        ypr = Rot.from_matrix(rot).as_euler(EULER_ORDER, degrees=True)
        ypr_str = np.array2string(ypr, precision=2, suppress_small=True)
        logger.info(f"Acc: {ypr_str}")

        ypr = Rot.from_matrix(rot_from_gyr).as_euler(EULER_ORDER, degrees=True)
        ypr_str = np.array2string(ypr, precision=2, suppress_small=True)
        logger.info(f"Gyr: {ypr_str}")


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
