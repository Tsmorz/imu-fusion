"""Doc string for my module."""

import argparse
import os

import numpy as np
from lie_groups_py.se3 import SE3
from loguru import logger
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

from imu_fusion_py.config.definitions import IMU_DATA_FILENAME
from imu_fusion_py.fusion_math import (
    apply_linear_acceleration,
    pitch_roll_from_acceleration,
    predict_rotation,
    rotation_matrix_from_yaw_pitch_roll,
)
from imu_fusion_py.imu_data import ImuIterator
from imu_fusion_py.imu_parser import ImuParser


def main(imu_filepath: str, show_plot: bool = False) -> None:
    """Run main pipeline."""
    imu_data = ImuParser().parse_filepath(imu_filepath)
    if show_plot:
        imu_data.plot()

    gyr_bias_x = np.mean(imu_data.gyr.x[5:100])
    gyr_bias_y = np.mean(imu_data.gyr.y[5:100])
    gyr_bias_z = np.mean(imu_data.gyr.z[5:100])
    gyr_bias = np.array([[gyr_bias_x], [gyr_bias_y], [gyr_bias_z]])

    pose = SE3(xyz=np.zeros(3), rot=np.eye(3))
    vel = np.zeros((3, 1))

    quat = Rot.from_matrix(matrix=pose.rot).as_quat(scalar_first=True)
    quat = np.reshape(quat, (4, 1))
    measurements = ImuIterator(imu_data)
    for t in tqdm(imu_data.time, disable=False, desc="Processing IMU data"):
        measurement = next(measurements)
        acc = measurement[0:3]
        gyr = measurement[3:6] - gyr_bias
        dt = 0.01
        if t < 0.05:
            continue

        pitch, roll, _ = pitch_roll_from_acceleration(acc, method=None)

        yaw_pitch_roll_init = np.array([0.0, pitch, roll])
        pose.rot = rotation_matrix_from_yaw_pitch_roll(ypr=yaw_pitch_roll_init)

        pose, vel = apply_linear_acceleration(pose=pose, vel=vel, accel_meas=acc, dt=dt)
        quat = predict_rotation(quat, angular_velocity=gyr, dt=dt)

    rot_from_quat = Rot.from_quat(np.reshape(quat, (4,)), scalar_first=True).as_matrix()
    rot_from_quat_str = np.array2string(
        rot_from_quat, precision=3, suppress_small=False
    )
    logger.info(f"Final SO3:\n{rot_from_quat_str}")


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
    parser.add_argument(
        "-i",
        "--int",
        required=True,
        type=int,
        action="store",
        help="Process test data",
    )
    args = parser.parse_args()

    cwd = os.getcwd()

    pipeline = args.int
    if pipeline == 0:
        filename = "stationary-" + IMU_DATA_FILENAME
    if pipeline == 1:
        filename = "mag-cal-" + IMU_DATA_FILENAME
    elif pipeline == 2:
        filename = "moving-" + IMU_DATA_FILENAME
    elif pipeline == 3:
        filename = IMU_DATA_FILENAME
    else:
        filename = IMU_DATA_FILENAME
    filepath = os.path.join(cwd, "tests", "test_data", filename)

    main(imu_filepath=filepath, show_plot=args.plot)
