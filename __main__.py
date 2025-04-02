"""Doc string for my module."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from lie_groups_py.se3 import SE3
from lie_groups_py.so3 import SO3
from loguru import logger
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

from imu_fusion_py.config.definitions import IMU_DATA_FILENAME
from imu_fusion_py.fusion_math import (
    apply_linear_acceleration,
    pitch_roll_from_acceleration,
    predict_rotation,
    quat2rot,
    rotation_matrix_from_yaw_pitch_roll,
)
from imu_fusion_py.imu_data import ImuIterator
from imu_fusion_py.imu_parser import ImuParser


def main(imu_filepath: str, show_plot: bool = False) -> None:
    """Run main pipeline."""
    imu_data = ImuParser().parse_filepath(imu_filepath)
    if show_plot:
        imu_data.plot()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    gyr_bias_x = np.nanmean(imu_data.gyr.x[5:100])
    gyr_bias_y = np.nanmean(imu_data.gyr.y[5:100])
    gyr_bias_z = np.nanmean(imu_data.gyr.z[5:100])
    gyr_bias = 0.0 * np.array([[gyr_bias_x], [gyr_bias_y], [gyr_bias_z]])

    pose = SE3(xyz=np.zeros(3), rot=np.eye(3))
    vel = np.zeros((3, 1))

    quat = Rot.from_matrix(matrix=pose.rot).as_quat()
    quat = np.reshape(quat, (4, 1))
    measurements = ImuIterator(imu_data)
    t_old = 0
    for ii, t in enumerate(
        tqdm(imu_data.time, disable=False, desc="Processing IMU data")
    ):
        measurement = next(measurements)
        acc = measurement[0:3]
        gyr = measurement[3:6] - gyr_bias
        dt = t - t_old
        t_old = t
        if dt > 0.02 or dt <= 0.0:
            continue

        pitch, roll, _ = pitch_roll_from_acceleration(acc, method=None)
        yaw_pitch_roll_init = np.array([0.0, pitch, roll])
        rot = rotation_matrix_from_yaw_pitch_roll(ypr=yaw_pitch_roll_init)

        pose, vel = apply_linear_acceleration(pose=pose, vel=vel, accel_meas=acc, dt=dt)
        quat = predict_rotation(quat, angular_velocity=gyr, dt=dt)

        if ii % 10 == 0:
            r = rot.T @ quat2rot(quat=quat)
            sk = r - r.T
            sk_str = np.array2string(sk, precision=3, suppress_small=True)
            logger.info(f"Skew matrix:\n{sk_str}")
            ax.clear()
            so3 = SO3(rot=quat2rot(quat=quat))
            so3.plot(ax=ax)
            plt.axis("equal")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            plt.pause(0.01)
    plt.show()

    rot_from_quat = Rot.from_quat(np.reshape(quat, (4,))).as_matrix()
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
    elif pipeline == 1:
        filename = "mag-cal-" + IMU_DATA_FILENAME
    elif pipeline == 2:
        filename = "moving-" + IMU_DATA_FILENAME
    elif pipeline == 3:
        filename = IMU_DATA_FILENAME
    else:
        raise NotImplementedError(f"Pipeline {pipeline} is not supported.")

    filepath = os.path.join(cwd, "tests", "test_data", filename)

    main(imu_filepath=filepath, show_plot=args.plot)
