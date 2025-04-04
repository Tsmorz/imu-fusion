"""Doc string for my module."""

import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from imu_fusion_py.config.definitions import IMU_DATA_FILENAME
from imu_fusion_py.fusion_math import (
    accel2pitch_roll,
    initialize_imu,
    plot_euler_angles,
    predict_rotation,
    predict_with_acceleration,
    quat2rot,
    rot2quat,
    update_plot,
    yaw_pitch_roll2rot,
)
from imu_fusion_py.imu_data import ImuIterator
from imu_fusion_py.imu_parser import ImuParser


def main(imu_filepath: str, show_plot: bool = False) -> None:
    """Run main pipeline."""
    logger.info("Running main pipeline.")
    imu_data = ImuParser().parse_filepath(imu_filepath)
    if show_plot:
        imu_data.plot()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    state_history1 = []
    state_history2 = []

    gyr_bias = np.array(
        [
            [np.nanmean(imu_data.gyr.x[5:100])],
            [np.nanmean(imu_data.gyr.y[5:100])],
            [np.nanmean(imu_data.gyr.z[5:100])],
        ]
    )

    # Initialize the orientation
    pose, vel = initialize_imu(accel=imu_data.get_idx(0)[:3])
    quat = rot2quat(pose.rot)

    t_old = 0
    for measurement in tqdm(
        ImuIterator(imu_data), desc="Processing IMU data", leave=True
    ):
        acc = measurement[0:3]
        gyr = measurement[3:6] - gyr_bias
        t = measurement[9, 0]

        # use the measurement if the dt is less than 0.02
        dt = t - t_old
        t_old = t
        if dt > 0.02 or dt <= 0.0:
            continue

        # use the acceleration values to find pitch and roll
        pitch, roll, _ = accel2pitch_roll(acc, method=None)
        rot = yaw_pitch_roll2rot(ypr=np.array([0.0, pitch, roll]))

        # use the gyroscope values to find yaw, pitch and roll
        quat = predict_rotation(quat, angular_velocity=gyr, dt=dt)

        # update the pose orientation values
        pose.rot = quat2rot(quat)
        state_history1.append(copy.copy(pose))
        pose.rot = rot
        state_history2.append(copy.copy(pose))

        # predict the position and velocity with the new orientation
        pose, vel = predict_with_acceleration(pose=pose, vel=vel, accel_meas=acc, dt=dt)

        # animate the orientation
        if show_plot:
            update_plot(ax=ax, pose=pose, pause_time=0.01)

    logger.info("Processing complete.")
    rot_from_quat_str = np.array2string(quat2rot(quat), precision=3)
    logger.info(f"Final SO3:\n{rot_from_quat_str}")

    if show_plot:
        plt.show()
    plot_euler_angles(state_history=[state_history1, state_history2])


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
