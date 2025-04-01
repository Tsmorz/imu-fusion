"""Doc string for my module."""

import pandas as pd

from imu_fusion_py.imu_data import Accelerometer, Gyroscope, ImuData, Magnetometer


class ImuParser:
    """Create a class for parsing IMU data from a CSV file."""

    def __init__(self) -> None:
        """Initialize ImuData object with imu data from a file."""

    @staticmethod
    def parse_filepath(filepath: str) -> ImuData:
        """Parse IMU data from a CSV file."""
        data = pd.read_csv(filepath, delimiter=",")
        data.columns = data.columns.str.strip()

        acc_data = Accelerometer(
            data["acc_x"].values, data["acc_y"].values, data["acc_z"].values
        )
        gyr_data = Gyroscope(
            data["gyr_x"].values, data["gyr_y"].values, data["gyr_z"].values
        )
        mag_data = Magnetometer(
            data["mag_x"].values, data["mag_y"].values, data["mag_z"].values
        )
        time = data["time"].values
        return ImuData(acc=acc_data, gyr=gyr_data, mag=mag_data, time=time)
