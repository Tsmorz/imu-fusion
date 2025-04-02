"""Doc string for my module."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from imu_fusion_py.config.definitions import FIG_SIZE, LEGEND_LOC


@dataclass
class Accelerometer:
    """Create a class for storing accelerometer data."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def get_idx(self, idx: int) -> np.ndarray:
        """Get a column vector of accelerometer data at a specific index."""
        return np.array([[self.x[idx]], [self.y[idx]], [self.z[idx]]])


@dataclass
class Gyroscope:
    """Create a class for storing gyroscope data."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def get_idx(self, idx: int) -> np.ndarray:
        """Get a column vector of gyroscope data at a specific index."""
        return np.array([[self.x[idx]], [self.y[idx]], [self.z[idx]]])


@dataclass
class Magnetometer:
    """Create a class for storing magnetometer data."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def get_idx(self, idx: int) -> np.ndarray:
        """Get a column vector of magnetometer data at a specific index."""
        return np.array([[self.x[idx]], [self.y[idx]], [self.z[idx]]])


@dataclass
class ImuData:
    """Create a class for storing IMU data."""

    acc: Accelerometer
    gyr: Gyroscope
    mag: Magnetometer
    time: np.ndarray

    def get_idx(self, idx: int) -> np.ndarray:
        """Get a column vector of all IMU data at a specific index."""
        return np.vstack(
            (
                self.acc.get_idx(idx),
                self.gyr.get_idx(idx),
                self.mag.get_idx(idx),
                self.time[idx],
            )
        )

    def plot(self, figsize: tuple[float, float] = FIG_SIZE) -> None:  # pragma: no cover
        """Plot IMU data.

        :param figsize: Figure size.
        :return: None
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        ax1.plot(self.time, self.acc.x, label="acc_x")
        ax1.plot(self.time, self.acc.y, label="acc_y")
        ax1.plot(self.time, self.acc.z, label="acc_z")
        ax1.legend(loc=LEGEND_LOC)
        ax1.set_title("IMU Accelerometer Data")
        ax1.set_ylabel("Acceleration (m/s^2)")
        ax1.grid(True)

        ax2.plot(self.time, self.gyr.x, label="gyr_x")
        ax2.plot(self.time, self.gyr.y, label="gyr_y")
        ax2.plot(self.time, self.gyr.z, label="gyr_z")
        ax2.legend(loc=LEGEND_LOC)
        ax2.set_title("IMU Gyroscope Data")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angular Velocity (rad/s)")
        ax2.grid(True)

        fig = plt.figure(figsize=(8, 8)).add_subplot(111, projection="3d")
        plt.plot(self.mag.x, self.mag.y, self.mag.z, label="mag_xyz")
        plt.title("IMU Magnetometer Data")
        fig.set_xlabel("X-axis (milliGauss)")
        fig.set_ylabel("Y-axis (milliGauss)")
        fig.set_zlabel("Z-axis (milliGauss)")
        plt.grid(True)
        plt.show()


class ImuIterator:
    """Create an iterator for IMU data."""

    def __init__(self, data: ImuData):
        self.data = data
        self.index = 0

    def __iter__(self):
        """Make sure the object is iterable."""
        return self

    def __next__(self):
        """Return the next IMU measurement."""
        if self.index < len(self.data.time):
            result = self.data.get_idx(self.index)
            self.index += 1
            return result
        else:
            raise StopIteration
