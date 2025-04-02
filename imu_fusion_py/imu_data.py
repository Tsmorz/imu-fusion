"""Doc string for my module."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from imu_fusion_py.config.definitions import FIG_SIZE


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
        plt.figure(figsize=figsize)
        plt.plot(self.time, self.acc.x, label="acc_x")
        plt.plot(self.time, self.acc.y, label="acc_y")
        plt.plot(self.time, self.acc.z, label="acc_z")
        plt.legend()
        plt.title("IMU Accelerometer Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s^2)")
        plt.grid(True)

        plt.figure(figsize=figsize)
        plt.plot(self.time, self.gyr.x, label="gyr_x")
        plt.plot(self.time, self.gyr.y, label="gyr_y")
        plt.plot(self.time, self.gyr.z, label="gyr_z")
        plt.legend()
        plt.title("IMU Gyroscope Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.grid(True)

        plt.figure(figsize=figsize)
        plt.plot(self.time, self.mag.x, label="mag_x")
        plt.plot(self.time, self.mag.y, label="mag_y")
        plt.plot(self.time, self.mag.z, label="mag_z")
        plt.legend()
        plt.title("IMU Magnetometer Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Field (milliGauss)")
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
