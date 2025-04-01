"""Basic docstring for my module."""

import numpy as np

from imu_fusion_py.imu_data import Accelerometer, Gyroscope, ImuData, Magnetometer


def test_imu_data() -> None:
    """Test ImuData class."""
    # Arrange
    acc_data = Accelerometer(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    gyr_data = Gyroscope(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    mag_data = Magnetometer(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    time = np.array([0.0])

    # Act
    imu_data = ImuData(acc=acc_data, gyr=gyr_data, mag=mag_data, time=time)

    # Assert
    assert isinstance(imu_data, ImuData)
    assert imu_data.get_idx(0).shape == (9, 1)
