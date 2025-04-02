"""Basic docstring for my module."""

import numpy as np

from imu_fusion_py.imu_data import (
    Accelerometer,
    Gyroscope,
    ImuData,
    ImuIterator,
    Magnetometer,
)

num = 5
ACC_DATA = Accelerometer(np.arange(num), np.arange(num), np.arange(num))
GYR_DATA = Gyroscope(np.arange(num), np.arange(num), np.arange(num))
MAG_DATA = Magnetometer(np.arange(num), np.arange(num), np.arange(num))
TIME = np.arange(num)


def test_imu_data() -> None:
    """Test ImuData class."""
    # Arrange
    acc_data = ACC_DATA
    gyr_data = GYR_DATA
    mag_data = MAG_DATA
    time = TIME

    # Act
    imu_data = ImuData(acc=acc_data, gyr=gyr_data, mag=mag_data, time=time)

    # Assert
    assert isinstance(imu_data, ImuData)
    assert imu_data.get_idx(0).shape == (10, 1)


def test_imu_iterator() -> None:
    """Test ImuIterator class."""
    # Arrange
    acc_data = ACC_DATA
    gyr_data = GYR_DATA
    mag_data = MAG_DATA
    time = TIME
    imu_data = ImuData(acc=acc_data, gyr=gyr_data, mag=mag_data, time=time)

    # Act
    iterator = ImuIterator(imu_data)

    # Assert
    for ii, measurement in enumerate(iterator):
        assert np.shape(measurement) == (10, 1)
        np.testing.assert_array_almost_equal(measurement, ii * np.ones((10, 1)))
