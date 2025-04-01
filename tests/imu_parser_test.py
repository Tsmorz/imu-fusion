"""Basic docstring for my module."""

import os

from imu_fusion_py.config.definitions import IMU_DATA_FILENAME
from imu_fusion_py.imu_data import ImuData
from imu_fusion_py.imu_parser import ImuParser
from tests.conftest import ROOT

TEST_FILEPATH = os.path.join(ROOT, "test_data", IMU_DATA_FILENAME)


def test_imu_parser() -> None:
    """Test ImuParser class."""
    # Arrange
    imu_filepath = str(TEST_FILEPATH)
    len_data = 3

    # Act
    parser = ImuParser()
    imu_data = parser.parse_filepath(imu_filepath)

    # Assert
    assert isinstance(imu_data, ImuData)
    assert imu_data.time.shape == (len_data,)
    assert imu_data.acc.x.shape == (len_data,)
    assert imu_data.acc.y.shape == (len_data,)
    assert imu_data.acc.z.shape == (len_data,)
    assert imu_data.gyr.x.shape == (len_data,)
    assert imu_data.gyr.y.shape == (len_data,)
    assert imu_data.gyr.z.shape == (len_data,)
    assert imu_data.mag.x.shape == (len_data,)
    assert imu_data.mag.y.shape == (len_data,)
    assert imu_data.mag.z.shape == (len_data,)
