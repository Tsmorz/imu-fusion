"""Doc string for my module."""

import pandas as pd
from loguru import logger


def main() -> None:
    """Run main pipeline."""
    imu_filepath = "/Users/tsmoragiewicz/Desktop/imu-data.txt"
    df = pd.read_csv(imu_filepath, delimiter=",")
    logger.info(f"Initial block of imu data:\n{df.head()}")


if __name__ == "__main__":
    main()
