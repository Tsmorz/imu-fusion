# imu-fusion

#### Table of Contents
[Background Information](#Background Information)
[Installation](#Install)
[Development](#Development)
[Usage](#Usage)
[References](#References)

## Background Information
The following sections cover the necessary mathematical to cover the fusion process of the IMU data.


### General 3D rotations

A general 3D rotation matrix can be obtained from these three matrices using matrix multiplication.  For example, the product:

$$
\begin{align}
  R = R_z(\alpha) \, R_y(\beta) \, R_x(\gamma) &=
  {\begin{bmatrix}
    \cos \alpha & -\sin \alpha & 0 \\
    \sin \alpha &  \cos \alpha & 0 \\
              0 &            0 & 1 \\
  \end{bmatrix}}
  {\begin{bmatrix}
     \cos \beta & 0 & \sin \beta \\
              0 & 1 &          0 \\
    -\sin \beta & 0 & \cos \beta \\
  \end{bmatrix}}
  {\begin{bmatrix}
    1 &  0          &            0 \\
    0 & \cos \gamma & -\sin \gamma \\
    0 & \sin \gamma &  \cos \gamma \\
  \end{bmatrix}} \\
  &= \begin{bmatrix}
        \cos\alpha\cos\beta &
          \cos\alpha\sin\beta\sin\gamma - \sin\alpha\cos\gamma &
          \cos\alpha\sin\beta\cos\gamma + \sin\alpha\sin\gamma \\
        \sin\alpha\cos\beta &
          \sin\alpha\sin\beta\sin\gamma + \cos\alpha\cos\gamma &
          \sin\alpha\sin\beta\cos\gamma - \cos\alpha\sin\gamma \\
       -\sin\beta & \cos\beta\sin\gamma & \cos\beta\cos\gamma \\
  \end{bmatrix}
\end{align}</math>$$

represents a rotation whose yaw, pitch, and roll angles are $\alpha$, $\beta$ and $\gamma$, respectively. More formally, it is an intrinsic rotation whose Tait–Bryan angles are  $\alpha$, $\beta$ and $\gamma$, about axes  $x$, $y$ and $z$, respectively. [^1]

> [!IMPORTANT]
> It is clear from looking at the last row of this matrix, that the yaw angle ($\alpha$) can not be found. This means that the yaw angle has no impact on the $z$ component. Therefore, the yaw angle has no observability with the accelerometer.


## Install
To install the library run: `pip install imu_fusion_py`

## Development
0. Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
1. `make init` to create the virtual environment and install dependencies
2. `make format` to format the code and check for errors
3. `make test` to run the test suite
4. `make clean` to delete the temporary files and directories
5. `poetry publish --build` to build and publish to https://pypi.org/project/lie_groups_py/


## Usage
```
"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from se3_group import se3


def main() -> None:
    """Run a simple demonstration."""
    pose_0 = se3.SE3(
        xyz=np.array([0.0, 0.0, 0.0]),
        rot=np.eye(3),
    )
    pose_1 = se3.SE3(
        xyz=np.array([[2.0], [4.0], [8.0]]),
        roll_pitch_yaw=np.array([np.pi / 2, np.pi / 4, np.pi / 8]),
    )

    logger.info(f"Pose 1: {pose_0}")
    logger.info(f"Pose 2: {pose_1}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pose_0.plot(ax)
    pose_1.plot(ax)

    for t in np.arange(0.0, 1.01, 0.1):
        pose_interp = se3.interpolate(pose_1, pose_0, t=t)
        pose_interp.plot(ax)
        logger.info(f"Interpolated Pose at t={t:.2f}: {pose_interp}")

    plt.axis("equal")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

```

## References
[^1]: [3D Rotation matrices](https://en.wikipedia.org/wiki/Rotation_matrix)
