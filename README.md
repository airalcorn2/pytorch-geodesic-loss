# PyTorch Geodesic Loss

A PyTorch criterion for computing the distance between rotation matrices as described [here](http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices) and [here](https://link.springer.com/article/10.1007/s10851-009-0161-2).
Specifically, the loss is calculated as:

<p align="center">
  <img src="geodesic_loss.png" width="400">
</p>


```python
import numpy as np
import torch

from geodesic_loss import GeodesicLoss


def gen_single_angle_rotation_matrix(which_angle, angle):
    if which_angle == "yaw":
        (first_idx, second_idx) = (0, 2)
        negs = np.array([1.0, 1.0, -1.0, 1.0])
    elif which_angle == "pitch":
        (first_idx, second_idx) = (1, 2)
        negs = np.array([1.0, -1.0, 1.0, 1.0])
    elif which_angle == "roll":
        (first_idx, second_idx) = (0, 1)
        negs = np.array([1.0, -1.0, 1.0, 1.0])

    R = np.eye(3)
    R[first_idx, first_idx] = negs[0] * np.cos(angle)
    R[first_idx, second_idx] = negs[1] * np.sin(angle)
    R[second_idx, first_idx] = negs[2] * np.sin(angle)
    R[second_idx, second_idx] = negs[3] * np.cos(angle)
    return R


def gen_rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    R_yaw = gen_single_angle_rotation_matrix("yaw", yaw)
    R_pitch = gen_single_angle_rotation_matrix("pitch", pitch)
    R_roll = gen_single_angle_rotation_matrix("roll", roll)
    return R_yaw @ R_pitch @ R_roll


N = 100
Rs = []
for i in range(2 * N):
    yaw = np.random.uniform(-np.pi, np.pi)
    pitch = np.random.uniform(-np.pi, np.pi)
    roll = np.random.uniform(-np.pi, np.pi)
    Rs.append(torch.Tensor(gen_rotation_matrix(yaw, pitch, roll)))

Rs = torch.stack(Rs)
R_Ss = Rs[:N]
R_Ts = Rs[N:]

criterion = GeodesicLoss(reduction="none")
dists = criterion(R_Ss, R_Ts)
```
