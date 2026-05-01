import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.array(points)
    T = np.array(T)

    if points.ndim == 1:
        pts_2d = points.reshape(1, -1)
    else:
        pts_2d = points

    ones = np.ones((pts_2d.shape[0], 1))
    pts_2d_aug = np.concatenate([pts_2d, ones], axis=1)

    pts_2d_new = pts_2d_aug @ T.T

    if pts_2d_new.shape[0] == 1:
        result = pts_2d_new[0, :3].tolist()
    else:
        result = pts_2d_new[:, :3].tolist()

    return result