import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    p_t = np.array([p if t == 1 else 1 - p for t, p in zip(targets, predictions)])
    fl = -np.mean(alpha * (1 - p_t) ** gamma * np.log(p_t))
    return fl