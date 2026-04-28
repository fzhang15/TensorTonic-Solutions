import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    preds = np.array(predictions)
    targs = np.array(targets)
    p_t = np.where(targs == 1, preds, 1 - preds)
    fl = -np.mean(alpha * (1 - p_t) ** gamma * np.log(p_t))
    return fl