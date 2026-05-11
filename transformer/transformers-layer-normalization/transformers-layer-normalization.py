import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mu = np.mean(x, axis=-1, keepdims=True)
    sigma_2 = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(sigma_2 + eps) + beta