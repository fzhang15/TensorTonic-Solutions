import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)

    y = np.maximum(x @ W1, 0)
    z = np.maximum(y @ W2, 0)
    h = z @ W3
    x = x @ Ws

    return np.maximum(x + h, 0)
