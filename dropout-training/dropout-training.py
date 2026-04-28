import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    if rng is not None:
        rand_tensor = rng.random(x.shape)
    else:
        rand_tensor = np.random.random(x.shape)
    output = np.where(rand_tensor < p, 0, x / (1 - p))
    dropout_pattern = np.where(rand_tensor < p, 0, 1 / (1 - p))

    return (output, dropout_pattern)