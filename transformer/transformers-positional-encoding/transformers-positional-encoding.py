import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    model_pos = np.arange(0, d_model, 2).reshape(1, -1)
    seq_pos = np.arange(seq_length).reshape(-1, 1)
    _base = 10000 ** (model_pos / d_model)

    pe = np.empty((seq_length, d_model))
    pe[:, : : 2] = np.sin(seq_pos / _base)
    if d_model % 2 == 0:
        pe[:, 1: : 2] = np.cos(seq_pos / _base)
    else:
        pe[:, 1: : 2] = np.cos(seq_pos / _base[: -1])

    return pe