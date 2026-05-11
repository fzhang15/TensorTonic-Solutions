import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mu = np.mean(x, axis=-1, keepdims=True)
    sigma_2 = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(sigma_2 + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch, seq_len, d_model = Q.shape
    d_head = d_model // num_heads
    Q_proj = np.matmul(Q, W_q).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K_proj = np.matmul(K, W_k).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V_proj = np.matmul(V, W_v).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = softmax(np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2)) / np.sqrt(d_head))
    attention = np.matmul(scores, V_proj)
    attention = attention.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    out = np.matmul(attention, W_o)
    return out

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    h = np.maximum(np.matmul(x, W1) + b1, 0)
    out = np.matmul(h, W2) + b2
    return out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    h = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    output = layer_norm(h + feed_forward(h, W1, b1, W2, b2), gamma2, beta2)
    return output