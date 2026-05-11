import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch, seq_len, d_model = Q.shape[0], Q.shape[1], Q.shape[2]
    d_head = d_model // num_heads
    Q_proj = np.matmul(Q, W_q).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K_proj = np.matmul(K, W_k).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V_proj = np.matmul(V, W_v).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = np.matmul(Q_proj, K_proj.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    scores_norm = softmax(scores, axis=-1)
    attentions = np.matmul(scores_norm, V_proj)

    attentions = attentions.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    output = np.matmul(attentions, W_o)
    return output    
    