import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N, D = X.shape[0], X.shape[1]
    w, b = np.random.randn(D), 0
    for i in range(steps):
        p = _sigmoid(np.dot(X, w) + b)
        dw = 1 / N * np.dot(X.T, (p - y))
        db = np.mean(p - y)
        w -= lr * dw
        b -= lr * db

    return w, b