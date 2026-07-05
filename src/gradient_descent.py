"""
Parts 3 and 4: gradient descent for linear regression, kept in matrix form.

Model:      y_hat = X @ m + b
Cost (MSE): J(m, b) = (1/n) * sum( (y_hat - y)^2 )
Gradients:  dJ/dm = (2/n) * X.T @ (y_hat - y)
            dJ/db = (2/n) * (y_hat - y)

Values given in the assignment:
    m0 = [-1, 2],  b0 = [1, 1]
    X  = [[1, 3], [4, 10]],  y = [5, 6]
"""

import numpy as np

X = np.array([[1.0, 3.0],
              [4.0, 10.0]])
y = np.array([5.0, 6.0])
M0 = np.array([-1.0, 2.0])
B0 = np.array([1.0, 1.0])
LEARNING_RATE = 0.01


def predict(X, m, b):
    """Predictions in matrix form: y_hat = X @ m + b."""
    return X @ m + b


def mse(y_hat, y):
    """Mean squared error J = (1/n) * sum((y_hat - y)^2)."""
    return np.mean((y_hat - y) ** 2)


def gradients(X, y, m, b):
    """
    The gradients we derived with the chain rule in Part 3:
        e     = y_hat - y                    (error vector)
        dJ/dm = (2/n) * X.T @ e
        dJ/db = (2/n) * e
    """
    n = len(y)
    e = predict(X, m, b) - y
    grad_m = (2 / n) * (X.T @ e)
    grad_b = (2 / n) * e
    return grad_m, grad_b, e


def gradient_descent(X, y, m, b, lr=LEARNING_RATE, iterations=4, verbose=True):
    """
    Run gradient descent and print every intermediate value, so each
    calculation step stays visible. We did not want to hide anything
    behind abstractions.

    Returns the final m and b plus a history dict with m, b and MSE for
    every iteration (we use it for the plots).
    """
    m, b = m.astype(float).copy(), b.astype(float).copy()
    hist = {"m": [m.copy()], "b": [b.copy()], "mse": [mse(predict(X, m, b), y)]}

    for it in range(1, iterations + 1):
        y_hat = predict(X, m, b)
        e = y_hat - y
        grad_m, grad_b, _ = gradients(X, y, m, b)
        if verbose:
            print(f"--- Iteration {it} ---")
            print(f"  y_hat        = X @ m + b = {y_hat}")
            print(f"  error e      = y_hat - y = {e}")
            print(f"  MSE          = {mse(y_hat, y):.6f}")
            print(f"  dJ/dm        = (2/n) X.T @ e = {grad_m}")
            print(f"  dJ/db        = (2/n) e       = {grad_b}")
        m = m - lr * grad_m
        b = b - lr * grad_b
        if verbose:
            print(f"  m_new        = m - lr*dJ/dm = {m}")
            print(f"  b_new        = b - lr*dJ/db = {b}\n")
        hist["m"].append(m.copy())
        hist["b"].append(b.copy())
        hist["mse"].append(mse(predict(X, m, b), y))

    return m, b, hist
