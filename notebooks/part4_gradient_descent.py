# %% [markdown]
# # Part 4: Gradient Descent in Code
#
# Here we convert our Part 3 manual calculations into Python. We keep
# **matrix form** the whole way through and keep every update step visible,
# nothing is hidden behind abstractions.
#
# Model and cost:
#
# $$\hat{y} = X m + b, \qquad J(m,b) = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$
#
# Gradients (derived with the chain rule in the Part 3 document):
#
# $$\frac{\partial J}{\partial m} = \frac{2}{n} X^\top (\hat{y} - y), \qquad
#   \frac{\partial J}{\partial b} = \frac{2}{n} (\hat{y} - y)$$

# %%
import sys
sys.path.append("../src")

import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import (X, y, M0, B0, LEARNING_RATE,
                              predict, mse, gradients, gradient_descent)

print("X =\n", X)
print("y =", y)
print("m0 =", M0, " b0 =", B0, " lr =", LEARNING_RATE)

# %% [markdown]
# ## 1. Derivative via SciPy
#
# The assignment asks for a SciPy-based function that accepts an equation
# and computes its derivative. We use `scipy.differentiate.derivative`,
# which numerically differentiates any callable. We apply it to $J$ with
# respect to each parameter and check that it matches the analytic matrix
# gradient we derived by hand.

# %%
from scipy.differentiate import derivative as scipy_derivative

def numerical_gradient(cost_fn, theta):
    """
    Accepts an equation `cost_fn` (a callable of a parameter vector) and
    returns its gradient at `theta`, one SciPy derivative per coordinate.
    """
    grad = np.zeros_like(theta, dtype=float)
    for i in range(len(theta)):
        def along_axis(t, i=i):
            t = np.asarray(t, dtype=float)
            out = np.empty_like(t)
            for j, tj in np.ndenumerate(t):     # scipy passes vectorized input
                th = theta.copy(); th[i] = tj
                out[j] = cost_fn(th)
            return out
        grad[i] = scipy_derivative(along_axis, theta[i]).df
    return grad

def cost_of_params(theta):
    """J as a function of the stacked parameter vector [m1, m2, b1, b2]."""
    m, b = theta[:2], theta[2:]
    return mse(predict(X, m, b), y)

theta0 = np.concatenate([M0, B0])
num_grad = numerical_gradient(cost_of_params, theta0)
ana_m, ana_b, _ = gradients(X, y, M0, B0)

print("SciPy numerical gradient  :", num_grad)
print("Analytic matrix gradient  :", np.concatenate([ana_m, ana_b]))
assert np.allclose(num_grad, np.concatenate([ana_m, ana_b]), atol=1e-6)
print("They match, so our chain rule derivation from Part 3 is confirmed.")

# %% [markdown]
# ## 2. Four gradient descent updates (one per group member)
#
# Every intermediate quantity gets printed: predictions, error vector, MSE,
# both gradients and the updated parameters. These numbers are exactly the
# same as our handwritten Part 3 calculations.

# %%
m_final, b_final, hist = gradient_descent(X, y, M0, B0,
                                          lr=LEARNING_RATE, iterations=4,
                                          verbose=True)

print("Final m =", m_final)
print("Final b =", b_final)
print("Predictions with final parameters:", predict(X, m_final, b_final))
print("Targets:                          ", y)
print("Final MSE:", mse(predict(X, m_final, b_final), y))

# %% [markdown]
# ## 3. Plots

# %%
its = np.arange(len(hist["mse"]))
ms = np.array(hist["m"]); bs = np.array(hist["b"])

plt.figure(figsize=(8, 4))
plt.plot(its, ms[:, 0], marker="o", label="m1")
plt.plot(its, ms[:, 1], marker="o", label="m2")
plt.plot(its, bs[:, 0], marker="s", ls="--", label="b1")
plt.plot(its, bs[:, 1], marker="s", ls="--", label="b2")
plt.title("How m and b change over the gradient descent iterations")
plt.xlabel("iteration"); plt.ylabel("value"); plt.xticks(its); plt.legend()
plt.tight_layout(); plt.savefig("../docs/img_params_over_iters.png", dpi=120)
plt.show()

# %%
plt.figure(figsize=(8, 4))
plt.plot(its, hist["mse"], marker="o", color="crimson")
plt.title("How the error (MSE) changes over the iterations")
plt.xlabel("iteration"); plt.ylabel("MSE"); plt.xticks(its)
plt.tight_layout(); plt.savefig("../docs/img_error_over_iters.png", dpi=120)
plt.show()

for i, e in enumerate(hist["mse"]):
    print(f"iteration {i}: MSE = {e:.6f}")

# %% [markdown]
# ### Trend we observe
# The MSE drops steeply after the first update (61.0 down to about 6.5) and
# keeps decreasing every iteration after that. m and b move opposite to
# their gradients: the large positive gradient on $m_2$ pulls it down hard
# first, then the parameters take smaller, stabilizing steps as the error
# surface flattens out. So yes, m and b are moving in a direction that
# reduces the error, which is exactly what gradient descent should do with
# a reasonable learning rate.
