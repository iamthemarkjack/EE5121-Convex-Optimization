import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

df = pd.read_csv("../Data/Data_Q2.csv", header=None)
X = df.iloc[1:, :-1].values.astype(float)
y = df.iloc[1:, -1].values.astype(float)
n, d = X.shape

def logistic_loss(w):
    z = y * (X @ w)
    out = np.log1p(np.exp(-np.clip(z, -50, 50)))
    return np.sum(out)

def logistic_grad(w):
    z = y * (X @ w)
    s = 1 / (1 + np.exp(np.clip(z, -50, 50)))
    grad = -(X.T @ (y * s))
    return grad

loss_hist = []
norm_hist = []

def fun(w):
    val = logistic_loss(w)
    loss_hist.append(val)
    norm_hist.append(np.linalg.norm(w))
    return val

def jac(w):
    return logistic_grad(w)

w0 = np.zeros(d)

res = minimize(
    fun, w0, jac=jac, method="L-BFGS-B",
    options={"maxiter": 500}
)

print("Status :", res.message)
print("Final loss :", loss_hist[-1])
print("Final norm :", norm_hist[-1])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.semilogy(loss_hist)
plt.xlabel("Iteration $k$")
plt.ylabel(r"$L(w^k)$")
plt.grid(True, which="both")
plt.title("Loss vs iteration (unregularized)")

plt.subplot(1,2,2)
plt.semilogy(norm_hist)
plt.xlabel("Iteration $k$")
plt.ylabel(r"$\|w^k\|_2$")
plt.grid(True, which="both")
plt.title("Weight norm vs iteration (unregularized)")

plt.tight_layout()
plt.savefig("../Outputs/q2_unreg_plots.png", dpi=300)
plt.close()