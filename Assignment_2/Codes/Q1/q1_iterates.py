import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_Phi = pd.read_csv("../Data/Phi_Q1.csv")
df_mu  = pd.read_csv("../Data/mu_Q1.csv")
df_Phi = df_Phi.select_dtypes(include=[np.number])
df_mu  = df_mu.select_dtypes(include=[np.number])
Phi = df_Phi.to_numpy(dtype=float)
mu  = df_mu.to_numpy(dtype=float).flatten()
n, k = Phi.shape

data_primal = np.load("../Outputs/q1_primal_output.npz")
H_pstar = float(data_primal["H_pstar"])

def g_and_grad(theta):
    s = Phi @ theta
    a = np.max(s)
    exp_s = np.exp(s - a)
    soft = exp_s / exp_s.sum()
    grad = Phi.T @ soft - mu
    return grad, soft

maxiter = 300
eta = 0.1
theta = np.zeros(k)

H_list = []
resid_list = []
sum_err_list = []

for it in range(maxiter):
    grad, p_k = g_and_grad(theta)
    theta -= eta * grad

    H_k = -np.sum(p_k * np.log(np.maximum(p_k, 1e-20)))
    H_list.append(H_k)

    resid_list.append(np.linalg.norm(Phi.T @ p_k - mu))
    sum_err_list.append(abs(np.sum(p_k) - 1))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(H_list)
plt.axhline(H_pstar, color='r', linestyle='--')
plt.title(r"$H(p^k)$")

plt.subplot(1,3,2)
plt.semilogy(resid_list)
plt.title(r"$\|\Phi^T p^k - \mu\|$")

plt.subplot(1,3,3)
plt.semilogy(sum_err_list)
plt.title(r"$|1^T p^k - 1|$")

plt.tight_layout()
plt.savefig("../Outputs/q1_iterate_plots.png", dpi=200)
plt.show()