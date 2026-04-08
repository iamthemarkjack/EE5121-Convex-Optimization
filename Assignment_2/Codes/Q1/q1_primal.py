import numpy as np
import pandas as pd
import cvxpy as cp

df_Phi = pd.read_csv("../Data/Phi_Q1.csv")
df_mu  = pd.read_csv("../Data/mu_Q1.csv")
df_Phi = df_Phi.select_dtypes(include=[np.number])
df_mu  = df_mu.select_dtypes(include=[np.number])
Phi = df_Phi.to_numpy(dtype=float)
mu  = df_mu.to_numpy(dtype=float).flatten()
n, k = Phi.shape

p = cp.Variable(n, nonneg=True)
entropy = cp.sum(cp.entr(p))

constraints = [
    Phi.T @ p == mu,
    cp.sum(p) == 1
]

prob = cp.Problem(cp.Maximize(entropy), constraints)
prob.solve(solver=cp.SCS, verbose=False)

p_star = p.value
H_pstar = -np.sum(p_star * np.log(np.maximum(p_star, 1e-20)))

print("Status:", prob.status)
print("H(p*):", H_pstar)

np.savez("../Outputs/q1_primal_output.npz",
         p_star=p_star,
         H_pstar=H_pstar)