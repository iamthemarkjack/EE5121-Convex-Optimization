import numpy as np
import pandas as pd
from scipy.optimize import minimize

df_Phi = pd.read_csv("../Data/Phi_Q1.csv")
df_mu  = pd.read_csv("../Data/mu_Q1.csv")
df_Phi = df_Phi.select_dtypes(include=[np.number])
df_mu  = df_mu.select_dtypes(include=[np.number])
Phi = df_Phi.to_numpy(dtype=float)
mu  = df_mu.to_numpy(dtype=float).flatten()
n, k = Phi.shape

data_primal = np.load("../Outputs/q1_primal_output.npz")
p_star = data_primal["p_star"]
H_pstar = float(data_primal["H_pstar"])

def g_and_grad(theta):
    s = Phi @ theta
    a = np.max(s)
    exp_s = np.exp(s - a)
    soft = exp_s / exp_s.sum()

    g = (a + np.log(exp_s.sum())) - theta @ mu

    grad = Phi.T @ soft - mu
    return g, grad

def fun(theta):
    g,_ = g_and_grad(theta)
    return g

def jac(theta):
    _,grad = g_and_grad(theta)
    return grad

theta0 = np.zeros(k)
res = minimize(fun, theta0, jac=jac, method="L-BFGS-B",
               options={"maxiter":1000})

theta_star = res.x
g_theta_star, _ = g_and_grad(theta_star)

s = Phi @ theta_star
exp_s = np.exp(s - np.max(s))
p_tilde = exp_s / exp_s.sum()

inf_norm = np.max(np.abs(p_star - p_tilde))
duality_gap = abs(H_pstar - g_theta_star)

print("g(theta*) : ", g_theta_star)
print("||p* - p_tilde||_inf : ", inf_norm)
print("Duality gap : ", duality_gap)

np.savez("../Outputs/q1_dual_output.npz",
         theta_star=theta_star,
         g_theta_star=g_theta_star,
         p_tilde=p_tilde,
         inf_norm=inf_norm,
         duality_gap=duality_gap)