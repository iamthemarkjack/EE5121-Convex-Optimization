import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Data
A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
              [0.8, 1.0, 0.0, 0.0, 0.0],
              [-0.2,0.8 ,1.0, 0.0, 0.0],
              [0.5,-0.2, 0.8 ,1.0, 0.0],
              [0.0 ,0.5,-0.2 ,0.8 ,1.0]])

B = np.array([[ 1.0, -1.0,  0.0,  0.0,  1.0],
              [ 2.8, -0.8,  1.0, -1.0,  0.8],
              [ 2.4,  1.2,  1.8, -1.8, -1.2],
              [ 2.9, -1.7,  0.6, -0.6,  1.7],
              [ 0.4, -1.8, -0.7,  0.7,  1.8]])

rng = np.random.default_rng(seed=0) # Seeded for reproducibility
N = np.sqrt(0.1) * rng.standard_normal((5,5))
B_noisy = B + N


lambdas = [1e-3, 1e-2, 1e-1, 1.]
results = []

for lam in lambdas:
    X = cp.Variable((5,5))
    obj = cp.Minimize(lam * cp.normNuc(X) + 0.5 * cp.sum_squares(A @ X - B_noisy))
    problem = cp.Problem(obj)
    problem.solve(solver=cp.SCS, verbose=False)

    Xs = X.value
    residual = np.linalg.norm(A @ Xs - B_noisy, ord='fro')
    nuc_norm = np.linalg.norm(Xs, ord='nuc')
    results.append((lam, Xs, residual, nuc_norm, problem.status))

for lam, Xs, residual, nuc_norm, status in results:
    print(f"lambda = {lam}")
    print(" status:", status)
    print("X* =\n", Xs)
    print(" nuclear norm ||X*||_* =", nuc_norm)
    print(" residual ||AX-B_noisy||_F =", residual)
    print("-----")

nuc_norms = [r[3] for r in results]
residuals = [r[2] for r in results]

# Plot
fig, ax1 = plt.subplots(figsize=(8,5))

ax1.set_xscale('log')
ax1.plot(lambdas, nuc_norms, 'o-', color='blue', label=r'Nuclear norm $\|X^*\|_*$')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'Nuclear norm $\|X^*\|_*$', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(lambdas, residuals, 's--', color='red', label=r'Residual $\|AX-B_{\mathrm{noisy}}\|_F$')
ax2.set_ylabel(r'Residual $\|AX-B_{\mathrm{noisy}}\|_F$', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Nuclear Norm and Residual vs Lambda")
fig.tight_layout()
plt.savefig("nuclear_norm_residual_vs_lambda.png", dpi=300)
plt.show()