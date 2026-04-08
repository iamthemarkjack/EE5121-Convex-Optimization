import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Given data
A = np.array([[1,0,1],
              [0,1,1],
              [1,1,2]], dtype=float)
b = np.array([0.2, 0.8, 0.95], dtype=float)

def solve_gamma(gamma, solver=cp.SCS):
    x = cp.Variable(3, nonneg=True)
    obj = cp.Minimize(cp.sum(x) + gamma * cp.sum_squares(A @ x - b))
    problem = cp.Problem(obj)
    problem.solve(solver=solver, verbose=False)
    x_star = x.value
    residual = np.linalg.norm(A @ x_star - b)
    l1 = np.sum(x_star)
    support = np.nonzero(np.abs(x_star) > 1e-6)[0]
    return {'gamma': gamma, 'x': x_star, 'support': support, 'residual': residual, 'l1': l1}

# Sweep over gamma values
gammas = [1., 10., 100., 1000., 10_000.]
results = [solve_gamma(g) for g in gammas]

# Print a summary
for r in results:
    print("gamma =", r['gamma'])
    print(" x* =", np.round(r['x'], 6))
    print(" support (1-based) =", (r['support'] + 1).tolist())
    print(" residual ||Ax-b||_2 =", np.round(r['residual'], 8))
    print(" l1 =", np.round(r['l1'], 8))
    print("---------------------------")

l1_vals = [r['l1'] for r in results]
residuals = [r['residual'] for r in results]

# Plot
fig, ax1 = plt.subplots(figsize=(8,5))

ax1.set_xscale('log')
ax1.plot(gammas, l1_vals, 'o-', color='blue', label=r'$||x||_1$')
ax1.set_xlabel(r'$\gamma$')
ax1.set_ylabel(r'$||x||_1$', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(gammas, residuals, 's--', color='red', label=r'$||Ax-b||_2$')
ax2.set_ylabel(r'Residual $||Ax-b||_2$', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('L1 norm and Residual vs Gamma')
plt.savefig('l1_residual_vs_gamma.png', dpi=300)
plt.show()