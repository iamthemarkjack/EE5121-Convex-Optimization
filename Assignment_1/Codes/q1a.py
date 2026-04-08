import cvxpy as cp
import numpy as np

# Given data
A = np.array([[1, 0, 1],
	      	  [0, 1, 1],
	          [1, 1, 2]])

b = np.array([0.2, 0.8, 1.0])

# Decision Variable
x = cp.Variable(3, nonneg=True) # Enforcing it to be >= 0

# Solving Sparse Recovery using L1 minimization as proxy
objective = cp.Minimize(cp.sum(x)) # L1-Norm of vector with non-negative entries is just the sum
constraints = [A @ x == b] # Linear equality constraints
problem = cp.Problem(objective, constraints)

problem.solve()

x_star = x.value
support = np.nonzero(x_star)[0]

print("x* = ", x_star)
print("Support = ", support + 1) # 1-based indexing
