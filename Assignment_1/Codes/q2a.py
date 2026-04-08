import numpy as np
import cvxpy as cp

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

# Defining the variable
X = cp.Variable((5,5))

objective = cp.Minimize(cp.normNuc(X))
constraints = [A @ X == B]
problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.SCS, verbose=False)

X_star = X.value

print("Status:", problem.status)
print("Optimal nuclear norm:", problem.value)
print("X* =\n", X_star)