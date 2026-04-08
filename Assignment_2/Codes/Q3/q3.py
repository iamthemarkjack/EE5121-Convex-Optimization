import cvxpy as cp
import numpy as np
import scipy.linalg as la

def solve_graph(n, edges, verbose=False):
    m = len(edges)
    G = cp.Variable((n, n), symmetric=True)
    rho = cp.Variable()

    constraints = [G >> 0, cp.diag(G) == 1]
    for (u, v) in edges:
        constraints.append(G[u, v] <= rho)

    probP = cp.Problem(cp.Minimize(rho), constraints)
    probP.solve(solver=cp.SCS, verbose=verbose)

    Gval = G.value
    rho_val = float(rho.value) if rho.value is not None else None

    lam = cp.Variable(n)
    alpha = cp.Variable(m, nonneg=True)

    mats = []
    for (u, v) in edges:
        M = np.zeros((n, n))
        M[u, v] = 1.0
        M[v, u] = 1.0
        mats.append(M)

    A_expr = sum(alpha[i] * mats[i] for i in range(m))

    probD = cp.Problem(
        cp.Maximize(cp.sum(lam)),
        [
            A_expr - cp.diag(lam) >> 0,
            cp.sum(alpha) == 1
        ]
    )
    probD.solve(solver=cp.SCS, verbose=verbose)

    lam_val = lam.value
    alpha_val = alpha.value
    dual_val = float(probD.value) if probD.value is not None else None

    primal_eigs = la.eigvalsh(Gval) if Gval is not None else None

    if lam_val is not None and alpha_val is not None:
        A_num = sum(alpha_val[i] * mats[i] for i in range(m))
        dual_mat = A_num - np.diag(lam_val)
        dual_eigs = la.eigvalsh(dual_mat)
    else:
        dual_eigs = None

    return {
        'primal_rho': rho_val,
        'G': Gval,
        'primal_eigs': primal_eigs,
        'dual_value': dual_val,
        'lambda': lam_val,
        'alpha': alpha_val,
        'duality_gap': (rho_val - dual_val) if (rho_val is not None and dual_val is not None) else None,
        'dual_eigs': dual_eigs,
    }

def print_result(name, r):
    print("Primal rho* =", r['primal_rho'])
    if r['primal_eigs'] is not None:
        print("Primal eigenvalues:", r['primal_eigs'][0], "...", r['primal_eigs'][-1])

    print("Dual value =", r['dual_value'])
    print("Duality gap =", r['duality_gap'])

    if r['lambda'] is not None:
        print("Sum lambda =", np.sum(r['lambda']))

    if r['dual_eigs'] is not None:
        print("Dual eigenvalues:", r['dual_eigs'][0], "...", r['dual_eigs'][-1])

K4_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
C5_edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]

r1 = solve_graph(4, K4_edges)
r2 = solve_graph(5, C5_edges)

print_result("K4", r1)
print_result("C5", r2)