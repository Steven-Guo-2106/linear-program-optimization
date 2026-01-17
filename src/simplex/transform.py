""" "
transform.py

Provides functions to transform linear programs into
the necessary forms for the simplex algorithm.
"""

import numpy as np


def negate_constraint(A, b):
    negated_rows = b < 0
    A[negated_rows.flatten(), :] = -A[negated_rows.flatten(), :]
    b[negated_rows] = -b[negated_rows]
    return A, b, negated_rows


def convert_sef(obj_sym, c, obj_const, A, constraints, b, x_constraints):
    if obj_sym == "min":
        c = -c
        obj_sym = "max"
        obj_const = -obj_const

    m = A.shape[0]
    for i in range(m):
        if constraints[i] == "<=":
            slack = np.zeros((m, 1))
            slack[i] = 1
            A = np.hstack((A, slack))
            c = np.hstack((c, 0))
            x_constraints = np.append(x_constraints, ">=0")
        elif constraints[i] == ">=":
            slack = np.zeros((m, 1))
            slack[i] = -1
            A = np.hstack((A, slack))
            c = np.hstack((c, 0))
            x_constraints = np.append(x_constraints, ">=0")
        elif constraints[i] == "=":
            continue
        else:
            raise ValueError("Invalid constraint type.")
    constraints = np.array(["="] * m)

    indices = np.where(x_constraints == "<=0")[0]
    A[:, indices] = -A[:, indices]
    c[indices] = -c[indices]
    x_constraints[indices] = ">=0"

    free_indices = np.where(x_constraints == "free")[0]
    for idx in free_indices:
        new_col = np.zeros((A.shape[0], 1))
        new_col[:, 0] = -A[:, idx]
        A = np.hstack((A, new_col))

        new_slack = np.array(-c[idx])
        c = np.hstack((c, new_slack))
        x_constraints = np.append(x_constraints, ">=0")

        x_constraints[idx] = ">=0"
    return obj_sym, c, obj_const, A, constraints, b, x_constraints


def canonical_form(c, obj_const, A, b, basis_indices):
    AB = A[:, basis_indices].astype(float)
    cB = c[basis_indices].astype(float)
    y = np.linalg.solve(AB.T, cB)
    c = c - (y @ A)
    obj_const = obj_const + (y @ b)
    A = np.linalg.solve(AB, A)
    b = np.linalg.solve(AB, b)

    return c, obj_const, A, b, basis_indices


def remove_artificial_vars(A, b, basis, n_orig, tol=1e-12):
    A = A.copy().astype(float)
    b = b.copy().astype(float).reshape(-1)
    basis = np.array(basis, dtype=int).copy()

    m, n_total = A.shape
    assert n_total == n_orig + m
    c_temp = np.zeros(n_total)
    _, _, A_can, b_can, basis = canonical_form(c_temp, 0.0, A, b, basis)

    keep_rows = np.ones(m, dtype=bool)
    for i in range(m):
        if basis[i] >= n_orig:
            row = A_can[i, :n_orig]
            candidates = np.where(np.abs(row) > tol)[0]
            if candidates.size > 0:
                j = int(candidates[0])
                basis[i] = j
                _, _, A_can, b_can, basis = canonical_form(
                    c_temp, 0.0, A_can, b_can, basis
                )
            else:
                keep_rows[i] = False
    A_orig = A_can[keep_rows, :n_orig]
    b_orig = b_can[keep_rows]
    basis_orig = basis[keep_rows]
    return A_orig, b_orig, basis_orig
