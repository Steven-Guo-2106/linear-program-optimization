""" "
simplex_core.py

Implements the simplex algorithm for linear programs and extracts certificates.
Assumes the LP is in canonical form with respect to the given basis.
"""

import numpy as np

from .lp_types import LPStatus, SimplexResult
from .transform import canonical_form

TOL = 1e-12


def simplex(c, obj_const, A, b, basis_indices, phase):
    # Save original LP for final value and certificate
    A0 = np.array(A, dtype=float).copy()
    c0 = np.array(c, dtype=float).copy()

    basis_indices = np.array(basis_indices, dtype=int)
    c, obj_const, A, b, basis_indices = canonical_form(
        c, obj_const, A, b, basis_indices
    )
    n = A.shape[1]
    unbounded = False
    while np.any(c > TOL):
        entering = int(np.argmax(c > 0))

        B = A[:, basis_indices].astype(float)
        a_entering = A[:, entering].astype(float)
        d = np.linalg.solve(B, a_entering)
        xB = np.linalg.solve(B, b).flatten()

        eligible = d > TOL
        if not np.any(eligible):
            unbounded = True
            ray = np.zeros(n, dtype=float)
            ray[entering] = 1.0
            ray[basis_indices] = -d
            d = ray
            break
        ratios = np.full_like(xB, np.inf, dtype=float)
        ratios[eligible] = xB[eligible] / d[eligible]
        min_ratio = ratios.min()
        candidates = np.where(np.abs(ratios - min_ratio) <= TOL)[0]
        leaving = int(candidates[np.argmin(np.array(basis_indices)[candidates])])

        basis_indices[leaving] = entering
        c, obj_const, A, b, basis_indices = canonical_form(
            c, obj_const, A, b, basis_indices
        )

    bfs = np.zeros(n)
    B = A[:, basis_indices].astype(float)
    xB = np.linalg.solve(B, b.astype(float)).flatten()
    bfs[basis_indices] = xB

    if unbounded:
        return SimplexResult(
            status=LPStatus.UNBOUNDED,
            x=bfs,
            ray=d,
        )
    elif phase == 1 and obj_const < -TOL:
        B = A0[:, basis_indices].astype(float)
        cB = c0[basis_indices]
        certificate = np.linalg.solve(B.T, cB)
        return SimplexResult(
            status=LPStatus.INFEASIBLE,
            y=certificate,
        )
    else:
        B = A0[:, basis_indices].astype(float)
        cB0 = c0[basis_indices]
        certificate = np.linalg.solve(B.T, cB0)
        return SimplexResult(
            status=LPStatus.OPTIMAL,
            x=bfs,
            objective_value=obj_const,
            y=certificate,
            basis=basis_indices,
        )
