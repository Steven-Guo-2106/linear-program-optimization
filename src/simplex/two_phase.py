"""
two_phase.py

Module implementing the two-phase simplex algorithm.
"""
import numpy as np

from .lp_types import LPStatus
from .simplex_core import simplex
from .transform import convert_sef, negate_constraint, remove_artificial_vars

TOL = 1e-12


def two_phase_simplex(lp):
    obj_sym = lp.obj
    obj_const = lp.obj_const
    c = np.array(lp.c)
    A = np.array(lp.A)
    constraints = np.array(lp.constraint_signs, dtype=str).reshape(-1)
    b = np.array(lp.b)
    x_constraints = np.array(lp.var_bounds)

    obj_sym, c, obj_const, A, constraints, b, x_constraints = convert_sef(
        obj_sym, c, obj_const, A, constraints, b, x_constraints
    )
    A, b, negated_constraints = negate_constraint(A, b)

    # Augment original LP with artificial variables for phase 1
    m, n = A.shape
    A_phase1 = np.hstack((A, np.identity(m)))
    c_phase1 = np.hstack((np.zeros(n), -np.ones(m)))
    basis_indices = list(range(n, n + m))

    result_phase1 = simplex(c_phase1, 0, A_phase1, b, basis_indices, phase=1)
    if result_phase1.status == LPStatus.INFEASIBLE:
        result_phase1.y[negated_constraints] *= -1
        return result_phase1

    basis_phase1 = result_phase1.basis
    A, b, basis = remove_artificial_vars(A_phase1, b, basis_phase1, n, tol=TOL)
    result_phase2 = simplex(c, obj_const, A, b, basis, phase=2)
    if result_phase2.status == LPStatus.OPTIMAL:
        result_phase2.y[negated_constraints] *= -1
    return result_phase2
