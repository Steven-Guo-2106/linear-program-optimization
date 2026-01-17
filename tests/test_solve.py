import numpy as np

from simplex.lp_types import LPProblem, LPStatus
from simplex.two_phase import two_phase_simplex

TOL = 1e-7


def test_optimal_small_lp():
    # max 3x + 2y
    # s.t. x + y <= 4
    #      x <= 2
    #      y <= 3
    #      x, y >= 0
    # Optimum at (2,2) with value 10
    lp = LPProblem(
        obj="max",
        c=np.array([3.0, 2.0]),
        A=np.array([
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        b=np.array([4.0, 2.0, 3.0]),
        constraint_signs=["<=", "<=", "<="],
        var_bounds=[">=0", ">=0"],
        obj_const=0.0,
    )

    res = two_phase_simplex(lp)

    assert res.status == LPStatus.OPTIMAL
    assert res.x is not None
    assert res.objective_value is not None
    assert res.y is not None

    # Objective check
    assert abs(res.objective_value - 10.0) <= 1e-6

    # Basic sanity on primal
    assert np.all(res.x >= -TOL)


def test_unbounded_simple_lp():
    # max x
    # s.t. x >= 0  (encoded as -x <= 0)
    lp = LPProblem(
        obj="max",
        c=np.array([1.0]),
        A=np.array([[-1.0]]),
        b=np.array([0.0]),
        constraint_signs=["<="],
        var_bounds=[">=0"],
        obj_const=0.0,
    )

    res = two_phase_simplex(lp)

    assert res.status == LPStatus.UNBOUNDED
    assert res.ray is not None
    # Ray should be nonzero
    assert np.linalg.norm(res.ray) > 0


def test_infeasible_equalities():
    # x = 0 and x = 1 simultaneously
    lp = LPProblem(
        obj="max",
        c=np.array([1.0]),
        A=np.array([
            [1.0],
            [1.0],
        ]),
        b=np.array([0.0, 1.0]),
        constraint_signs=["=", "="],
        var_bounds=[">=0"],
        obj_const=0.0,
    )

    res = two_phase_simplex(lp)

    assert res.status == LPStatus.INFEASIBLE
    assert res.y is not None
