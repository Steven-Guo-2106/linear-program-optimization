"""
io.py

Module for reading linear programs from user input or JSON files.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from .lp_types import LPProblem


def get_inputs():
    obj_sym = input("max/min?: ").strip().lower()
    if obj_sym not in ["max", "min"]:
        raise ValueError("Objective function type must be 'max' or 'min'")

    obj_coeffs = list(
        map(
            float,
            input("Objective function coefficients (space-separated): ")
            .strip()
            .split(),
        )
    )
    c = np.array(obj_coeffs, dtype=float)

    obj_const = float(input("Objective function constant (enter 0 if none): ").strip())

    A_rows = int(input("Number of constraint rows: ").strip())
    A_cols = int(input("Number of constraints columns: ").strip())
    A_coeffs = list(
        map(float, input("Constraint coefficients (space-separated): ").strip().split())
    )
    if len(A_coeffs) != A_rows * A_cols:
        raise ValueError(
            f"Expected {A_rows * A_cols} coefficients for A, got {len(A_coeffs)}"
        )
    A = np.array(A_coeffs, dtype=float).reshape(A_rows, A_cols)

    constraints = (
        input("Constraint types (=, >=, <=) (space-separated): ").strip().split()
    )
    if len(constraints) != A_rows:
        raise ValueError("Number of constraints must match number of rows in A")
    constraints = np.array(constraints).reshape(
        A_rows,
    )

    b_values = list(
        map(float, input("Right-hand side values (space-separated): ").strip().split())
    )
    b = np.array(b_values, dtype=float).reshape(
        A_rows,
    )

    if c.shape[0] != A_cols:
        raise ValueError(
            "Number of objective coefficients must match number of columns in A"
        )
    if A.shape[0] != b.shape[0]:
        raise ValueError("Number of rows in A must match number of rows in b")

    x_constraints = (
        input("Enter variable constraints (space-separated, >=0 <=0 free): ")
        .strip()
        .split()
    )
    if len(x_constraints) != A_cols:
        raise ValueError(
            "Number of variable constraints must match number of variables"
        )
    x_constraints = np.array(x_constraints)

    return LPProblem(
        obj=obj_sym,
        c=c,
        A=A,
        b=b,
        constraint_signs=constraints.flatten().tolist(),
        var_bounds=x_constraints.tolist(),
        obj_const=obj_const,
    )


class LPInputError(ValueError):
    """Raised when an LP JSON file is invalid."""


_ALLOWED_OBJ = {"max", "min"}
_ALLOWED_SIGNS = {"<=", ">=", "="}
_ALLOWED_BOUNDS = {">=0", "<=0", "free"}


def _require(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise LPInputError(f"Missing required field '{key}'.")
    return data[key]


def _as_1d_float_array(x: Any, name: str) -> np.ndarray:
    try:
        arr = np.array(x, dtype=float)
    except Exception as e:
        raise LPInputError(f"Field '{name}' must be numeric. Parse error: {e}") from e
    if arr.ndim != 1:
        raise LPInputError(
            f"Field '{name}' must be a 1D list/array. Got shape {arr.shape}."
        )
    if arr.size == 0:
        raise LPInputError(f"Field '{name}' must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise LPInputError(
            f"Field '{name}' must contain only finite numbers (no NaN/inf)."
        )
    return arr


def _as_2d_float_array(x: Any, name: str) -> np.ndarray:
    try:
        arr = np.array(x, dtype=float)
    except Exception as e:
        raise LPInputError(
            f"Field '{name}' must be numeric 2D. Parse error: {e}"
        ) from e
    if arr.ndim != 2:
        raise LPInputError(
            f"Field '{name}' must be a 2D list/array. Got shape {arr.shape}."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise LPInputError(
            f"Field '{name}' must have positive dimensions. Got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise LPInputError(
            f"Field '{name}' must contain only finite numbers (no NaN/inf)."
        )
    return arr


def _as_str_list(x: Any, name: str) -> List[str]:
    if not isinstance(x, list) or not all(isinstance(v, str) for v in x):
        raise LPInputError(f"Field '{name}' must be a list of strings.")
    return [v.strip() for v in x]


def load_lp_from_json(path: Union[str, Path]) -> LPProblem:
    path = Path(path)
    if not path.exists():
        raise LPInputError(f"File not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise LPInputError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise LPInputError("Top-level JSON must be an object/dict.")

    obj = str(_require(data, "obj")).strip().lower()
    if obj not in _ALLOWED_OBJ:
        raise LPInputError(f"'obj' must be one of {_ALLOWED_OBJ}. Got '{obj}'.")

    c = _as_1d_float_array(_require(data, "c"), "c")
    A = _as_2d_float_array(_require(data, "A"), "A")

    b_raw = _require(data, "b")
    b_arr = np.array(b_raw, dtype=float)
    if b_arr.ndim == 2 and b_arr.shape[1] == 1:
        b_arr = b_arr.reshape(-1)
    b = _as_1d_float_array(b_arr, "b")

    obj_const = float(data.get("obj_const", 0.0))
    if not np.isfinite(obj_const):
        raise LPInputError("'obj_const' must be finite.")

    constraint_signs = _as_str_list(
        _require(data, "constraint_signs"), "constraint_signs"
    )
    bad_signs = [s for s in constraint_signs if s not in _ALLOWED_SIGNS]
    if bad_signs:
        raise LPInputError(
            f"Invalid entries in 'constraint_signs'. Allowed: {sorted(_ALLOWED_SIGNS)}."
        )

    var_bounds = _as_str_list(_require(data, "var_bounds"), "var_bounds")
    bad_bounds = [v for v in var_bounds if v not in _ALLOWED_BOUNDS]
    if bad_bounds:
        raise LPInputError(
            f"Invalid entries in 'var_bounds'. Allowed: {sorted(_ALLOWED_BOUNDS)}."
        )

    m, n = A.shape
    if c.shape[0] != n:
        raise LPInputError(
            f"len(c) must equal number of columns in A. Got len(c)={c.shape[0]}."
        )
    if b.shape[0] != m:
        raise LPInputError(
            f"len(b) must equal number of rows in A. Got len(b)={b.shape[0]}."
        )
    if len(constraint_signs) != m:
        raise LPInputError(
            f"len(constraint_signs) must equal m={m}. Got {len(constraint_signs)}."
        )
    if len(var_bounds) != n:
        raise LPInputError(f"len(var_bounds) must equal n={n}. Got {len(var_bounds)}.")

    return LPProblem(
        obj=obj,
        c=c,
        A=A,
        b=b,
        constraint_signs=constraint_signs,
        var_bounds=var_bounds,
        obj_const=obj_const,
    )
