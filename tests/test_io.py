import json
from pathlib import Path

import numpy as np

from simplex.io import LPInputError, load_lp_from_json


def test_load_lp_from_json_ok(tmp_path: Path):
    p = tmp_path / "lp.json"
    p.write_text(json.dumps({
        "obj": "max",
        "c": [3, 2],
        "A": [[1, 1], [1, 0], [0, 1]],
        "b": [4, 2, 3],
        "constraint_signs": ["<=", "<=", "<="],
        "var_bounds": [">=0", ">=0"],
        "obj_const": 0
    }))

    lp = load_lp_from_json(p)
    assert lp.obj == "max"
    assert np.allclose(lp.c, [3, 2])
    assert lp.A.shape == (3, 2)
    assert np.allclose(lp.b, [4, 2, 3])


def test_load_lp_from_json_bad_shapes(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "obj": "max",
        "c": [1, 2],
        "A": [[1, 0]],
        "b": [1, 1],  # wrong length
        "constraint_signs": ["<="],
        "var_bounds": [">=0", ">=0"],
        "obj_const": 0
    }))

    try:
        load_lp_from_json(p)
        assert False, "Expected LPInputError"
    except LPInputError:
        assert True
