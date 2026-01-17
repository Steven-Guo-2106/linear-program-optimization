"""
cli.py

Command-line interface for the two-phase simplex solver.
"""

import argparse
from pathlib import Path

from .io import LPInputError, get_inputs, load_lp_from_json
from .lp_types import LPStatus
from .two_phase import two_phase_simplex


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simplex", description="Two-phase simplex solver")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--manual", action="store_true", help="Enter LP manually (interactive)"
    )
    g.add_argument("--json", type=Path, help="Load LP from a JSON file")

    p.add_argument(
        "--verbose", action="store_true", help="Verbose output (if supported)"
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.manual:
        lp = get_inputs()
    else:
        try:
            lp = load_lp_from_json(args.json)
        except LPInputError as e:
            print(f"Input error: {e}")
            return 1

    res = two_phase_simplex(lp)

    if res.status == LPStatus.OPTIMAL:
        print("Optimal value: ", res.objective_value)
        print("Optimal x = ", res.x)
        print("Certificate of optimality: y = ", res.y)
        return 0

    elif res.status == LPStatus.UNBOUNDED:
        print("Unbounded")
        print("Certificate of Unboundedness: x+tr = ", res.x, "+ t *", res.ray)
        return 0

    elif res.status == LPStatus.INFEASIBLE:
        print("Infeasible")
        print("Certificate of Infeasibility: y = ", res.y)
        return 0

    print(f"Solver status: {res.status.value}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
