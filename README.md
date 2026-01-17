# linear-program-optimization
## About
An implementation of the two-phase simplex algorithm by canonical forms.
This solver accepts linear programs either through interactive manual input or via a JSON file, and runs the two-phase simplex algorithm to report the outcome (Optimal, Unbounded, Infeasible) and the appropriate certificate. 

## Installation 
Clone the repository and install in editable mode:
```bash
pip install -e .
```
For development (tests, linting):
```bash
pip install -e ".[dev]"
```

## Usage
After installation, the ```simplex``` command is available.
### Manual Input
```bash
simplex --manual
```
You will be prompted for:
- Objective function type (Max / Min)
- Objective function coefficients 
- Constraint matrix and right-hand side values
- Constraint signs (>=, <=, =)
- Variable bounds (>=0, <=0, =)

### JSON Input
```bash
simplex --json path/to/lp.json
```
#### Example JSON File
```json
{
  "obj": "max",
  "c": [3, 2],
  "A": [
    [1, 1],
    [1, 0],
    [0, 1]
  ],
  "b": [4, 2, 3],
  "constraint_signs": ["<=", "<=", "<="],
  "var_bounds": [">=0", ">=0"],
  "obj_const": 0.0
}
```

## Output 
Depending on the problem, the solver prints one of:
### Optimal 
```
Optimal value: x.x
Optimal x: [...]
Certificate of Optimality: y = [...]
```
### Unbounded 
```
Unbounded
Certificate of Unboundedness: x + tr = [...] + t*[...]
```
### Infeasible
```
Infeasible
Certificate of Infeasibility: y = [...]
```

## License
MIT License
