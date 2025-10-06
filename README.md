# MethodForge

High-precision generators for Gauss–Legendre, Lobatto III C, Radau II A (IRK) and Adams–Bashforth / Adams–Moulton (multistep) coefficients.

MethodForge is a collection of standalone Python scripts for deriving Butcher tables (A, b, c) for classic implicit Runge–Kutta families and high-order multistep schemes.
Each method uses mpmath for arbitrary-precision arithmetic (hundreds of digits), producing reference-grade coefficients suitable for benchmarking, symbolic derivation, or solver validation.

---

## Contents

| File              | Description                                                      |
| ----------------- | ---------------------------------------------------------------- |
| gauss-legendre.py | Builds Gauss–Legendre IRK tables (order = 2 × s)                 |
| lobatto.py        | Builds Lobatto III C IRK tables (order = 2 × s − 2)              |
| radau.py          | Builds Radau II A IRK tables (order = 2 × s − 1)                 |
| multistep.py      | Generates Adams–Bashforth / Adams–Moulton multistep coefficients |

## Installation

Run in your terminal:
pip install mpmath numpy

Requires Python ≥ 3.9.

---

## Usage

Each script can be executed directly.
To change the method order or number of stages, open the script and edit the `for s in (...)` or `test_orders = [...]` line inside the `__main__` section.
Adjust precision by setting `mp.mp.dps = desired_digits`.

---

### Example 1 – Gauss–Legendre IRK

At the bottom of `guass-legendre.py` you’ll see:

```
if __name__ == "__main__":
    mp.mp.dps = 200
    for s in (6, 7):
        A, b, c = build_gauss_legendre_irk(s)
```

Change `(6, 7)` to any list of stages – for example `(4,)`.
Then run:
python guass-legendre.py

The program prints numerical checks and writes:
gauss_legendre_s4_A.txt
gauss_legendre_s4_b.txt
gauss_legendre_s4_c.txt
gauss_legendre_s4_triplets.txt

Each file holds coefficients at your chosen precision.

---

### Example 2 – Lobatto III C IRK

In `lobatto.py`:

```
if __name__ == "__main__":
    mp.mp.dps = 400
    for s in (7, 8):
        A, b, c = build_lobatto_irk(s)
```

Change the stage list and run the file:
python lobatto.py

For s = 8 you’ll get:
lobatto_s8_A.txt
lobatto_s8_b.txt
lobatto_s8_c.txt
lobatto_s8_triplets.txt

---

### Example 3 – Radau II A IRK

In `radau.py`:

```
if __name__ == "__main__":
    mp.mp.dps = 200
    for s in (7, 8):
        A, b, c = build_radau_irk(s)
```

Change the stage count to any value you want and run:
python radau.py

For s = 8 you’ll see:
radau_s8_A.txt
radau_s8_b.txt
radau_s8_c.txt
radau_s8_triplets.txt

---

### Example 4 – Adams Multistep Methods

Open `multistep.py` and edit:
test_orders = [12, 14]

Replace it with any order list (e.g. [6, 8, 10]) and run:
python multistep.py

This creates files like:
ab10_high_precision.txt
am10_high_precision.txt

Each contains the computed coefficients and their order-condition error.

---

## Import Usage in Python

```
from gauss_legendre import build_gauss_legendre_irk
import mpmath as mp

mp.mp.dps = 100
A, b, c = build_gauss_legendre_irk(6)

print("Order:", 12)
print("sum(b) =", sum(b))
```

---

## Features

* Arbitrary-precision arithmetic via mpmath
* Exact quadrature node calculation using Newton iteration
* Automatic correction enforcing A·1 = c
* Portable text outputs for Fortran, Python, Julia, and MATLAB
* Symbolic quadrature integration for Adams coefficients
* User-controlled precision and stage/order settings

---

## Output Format

Each `_triplets.txt` file lists entries as:
i  j  value

First come the A(i,j) rows, then c(i,0) and b(0,j).
All files are plain text and compatible with any language.

---

## License

MIT License — free for research and educational use.

---

## Citation

Sreeram Shankar (2025). MethodForge: High-Precision Generator for Runge–Kutta and Adams Methods. GitHub repository.

---

## Summary

To derive different orders or stage counts:

1. Open the corresponding .py file.
2. Change the stage or order list inside the `for` loop under `__main__`.
3. Modify `mp.mp.dps` to set desired precision.
4. Run the script to generate new coefficient files in the current directory.

You can now generate any IRK or multistep coefficients at arbitrary precision.
