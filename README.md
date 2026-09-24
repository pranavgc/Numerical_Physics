# Numerical Physics

Solutions to the weekly worksheets of the **Numerical Physics** course at IISER Kolkata.

Nine worksheets covering classical integrators, random number generation, stochastic dynamics, boundary value problems, and eigenvalue algorithms. Every method is implemented from scratch with NumPy — no `scipy.integrate`, no `numpy.linalg.eig` — and almost every solution is validated against a closed-form result.

---

## Contents

| # | Folder | Topic | Scripts |
|---|--------|-------|---------|
| 1 | [`WS01_Verlet`](WS01_Verlet) | Euler, Euler-Cromer and Verlet integrators; energy conservation and O(Δt²) convergence | `p1.py`, `p2.py` |
| 2 | [`WS02_RK4`](WS02_RK4) | RK4 for the damped oscillator (critical damping) and the driven oscillator (resonance curves) | `p1.py`, `p2.py` |
| 3 | [`WS03_box_muller`](WS03_box_muller) | Uniform, Gaussian (Box-Muller) and Lorentzian (inverse transform) random variates | `p1.py`, `p2.py`, `p3.py` |
| 4 | [`WS04_Kuramoto`](WS04_Kuramoto) | Kuramoto model of N coupled oscillators; order parameter vs coupling strength | `p1.py` |
| 5 | [`WS05_Random_walk`](WS05_Random_walk) | 1-D random walk (Var ∝ N) and the diffusion equation by Euler and RK4 | `p1.py`, `p2E.py`, `p2RK4.py` |
| 6 | [`WS06_sde`](WS06_sde) | Euler-Maruyama for the Black-Scholes SDE and a stochastic birth-death process | `p1.py`, `p2.py` |
| 7 | [`WS07-shooting`](WS07-shooting) | Shooting method with secant root-finding; single- and double-sided | `p1.py`, `p2a.py`, `p2b.py` |
| 8 | [`WS08-Numerov`](WS08-Numerov) | Numerov integration of the 1-D Schrödinger equation; infinite and finite wells | `p1.py`, `p2.py` |
| 9 | [`WS09-power_method`](WS09-power_method) | Power method with deflation, and QR iteration via Gram-Schmidt | `p1.py`, `p2.py` |

Each folder is self-contained and holds:

- `WSn.pdf` — the problem sheet
- `pN.py` — the solution to Problem N
- `*.png` — the figures the scripts produce
- lecture notes for the method, where they were supplied (`verlet.pdf`, `RungeKutta4.pdf`, `boxmuller.pdf`, `sde.pdf`, `GS_QR.pdf`, `mq_numerov1.pdf`, `Ln.pdf`)

---

## What each worksheet does

**WS01 — Integrators.** Compares plain Euler against Euler-Cromer on the harmonic oscillator by plotting total energy over time: Euler drifts, the symplectic variant stays bounded. The second script runs Verlet and measures the velocity error as a function of step size, confirming second-order convergence.

**WS02 — RK4.** Fourth-order Runge-Kutta applied to θ̈ + 2λω₀θ̇ + ω₀²θ = 0. Sweeping λ downward locates the critical damping value. The driven case adds F₀sin(ωt), extracts the steady-state amplitude after discarding the transient, and traces resonance curves A(ω/ω₀) for several damping ratios — the resonance peak shifting downward with damping as ω_r = ω₀√(1−2λ²).

**WS03 — Random numbers.** Builds a normalised PDF from raw uniform variates by binning. Generates Gaussians by Box-Muller and Lorentzians by inverse transform, each overlaid on its analytic density for three parameter choices.

**WS04 — Kuramoto.** N oscillators with intrinsic frequencies drawn from a Cauchy distribution, coupled through the mean field. Integrates all N phases as a single vectorised RK4 and plots the steady-state order parameter |r| against coupling K for N = 100, 200, 400, showing the synchronisation transition.

**WS05 — Diffusion.** An ensemble of 1-D random walkers, with the variance of the final position confirmed to grow linearly in the number of steps. The continuum counterpart solves ∂ρ/∂t = D∂²ρ/∂x² on a finite-difference grid by both Euler and RK4 time-stepping.

**WS06 — Stochastic differential equations.** Euler-Maruyama integration of geometric Brownian motion (Black-Scholes), with the numerical path compared to the exact log-normal solution and the RMS error traced against dt. The second script simulates a birth-death process with state-dependent noise over 100 realisations, compares the ensemble mean to the deterministic rate equation, and fits the steady-state occupancy distribution to a Poisson.

**WS07 — Boundary value problems.** Converts two-point BVPs into initial value problems by guessing the initial slope and refining it with a secant iteration. Applied to the linear problem y″ + 4y = 0 and the nonlinear y″ = −y + 2y′²/y, both from one end and by shooting from both ends and matching in the middle.

**WS08 — Schrödinger.** Numerov's method — which exploits the absence of a first-derivative term to reach O(Δx⁶) — applied to the 1-D Schrödinger equation in Hartree units. Scans energy until the wavefunction satisfies the boundary condition, recovering the ground state of the infinite and finite square wells.

**WS09 — Eigenvalues.** Power iteration with scaling for the dominant eigenpair, then deflation for the subdominant one. Separately, QR iteration built on a hand-written Gram-Schmidt factorisation, driving the matrix toward upper-triangular form until the off-diagonal elements fall below tolerance.

---

## Running the code

**Requirements:** Python 3, NumPy, Matplotlib.

```bash
pip install numpy matplotlib
```

Each script is standalone and opens a Matplotlib window when run:

```bash
cd WS08-Numerov
python p1.py
```

Two scripts in `WS09-power_method` read the matrix from stdin:

```
order of matrix = 3
Enter 0th row = 1 3 4
Enter 1th row = 3 1 2
Enter 2th row = 4 2 1
```

Parameters (step sizes, coupling constants, well depths, energy brackets) are literals near the top of each file — edit them in place to explore other cases.

---

## Status

These scripts were written between 2018 and 2019, spanning the Python 2 → Python 3 transition, and have not been maintained since. **Several do not run on a current Python/NumPy stack.** Known breakages in the *original* files:

| Script | Issue |
|---|---|
| `WS02_RK4/p1.py` | Python 2 `print` statement |
| `WS03_box_muller/p1.py` | Python 2 `print` statement |
| `WS06_sde/p2.py` | Python 2 `print` statement |
| `WS07-shooting/p1.py` | Ragged `np.array(...)` — an error since NumPy 1.24 |
| `WS07-shooting/p2b.py` | Ragged `np.array(...)` — an error since NumPy 1.24 |
| `WS09-power_method/p1.py` | `SyntaxError` — missing comma in a `print` call |
| `WS09-power_method/p2.py` | `NameError` — undefined variable in the main loop |

The remaining 13 scripts run on Python 3.11 with NumPy 2.x, though several produce incorrect numbers. The header comment in each file records the interpreter it was originally written for.

### Corrected versions

Every original `pN.py` has a `pN_python3.py` beside it. These run on Python 3.11 / NumPy 2.x with no warnings, fix the numerical bugs, and complete the parts of each worksheet the originals left out — while keeping every algorithm hand written (no `scipy`, no `numpy.linalg`, no library solvers). Each one prints a check against an analytic result. The originals are untouched.

See [`PYTHON3_VERSIONS.md`](PYTHON3_VERSIONS.md) for the full list of changes and verified results, and [`CODE_REVIEW.md`](CODE_REVIEW.md) for the per-problem audit of the originals.

---

## Verified results

Where an analytic answer exists, the code has been checked against it:

| Quantity | Computed | Analytic | Source |
|---|---|---|---|
| Infinite-well ground state (L = 10, Hartree) | 0.0493480220 | 0.0493480220 | `WS08-Numerov/p1_python3.py` |
| Finite-well ground state (V₀ = 2, L = 10, Hartree) | −1.9593161 | −1.9592419 | `WS08-Numerov/p2_python3.py` |
| Resonant frequency (ω₀ = 3, λ = 0.1) | 2.9689 | 2.9698 | `WS02_RK4/p2_python3.py` |
| Critical damping λ_c | 0.99979 | 1 | `WS02_RK4/p1_python3.py` |
| Verlet velocity-error order | 2.000 | 2 | `WS01_Verlet/p2_python3.py` |
| RK4 shooting error order | 3.976 | 4 | `WS07-shooting/p1_python3.py` |
| Euler-Maruyama strong order | 0.498 | 0.5 | `WS06_sde/p1_python3.py` |
| Random walk Var(x) vs N slope | 0.9939 | 1 | `WS05_Random_walk/p1_python3.py` |

Figures are from the corrected scripts. The finite-well residual is not Numerov truncation error: it halves exactly as the grid halves, which identifies it as the O(h) effect of the potential step falling between grid points.

---

## Licence

No licence specified. Coursework, shared for reference.
