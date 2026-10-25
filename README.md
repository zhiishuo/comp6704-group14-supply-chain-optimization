# Carbon-constrained supply-chain linear programming

COMP6704 Group 14 coursework code for a continuous two-echelon network: factories → warehouses → distribution centers. The model minimizes transport and handling costs subject to factory capacity, warehouse flow conservation, demand satisfaction, warehouse throughput, route capacity, and a global carbon budget. All instances use deterministic synthetic, normalized data; there are no facility-location integer decisions or real company records.

The team reports reproducing the supplied code. Preparation of the reference code and documentation involved AI assistance. This repository does not assign individual implementation credit; individual contributions should reflect actual work.

## Methods

Four custom implementations in `network/solvers.py` are compared with two production baselines:

| Implementation | Role |
|---|---|
| Tableau simplex | Educational dense tableau and Phase I |
| Revised simplex | Educational dense basis factorization |
| Sparse predictor-corrector IPM | Sparse Schur-system interior-point method |
| PR-PDHG | Diagonal preconditioning and a residual-based restart heuristic |
| SciPy HiGHS | Production LP baseline and objective reference |
| Native OR-Tools PDLP | Production first-order LP baseline |

PR-PDHG is not the complete PDLP algorithm. Its restart rule carries no new convergence theorem. Dense simplex methods run only on the small cohort, as specified before evaluation.

## Setup and commands

Python 3.12 is recommended. From the repository root:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B network/test_network.py
```

The pinned dependencies are NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.3, OR-Tools 9.14.6206, and threadpoolctl 3.6.0. Execution is CPU-only and needs no credentials or paid services.

The following commands **overwrite archived outputs**. Keep a copy of `network/results/` before running them:

```sh
python -B network/run.py       # full benchmark, ablations, and sensitivity studies
python -B network/verify.py    # certificate audits and HiGHS-IPM cross-checks
mkdir -p report               # generated LaTeX table fragments; not the report manuscript
python -B network/analyze.py  # summary, figures, and table fragments
python -B network/build_guide.py
```

On Windows, create the `report` directory using `mkdir report` if it does not exist. A full benchmark can take minutes to tens of minutes, depending on the machine. `verify.py` also runs auxiliary/reference LP solves. The Chinese guide is available as `project_guide.html`; rebuilding it writes `中文阅读说明.html`. The coursework report and presentation are distributed separately.

## Archived experiments and provenance

`network/protocol.json` fixes sizes, seeds, budgets, tolerances, and the development/evaluation separation. The archive contains 25 evaluation instances: five seeds each at 18, 72, and 162 routes; three each at 450, 3,600, and 14,400; and one at 32,400. Every measured solver/instance has three shuffled timing repetitions. Repetitions are not independent problem instances.

The supplied archive records a Windows 11 CPU run with Python 3.12.14, an Intel64 Family 6 Model 154 Stepping 3 processor, and single-threaded OpenBLAS. See `network/results/environment.json` for versions and BLAS details. Its two machine-specific DLL directory prefixes have been redacted; library filenames, versions, hardware descriptors, thread counts, and measured data are preserved. Archived timings are from that recorded environment, not a new run performed while preparing this repository. The team's reproduction is reported separately and does not establish the origin of every archived timing.

Timing includes setup, iterations, and checks; PDLP model building is included. Instance calibration, serialization, and plotting are excluded. Carbon calibration uses auxiliary LPs whose solutions never initialize compared solvers. Runtime, time-limit iterates, and potentially iteration counts can vary across platforms.

| Archive under `network/results/` | Content |
|---|---|
| `instances/*.npz`, `solutions/*.npz` | Frozen sparse LPs and all final primal/dual vectors |
| `comparison.csv`, `timing_repetitions.csv`, `run.log` | Per-instance comparisons and every timing repetition |
| `histories.json` | Measured custom-method trajectories |
| `ablation.csv`, `carbon_frontier.csv`, `constraint_ablation.csv` | Algorithm diagnostics and model sensitivity |
| `constraint_audit.json`, `verification.json`, `tests.json` | Constraint activity, certificate checks, and archived tests |
| `summary.json`, `index.json` | Aggregated results and experiment index |
| `environment.json`, `source_hashes.json` | Recorded runtime environment and benchmark source identities |

`SHA256SUMS.json` lists the publication package's file hashes. `network/results/source_hashes.json` retains the original experiment source identities. Algorithm source, frozen instances, and numerical result files are unchanged from the supplied package.

## Interpretation and limits

Success requires normalized original-model primal residual, dual-sign residual, and corrected primal-dual gap at most `1e-6`; native solver status alone is insufficient. The archived independent audit checks 390 main certificates, of which 375 meet that threshold. Sparse IPM, PDLP, and HiGHS succeed on 25/25 instances; PR-PDHG succeeds on 20/25; both dense simplex methods succeed on all 15 eligible small instances. Unconverged outputs remain in the archive.

These are comparisons of specific implementations on a calibrated synthetic network family. Presolve, formulations, and production engineering affect runtime. Input storage estimates are not measured peak memory. Shared-machine timings and small instance samples limit statistical and industrial-scale claims. Development results are excluded from the final tables; diagnostic ablations are separate from the held-out benchmark.

No license grant is supplied with this coursework package. Dependency packages retain their own licenses.
