# Steel supply-chain climate-risk reproduction package

This repository is for the reproduction of figures and analysis results for paper "Facility-level climate shocks propagate unevenly across the global steel supply chain". Please cite properly if any part of our code and data is used.

This package separates **scientific analysis** from **figure production**.
The analysis is a continuous workflow from initial source inputs through MFA,
facility-level climate impacts, national constraints, weekly dynamic simulation,
domestic supply chain capability (DSCCI), statistics and numerical figure inputs.

| Folder | Purpose |
|---|---|
| `analyse/` | Ordered processing scripts, one shared library, `data/`, `work/` and `result/` |
| `graphing/plotting/` | Accepted figure scripts and editable drawio sources |
| `graphing/data/` | Previously verified numerical inputs for the accepted figures |

Start with [analyse/README.md](analyse/README.md). For drawing only, follow the
README in each figure's `graphing/plotting/` folder (for example,
[Figure 2](graphing/plotting/Figure_2/README.md)); no upstream reconstruction is required.
Figure 1 and Supplementary Figure 1 are authored diagrams: open their editable
sources in draw.io.

## Quick start

Use Python 3.13 (tested on Windows with Python 3.13.5). From this directory:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r analyse/requirements.txt
.venv\Scripts\python analyse/run_analyse.py plan
```

The entry point has one full reconstruction workflow. `plan` only lists the steps
and required inputs. Obtain and prepare the sources in [DATA_ACCESS.md](DATA_ACCESS.md)
before using `check` or `run`. Supplied results under `analyse/data/provided/`
remain available for reference and comparison, not as alternate starting points.
New figure tables go to `analyse/result/graphing_inputs/`; the accepted `graphing/`
files are never overwritten. A repeated run needs `--resume`.

The real miniature engine test is:

```text
.venv\Scripts\python analyse/run_analyse.py smoke
.venv\Scripts\python -m pytest -q analyse/tests -p no:cacheprovider
```

The smoke case is explicitly synthetic, uses weekly steps and writes to
`analyse/work/smoke/`; choose a fresh workspace for another run.

## Full reconstruction and access

```text
python analyse/run_analyse.py plan
python analyse/run_analyse.py check
python analyse/run_analyse.py run
```

The full workflow starts from the study-version normalized source tables and climate files
listed in [DATA_ACCESS.md](DATA_ACCESS.md), including licensed facility records.
Users must obtain and prepare these inputs; the workflow does not automate
conversion of providers' original downloads. No existing MFA,
exposure cache, hazard output, ABM trajectory, author log or temporary folder is an
input to the workflow. Missing data cause a failure, not a fallback to supplied results.

All 411 scenarios and raw climate processing are computationally expensive;
they run only after an explicit `run` command. Previous export validation used
bounded cases, not the full climate ensemble or ABM grid. See
[the analysis README](analyse/README.md#validation-and-release-status) for its scope.

Persistent archive/DOI, code licence and any additional third-party redistribution permissions remain
release decisions. The presence of a source reference is not a grant of rights.
