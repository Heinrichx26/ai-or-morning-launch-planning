# AI-OR Morning Launch Planning

This repository contains the reproducibility code for an AI-OR decision-intelligence approach to weather-indexed morning launch planning in transportation operations.

The code implements learning-guided candidate-edge generation, adaptive certificate candidate graphs, min-max regret mixed-integer optimization, and full-regret audit fallback. It also includes scripts for comparator runs, weather-feature construction, stress audits, transferability audits, and deterministic figure generation.

## Repository Contents

- `src/trc_launch/`: data construction, optimization, audit, and plotting code.
- `tests/`: unit and smoke tests for regret separation, certificate closure, weather features, decision-intelligence records, and transferability checks.
- `DATA.md`: public data sources and local directory layout for reproducing the benchmark.
- `experiment_manifest.md`: commands used for smoke, benchmark, and audit runs.

The manuscript source files, submission documents, processed data, raw public-data caches, and result tables are not included in this code repository.

## Public Data Sources

The experiments use publicly available transportation and weather data:

- Bureau of Transportation Statistics On-Time Performance records.
- Iowa Environmental Mesonet airport weather observations, including ASOS, AWOS, and METAR fields.
- Public General Transit Feed Specification feeds from Bay Area Rapid Transit, the Massachusetts Bay Transportation Authority, and TriMet.

See `DATA.md` for source links, expected local folders, and regeneration notes.

## Basic Setup

Use Python 3.11 or later. The scripts rely on common scientific Python packages, including `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `pytest`. Mixed-integer optimization scripts require a solver supported by the local Python environment.

Install the package in editable mode from the repository root:

```powershell
python -m pip install -e .
```

Run the test suite:

```powershell
python -m pytest tests
```

## Reproduction Notes

Before any full experiment, run a small smoke command to confirm that the local data folders and solver are configured correctly. The experiment manifest gives the commands used for the reported audit families. Large benchmark runs download or read public data locally and may take substantial time depending on solver availability and endpoint rate limits.

Generated data and results should be written under local `data/` and `results/` folders. These folders are intentionally ignored by version control.
