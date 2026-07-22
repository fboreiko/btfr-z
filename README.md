# btfr-z

Code behind Boreiko et al. (2026), *Testing subhalo abundance matching with galaxy kinematics* (MNRAS, arXiv).

## What's here

A JAX + MPI forward model that maps the ΛCDM halo population to SPARC `Vmax` via SHAM + halo response + optional selection, plus the machinery to turn it into posteriors:

```
contra/                     Halo-response (contraction/expansion) emulator
  contra_core.py            Modified Gnedin+11 contraction, generalised with nu
  train_emulator.py         Precompute the contra grids -> pickled interpolators
  validate_emulator.py      Check emulated m_dm against direct solves

btfr/                       Forward model + inference
  forward_model.py          JAX/MPI forward model, per-galaxy log-likelihood
  utils/                    Data loading, mass functions, emulator interpolation, plotting
  run_likelihood_grid.py    Evaluate ln L on the (alpha, sigma, [x,] nu) grid
  mcmc_evidence.py          Stitch grids -> emcee -> harmonic evidence -> corner/trace
  run_single_model.py       One parameter point: BTFR prediction + plot (paper Fig. 1)
  run_model_comparison.py   Two points: BTFR / SHMR / per-galaxy dln L panel (Fig. 6)
  run_mock_test.py          Mock-truth injection/recovery (Appendix A / Fig. 7)
  mcmc_sampler_mocktruths.py  MCMC on the mock datasets
  code_for_plotting/        Regenerate the paper figures
  samples/                  Saved evidence outputs (ln Z etc.) as JSON

comparisonBTFR/             Reweight & overlay S21 clustering posteriors (Fig. 5)
Tabular_data/               SPARC-derived tables, mass models, convergence tests
plots/                      Output figures, including plots/paper_plots/
```

## The two emulators

The pipeline is built on **two** precomputed emulators, and they must be built in order:

1. **Contra emulator (halo response).** The contraction equation is expensive to solve per call, so `contra/train_emulator.py` tabulates `log10 m_dm` on a regular `(log c, log fb, log rb, log rf)` grid for each `nu` and pickles it into a dictionary. The forward model reads these via `load_emulators()` and interpolates with `btfr/utils/emulator.py`. **Nothing downstream runs without these grids** — `run_likelihood_grid.py` fails fast if any grid `nu` lacks a trained contra grid.

2. **Likelihood emulator.** With the contra grids in place, `btfr/run_likelihood_grid.py` evaluates the per-galaxy `Vmax` log-likelihood over the parameter grid and writes it to `.npy`. `btfr/mcmc_evidence.py` then interpolates that grid (this *is* the likelihood emulator the MCMC samples), runs `emcee`, and estimates `ln Z` with `harmonic`.

So the hard dependency chain is: **contra grids → likelihood grid → MCMC + evidence.**

## Running

Set up dependencies (JAX with float64, `mpi4py`, `numpy`/`scipy`/`pandas`, `emcee`, `harmonic`, `corner`, `colossus`, `matplotlib`, `tqdm`, and the `BAM` abundance-matching package). Most drivers are MPI and launched with `mpiexec`/`mpirun`.

### 1. Build the contra emulators first

```bash
mpirun -n <nproc> python contra/train_emulator.py
# optional sanity check against direct solves:
python contra/validate_emulator.py
```

Set the `nu` grid inside `train_emulator.py` so it *covers the `nu` grid you intend to use in the likelihood run* (the selection and baseline models use different ranges — see the commented options in the file). If a needed `nu` is missing, the likelihood grid run will stop immediately.

### 2. Compute the likelihood grid (the likelihood emulator)

```bash
# Baseline (3-param, x = 0): single 3D grid
mpiexec -n 4 python btfr/run_likelihood_grid.py   # selection_model = False

# Selection (4-param): chunk over x across independent jobs, then stitch
mpiexec -n 4 python btfr/run_likelihood_grid.py   # selection_model = True, set x_chunk_start
```

Configure via `SparcGridConfig` (model choice, grid ranges/resolution, `n_am_reals`, x-chunking, `output_dir`). For the selection model the x-axis is chunked, so you run several jobs at different `x_chunk_start` and stitch them in the next step.

### 3. Sample the posterior and compute the evidence

```bash
# Baseline (single grid file)
python btfr/mcmc_evidence.py --grid /path/to/baseline_grid.npy \
    --tag 3param_baseline --check-padding

# Selection (stitch the x-chunks first)
python btfr/mcmc_evidence.py \
    --stitch '/path/to/likelihood_grid_..._xchunk_*.npy' \
    --tag 4param_selection --check-padding

# Bayes factor between two finished runs
python btfr/mcmc_evidence.py --delta evidence_4param_selection.json evidence_3param_baseline.json
```

This writes chains, flat samples, corner/trace plots, and an evidence JSON (like those in `btfr/samples/`), giving the posteriors and the `ln Z` used for the baseline-vs-selection Bayes factor.

### Inspect models without a full grid

Once the contra grids exist you can evaluate individual parameter points directly:

```bash
# Single point: predicted vs. observed BTFR (paper Fig. 1)
mpiexec -n 4 python btfr/run_single_model.py

# Two points: BTFR + SHMR + per-galaxy dln L comparison panel (paper Fig. 6)
mpiexec -n 4 python btfr/run_model_comparison.py
```

`run_single_model.py` / `run_model_comparison.py` take their `(alpha, sigma, x, nu)` points from their config dataclasses — handy for eyeballing the baseline vs. selection best-fits from the paper before committing to a grid run.

### Mock validation and clustering comparison

```bash
mpiexec -n 4 python btfr/run_mock_test.py       # Appendix A mocks / Fig. 7
python comparisonBTFR/compare.py                # overlay S21 posteriors / Fig. 5
```

## Notes

- Several scripts contain absolute paths from the author's environment (e.g. the halo catalogue in `btfr/utils/data_loader.py`, a `chdir` in `btfr/utils/mass_functions.py`, and output dirs in `comparisonBTFR/compare.py` and `validate_emulator.py`). Update these to your setup.
- Large binaries (`*.npy`, `*.pkl`, `*.pickle`) are git-ignored, so the contra grids, halo catalogue, and likelihood grids are **not** shipped — you generate them via steps 1–2 above (the halo catalogue must be supplied separately from the Uchuu data) or request the author.
- Data sources: SPARC (astroweb.cwru.edu/SPARC), the Uchuu suite (skiesanduniverses.org), and the NSA catalogue, as described in the paper.

## Citation

If you use this code, please cite Boreiko, Yasin, Desmond, Stiskalek & Jarvis (2026), *Testing subhalo abundance matching with galaxy kinematics*, MNRAS.
