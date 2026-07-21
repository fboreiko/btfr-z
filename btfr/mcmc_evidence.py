"""
Stitching + MCMC + harmonic-evidence pipeline for the SPARC SHAM grids.

Implements the six required changes on top of the old mcmc_sampler.py /
evidence.py scripts:

  1. Normalised prior inside the log-posterior:
         ln_post(theta) = interp_lnL(theta) + ln_prior(theta)
     with ln_prior = -ln(V_full) inside the FULL stated prior box (paper
     Table 1) and -inf outside.  V_full is computed from the stated priors,
     NOT from the grid box, so evidences are comparable across models and
     to Stiskalek+21.
  2. The grid is padded out to the full prior box with a large negative
     FINITE filler (default -1e6) at the truncated grid's own node spacing,
     so multilinear interpolation never mixes finite and -inf corners
     (-> NaN -> emcee raises) and never returns an accidentally-harmless 0.
  3. harmonic is fed ln(L*pi) directly from the sampler:
     get_chain()/get_log_prob() (already ln L + ln pi after change 1),
     transposed to (nwalkers, nsteps, ...) with walkers as harmonic chains.
     L*pi is never reconstructed outside the sampler.
  4. Consistency checks: per-face truncation-mass check against the old
     full-range grid (threshold e^-7 of total mass), and a padding /
     normalisation check (control run confined to the grid box must give
     ln Z_main = ln Z_box - ln(V_full / V_box) within harmonic's error
     bars; for the baseline half-alpha-range grid this difference is
     ~ln 2 = 0.693 when alpha is the only truncated dimension).

Output ---
    <samples_dir>/full_chain_<tag>.npy    sampler.get_chain()        (nsteps, nwalkers, ndim)
    <samples_dir>/log_prob_<tag>.npy      sampler.get_log_prob()     (nsteps, nwalkers)
    <samples_dir>/flat_samples_<tag>.npy  sampler.get_chain(flat=True)
    <plots_dir>/corner_plot_<tag>.png
    <plots_dir>/trace_plots_<tag>.png

Typical usage
-------------
# 1) stitch selection-model x-chunks, then sample + evidence:
python mcmc_evidence.py \
    --stitch '/path/to/likelihood_grid_100am_100postsel_shape_15x15x20x25_*_xchunk_*.npy' \
    --tag 4param_selection \
    --old-full-grid /path/to/old_20x20x20x20_grid.npy \
    --check-padding

# 2) baseline (already a single 3D file):
python mcmc_evidence.py \
    --grid /path/to/likelihood_grid_baseline_x0_..._shape_15x15x25_....npy \
    --tag 3param_baseline --check-padding

# 3) Delta ln Z between two finished runs:
python mcmc_evidence.py --delta evidence_4param_selection.json evidence_3param_baseline.json
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp
import emcee

# Stated priors (paper Table 1)
ALPHA_EPS = 0.01  # epsilon inset on tan^-1(alpha); grid code uses the same value
PRIORS = {
    "alpha": (-np.pi / 2 + ALPHA_EPS, np.pi / 2 - ALPHA_EPS),  # tan^-1(alpha)
    "scatter": (0.0, 1.0),                                     # sigma_SHAM
    "x": (0.01, 0.99),                                         # selection threshold
    "nu": (-3.0, 3.0),                                         # nu
}

PARAM_ORDER_4 = ["alpha", "scatter", "x", "nu"]   # selection-grid axis order
PARAM_ORDER_3 = ["alpha", "scatter", "nu"]        # baseline-grid axis order

LABELS = {
    "alpha": r"$\tan^{-1}(\alpha)$",
    "scatter": r"$\sigma$",
    "x": r"$x$",
    "nu": r"$\nu$",
}

FILLER = -1.0e6         # large negative FINITE lnL filler
TRUNC_THRESHOLD = np.exp(-7.0)  # allowed out-of-box mass fraction per face

FLOAT_RE = r"(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)"


def parse_grid_filename(path):
    """Parse model type, per-axis ranges, shape tag and chunk info from a
    likelihood-grid filename (new `shape_...` format; falls back to the
    array shape for old-format names without a shape tag)."""
    name = os.path.basename(path)
    meta = {
        "path": path,
        "name": name,
        "baseline": "baseline_x0" in name,
        "vmaxshift": "_vmaxshift" in name,
        "ranges": {},
        "shape_tag": None,
        "xchunk": None,
    }

    for key, token in [("alpha", "alphaproxy"), ("scatter", "scatter"),
                       ("x", "x"), ("nu", "nu")]:
        m = re.search(rf"_{token}_{FLOAT_RE}_{FLOAT_RE}", name)
        if m:
            meta["ranges"][key] = (float(m.group(1)), float(m.group(2)))

    m = re.search(r"shape_(\d+)x(\d+)x(\d+)x(\d+)", name)
    if m:
        meta["shape_tag"] = tuple(int(g) for g in m.groups())
    else:
        m = re.search(r"shape_(\d+)x(\d+)x(\d+)", name)
        if m:
            meta["shape_tag"] = tuple(int(g) for g in m.groups())

    m = re.search(r"xchunk_(\d+)to(\d+)", name)
    if m:
        meta["xchunk"] = (int(m.group(1)), int(m.group(2)))

    if meta["baseline"] and "x" in meta["ranges"]:
        # 'x_' regex can false-positive on baseline names; baseline has no x axis
        del meta["ranges"]["x"]

    meta["param_order"] = PARAM_ORDER_3 if meta["baseline"] else PARAM_ORDER_4
    missing = [p for p in meta["param_order"] if p not in meta["ranges"]]
    if missing:
        raise ValueError(f"Could not parse ranges for {missing} from '{name}'")
    return meta


def build_axes(meta, grid_shape):
    """Reconstruct per-axis node arrays from filename ranges + shape.

    The shape tag is authoritative when present; otherwise the array shape
    is used.  nu nodes within 1e-12 of zero are snapped to exactly 0.0
    (the grid code's linspace can put ~1.1e-16 where 0.0 is meant)."""
    n_axes = meta["shape_tag"] if meta["shape_tag"] is not None else grid_shape
    if tuple(n_axes) != tuple(grid_shape):
        raise ValueError(
            f"Filename shape tag {n_axes} != loaded array shape {grid_shape} "
            f"for {meta['name']}"
        )
    axes = []
    for dim, param in enumerate(meta["param_order"]):
        lo, hi = meta["ranges"][param]
        nodes = np.linspace(lo, hi, grid_shape[dim])
        if param == "nu":
            nodes[np.abs(nodes) < 1e-12] = 0.0
        axes.append(nodes)
    return axes


def stitch_chunks(pattern, out_dir=None):
    """Concatenate x-chunk files along axis 2 (alpha, scatter, x, nu).

    Chunks are only combined if every non-chunk config field encoded in the
    filename is identical; the result must be contiguous in x, start at
    chunk 0, cover all n_x nodes, and match the filename shape tag."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No chunk files match pattern: {pattern}")
    metas = [parse_grid_filename(p) for p in paths]

    for m in metas:
        if m["baseline"]:
            raise ValueError(f"{m['name']} is a baseline grid; nothing to stitch.")
        if m["xchunk"] is None:
            raise ValueError(f"{m['name']} has no xchunk tag; is it already combined?")

    # all non-chunk filename fields must be identical
    def strip_chunk(name):
        return re.sub(r"_xchunk_\d+to\d+", "", name)

    base_names = {strip_chunk(m["name"]) for m in metas}
    if len(base_names) != 1:
        raise ValueError(
            "Chunk files differ in non-chunk config fields; refusing to stitch:\n  "
            + "\n  ".join(sorted(base_names))
        )
    combined_name = base_names.pop()

    metas.sort(key=lambda m: m["xchunk"][0])
    n_x_full = metas[0]["shape_tag"][2]

    arrays, expected_start = [], 0
    for m in metas:
        s, e = m["xchunk"]
        if s != expected_start:
            raise ValueError(
                f"x-chunks not contiguous: expected chunk starting at "
                f"{expected_start}, got {s} ({m['name']})"
            )
        arr = np.load(m["path"])
        if arr.ndim != 4:
            raise ValueError(f"{m['name']}: expected 4D chunk, got {arr.shape}")
        if arr.shape[2] != e - s + 1:
            raise ValueError(
                f"{m['name']}: x-extent {arr.shape[2]} != xchunk tag {s}to{e}"
            )
        want = (m["shape_tag"][0], m["shape_tag"][1], e - s + 1, m["shape_tag"][3])
        if arr.shape != want:
            raise ValueError(f"{m['name']}: shape {arr.shape} != expected {want}")
        arrays.append(arr)
        expected_start = e + 1

    if expected_start != n_x_full:
        raise ValueError(
            f"x-chunks cover indices 0..{expected_start - 1} but shape tag says "
            f"n_x = {n_x_full}"
        )

    combined = np.concatenate(arrays, axis=2)
    assert combined.shape == metas[0]["shape_tag"], (
        f"Stitched shape {combined.shape} != filename shape tag "
        f"{metas[0]['shape_tag']}"
    )

    out_dir = Path(out_dir) if out_dir else Path(paths[0]).parent
    out_path = out_dir / combined_name
    np.save(out_path, combined)
    print(f"[stitch] {len(arrays)} chunks -> {combined.shape} -> {out_path}")
    return str(out_path)


# Grid cleaning (NaN / -inf handling before interpolation)
def average_neighbors(arr, index):
    """Mean of finite neighbours (including diagonals) of one cell."""
    dims = len(index)
    vals = []
    for offsets in np.ndindex(*([3] * dims)):
        off = tuple(o - 1 for o in offsets)
        if all(o == 0 for o in off):
            continue
        nb = tuple(index[d] + off[d] for d in range(dims))
        if all(0 <= nb[d] < arr.shape[d] for d in range(dims)):
            v = arr[nb]
            if np.isfinite(v):
                vals.append(v)
    return np.mean(vals) if vals else np.nan


def clean_grid(grid, filler=FILLER):
    """Replace non-finite cells so interpolation stays finite everywhere.

    - -inf cells (linear-space MPI reduction underflow in the grid code)
      are replaced by the finite filler directly: they mean 'astronomically
      unlikely', and neighbour-averaging them would fabricate support.
    - NaN cells (failed grid points) are filled iteratively by neighbour
      averaging; any that cannot be filled fall back to the filler.
    All replacements are reported so they can't silently leak into
    sampled regions."""
    grid = grid.copy()

    n_neginf = int(np.sum(np.isneginf(grid)))
    if n_neginf:
        idx = np.argwhere(np.isneginf(grid))
        print(f"[clean] replacing {n_neginf} -inf cells with filler {filler:.3g}:")
        for i in idx[:20]:
            print(f"        {tuple(i)}")
        if n_neginf > 20:
            print(f"        ... and {n_neginf - 20} more")
        grid[np.isneginf(grid)] = filler

    n_nan = int(np.sum(np.isnan(grid)))
    if n_nan:
        print(f"[clean] filling {n_nan} NaN cells by neighbour averaging...")
        for _ in range(1000):
            nan_idx = np.argwhere(np.isnan(grid))
            if nan_idx.size == 0:
                break
            filled = 0
            for i in nan_idx:
                v = average_neighbors(grid, tuple(i))
                if np.isfinite(v):
                    print(f"        {tuple(i)} <- neighbour mean {v:.4f}")
                    grid[tuple(i)] = v
                    filled += 1
            if filled == 0:
                print(f"[clean] {len(nan_idx)} NaNs have no finite neighbours; "
                      f"using filler {filler:.3g}")
                grid[np.isnan(grid)] = filler
                break

    assert np.all(np.isfinite(grid)), "grid still contains non-finite values"
    return grid


# Padding to the full prior box (change 2)
def pad_axis_nodes(nodes, lo, hi):
    """Extend uniform nodes at their OWN spacing until they cover [lo, hi].
    Returns (new_nodes, n_left, n_right)."""
    d = nodes[1] - nodes[0]
    n_left = max(0, int(np.ceil((nodes[0] - lo) / d - 1e-12)))
    n_right = max(0, int(np.ceil((hi - nodes[-1]) / d - 1e-12)))
    left = nodes[0] - d * np.arange(n_left, 0, -1)
    right = nodes[-1] + d * np.arange(1, n_right + 1)
    return np.concatenate([left, nodes, right]), n_left, n_right


def pad_grid_to_prior(grid, axes, param_order, priors=PRIORS, filler=FILLER):
    """Pad every dimension out to the full prior box with the filler,
    keeping each truncated grid's own node spacing (no mixing of spacings
    from different runs)."""
    pad_widths, new_axes = [], []
    for dim, param in enumerate(param_order):
        lo, hi = priors[param]
        nodes, nl, nr = pad_axis_nodes(axes[dim], lo, hi)
        pad_widths.append((nl, nr))
        new_axes.append(nodes)
        if nl or nr:
            print(f"[pad] {param}: grid [{axes[dim][0]:.4f}, {axes[dim][-1]:.4f}] "
                  f"-> padded [{nodes[0]:.4f}, {nodes[-1]:.4f}] "
                  f"(+{nl} low, +{nr} high nodes at spacing "
                  f"{axes[dim][1] - axes[dim][0]:.4f})")
    padded = np.pad(grid, pad_widths, mode="constant", constant_values=filler)
    return padded, new_axes


def make_interpolator(axes, grid, filler=FILLER):
    return RegularGridInterpolator(
        tuple(axes), grid, bounds_error=False,
        fill_value=filler,  # finite, never -inf (change 2)
    )


# Log-posterior with normalised prior (change 1)
def prior_volume(param_order, priors=PRIORS):
    v = 1.0
    for p in param_order:
        lo, hi = priors[p]
        v *= (hi - lo)
    return v


def make_log_posterior(interp, prior_box, ln_prior_const):
    """ln_post = interp_lnL + ln_prior; ln_prior = ln_prior_const inside
    prior_box (list of (lo, hi) per dim), -inf outside."""
    los = np.array([b[0] for b in prior_box])
    his = np.array([b[1] for b in prior_box])

    def log_posterior(theta):
        theta = np.asarray(theta, dtype=float)
        if np.any(theta < los) or np.any(theta > his):
            return -np.inf
        lnl = float(interp(theta)[0])
        if np.isnan(lnl):  # should never happen after clean+pad; hard guard
            return -np.inf
        return lnl + ln_prior_const

    return log_posterior


# Sampling
def run_sampler(log_posterior, init_box, ndim, nwalkers=100, warmup=5000,
                max_steps=100000, check_interval=100, seed=None):
    """Warm-up + reset, then sample with the same convergence criteria as
    the old scripts (tau*100 < iteration and |dtau|/tau < 0.005)."""
    rng = np.random.default_rng(seed)
    initial_pos = np.column_stack([
        rng.uniform(lo, hi, size=nwalkers) for lo, hi in init_box
    ])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    print("Running warm-up phase...")
    sampler.run_mcmc(initial_pos, warmup, progress=True)
    sampler.reset()  # discard burn-in; everything kept below is post-burn

    old_tau = np.full(ndim, np.inf)
    print("Running main sampling with convergence checks...")
    for _ in range(max_steps // check_interval):
        sampler.run_mcmc(None, check_interval, progress=True)
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            print("...still too few samples to estimate tau; continuing.")
            continue
        iteration = sampler.iteration
        crit1 = np.all(tau * 100 < iteration)
        crit2 = np.all(np.abs(old_tau - tau) / tau < 0.005)
        print(f"Iteration={iteration}, tau={np.round(tau, 1)}")
        if crit1 and crit2:
            print("Chains have likely converged.")
            break
        old_tau = tau.copy()
    return sampler


# harmonic evidence (changes 3 + 4)
def harmonic_ln_evidence(sampler, discard=0, temperature=0.8, epochs=20,
                         training_proportion=0.5, verbose=True):
    """Feed ln(L*pi) straight from emcee into harmonic and return ln Z.

    emcee's get_chain()/get_log_prob() are (nsteps, nwalkers, ...); harmonic
    wants (nchains, nsamples, ...) with walkers as chains, so both arrays
    are transposed -- never reconstructed (change 3).

    harmonic estimates the INVERSE evidence; ln Z = -ln_evidence_inv, and
    the asymmetric error bar (e_minus, e_plus) on ln(1/Z) maps to
    (-e_plus, -e_minus) on ln Z (change 4).  If the installed harmonic
    version ever changes this convention, the printed version string below
    is the thing to check against its docs.
    """
    import harmonic as hm
    print(f"[harmonic] version {getattr(hm, '__version__', 'unknown')} "
          f"(sign convention: ev.ln_evidence_inv = ln(1/Z); verify once per "
          f"version)")

    chain = sampler.get_chain(discard=discard)        # (nsteps, nwalkers, ndim)
    log_prob = sampler.get_log_prob(discard=discard)  # (nsteps, nwalkers) = ln L + ln pi
    samples = np.ascontiguousarray(np.swapaxes(chain, 0, 1))
    lnp = np.ascontiguousarray(np.swapaxes(log_prob, 0, 1))

    ndim = samples.shape[-1]
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, lnp)
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=training_proportion
    )

    model = hm.model.RQSplineModel(ndim, standardize=True, temperature=temperature)
    model.fit(chains_train.samples, epochs=epochs, verbose=verbose)

    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)

    ln_inv = float(ev.ln_evidence_inv)
    e_minus, e_plus = ev.compute_ln_inv_evidence_errors()  # on ln(1/Z); e_minus<0<e_plus

    ln_z = -ln_inv
    ln_z_err = (-float(e_plus), -float(e_minus))  # (negative, positive) error on ln Z
    return ln_z, ln_z_err


# Plots + saving (kept identical to the old scripts' conventions)
def save_outputs(sampler, param_order, tag, samples_dir, plots_dir, usetex=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import corner

    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Computer Modern"]
    rcParams["text.usetex"] = usetex

    samples_dir = Path(samples_dir); samples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(plots_dir); plots_dir.mkdir(parents=True, exist_ok=True)

    ndim = len(param_order)
    labels = [LABELS[p] for p in param_order]

    flat = sampler.get_chain(flat=True)
    fig = corner.corner(
        flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
        label_kwargs={"fontsize": 14}, title_kwargs={"fontsize": 14},
    )
    for ax in fig.get_axes():
        ax.tick_params(labelsize=14)
    fig.savefig(plots_dir / f"corner_plot_{tag}.png", dpi=300)
    plt.close(fig)

    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    fig, axes = plt.subplots(ndim, 1, figsize=(8, 2 * ndim), sharex=True)
    iters = np.arange(chain.shape[0])
    for i in range(ndim):
        for w in range(chain.shape[1]):
            axes[i].plot(iters, chain[:, w, i], alpha=0.3)
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    fig.tight_layout()
    fig.savefig(plots_dir / f"trace_plots_{tag}.png", dpi=300)
    plt.close(fig)

    # same file names / array conventions as before
    np.save(samples_dir / f"full_chain_{tag}.npy", sampler.get_chain())
    np.save(samples_dir / f"log_prob_{tag}.npy", sampler.get_log_prob())
    np.save(samples_dir / f"flat_samples_{tag}.npy", flat)
    print(f"[save] chains -> {samples_dir}/(full_chain|log_prob|flat_samples)_{tag}.npy")
    print(f"[save] plots  -> {plots_dir}/(corner_plot|trace_plots)_{tag}.png")

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    print("\nPosterior summary (median +/- (q84-q16)/2):")
    for i, p in enumerate(param_order):
        print(f"  {p:8s}: {q50[i]:.4f} +/- {(q84[i] - q16[i]) / 2:.4f} "
              f"(16/84: {q16[i]:.4f}/{q84[i]:.4f})")


# Delta ln Z between two saved runs
def delta_ln_z(json_a, json_b):
    with open(json_a) as f:
        a = json.load(f)
    with open(json_b) as f:
        b = json.load(f)
    d = a["ln_z"] - b["ln_z"]
    err = np.sqrt(
        max(abs(a["ln_z_err"][0]), abs(a["ln_z_err"][1])) ** 2
        + max(abs(b["ln_z_err"][0]), abs(b["ln_z_err"][1])) ** 2
    )
    print(f"ln Z [{a['tag']}] = {a['ln_z']:.4f} "
          f"({a['ln_z_err'][0]:+.4f}/{a['ln_z_err'][1]:+.4f})")
    print(f"ln Z [{b['tag']}] = {b['ln_z']:.4f} "
          f"({b['ln_z_err'][0]:+.4f}/{b['ln_z_err'][1]:+.4f})")
    print(f"Delta ln Z = {d:.4f} +/- {err:.4f}  (positive favours {a['tag']})")


# Main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", help="combined likelihood grid .npy "
                    "(selection 4D or baseline 3D; ranges parsed from filename)")
    ap.add_argument("--stitch", metavar="GLOB",
                    help="glob pattern of x-chunk files to stitch first; the "
                    "stitched file is then used as --grid")
    ap.add_argument("--stitch-only", action="store_true",
                    help="stop after stitching")
    ap.add_argument("--tag", default=None,
                    help="output tag (default: '4param' / '3param' "
                    "[+ '_vmaxshift'] from the grid filename)")
    ap.add_argument("--samples-dir", default="samples")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--nwalkers", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--check-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--filler", type=float, default=FILLER)
    ap.add_argument("--no-evidence", action="store_true",
                    help="skip harmonic (sampling + saving only)")
    ap.add_argument("--harmonic-discard", type=int, default=0,
                    help="extra steps to discard before harmonic (warm-up is "
                    "already dropped via sampler.reset())")
    ap.add_argument("--harmonic-temperature", type=float, default=0.8)
    ap.add_argument("--harmonic-epochs", type=int, default=20)
    ap.add_argument("--no-usetex", action="store_true",
                    help="disable LaTeX text rendering in plots")
    ap.add_argument("--delta", nargs=2, metavar=("A.json", "B.json"),
                    help="print Delta ln Z = A - B from two saved evidence "
                    "json files and exit")
    args = ap.parse_args()

    if args.delta:
        delta_ln_z(*args.delta)
        return

    grid_path = args.grid
    if args.stitch:
        grid_path = stitch_chunks(args.stitch)
        if args.stitch_only:
            return
    if not grid_path:
        ap.error("provide --grid or --stitch")

    meta = parse_grid_filename(grid_path)
    grid = np.load(grid_path)
    param_order = meta["param_order"]
    ndim = len(param_order)
    model_name = "baseline (3-param)" if meta["baseline"] else "selection (4-param)"
    print(f"Loaded {model_name} grid {grid.shape} from {meta['name']}")

    tag = args.tag
    if tag is None:
        tag = "3param" if meta["baseline"] else "4param"
        if meta["vmaxshift"]:
            tag += "_vmaxshift"

    axes = build_axes(meta, grid.shape)
    grid_box = [(ax[0], ax[-1]) for ax in axes]

    # ---- prior reconciliation table (change 6) ----
    v_full = prior_volume(param_order)
    print("\nPrior reconciliation (paper Table 1 must state exactly these):")
    print(f"  {'param':8s} {'grid box':>24s} {'stated prior':>24s}")
    for dim, p in enumerate(param_order):
        glo, ghi = grid_box[dim]
        plo, phi = PRIORS[p]
        print(f"  {p:8s} [{glo:10.4f}, {ghi:10.4f}] [{plo:10.4f}, {phi:10.4f}]")
    print(f"  V_full (prior normalisation volume) = {v_full:.6f}; "
          f"ln_prior = {-np.log(v_full):.6f} inside the prior box")

    # ---- clean + pad + interpolate (change 2) ----
    grid = clean_grid(grid, filler=args.filler)
    padded, padded_axes = pad_grid_to_prior(grid, axes, param_order,
                                            filler=args.filler)
    interp = make_interpolator(padded_axes, padded, filler=args.filler)

    # ---- posterior + sampling (change 1) ----
    prior_box = [PRIORS[p] for p in param_order]
    log_post = make_log_posterior(interp, prior_box, -np.log(v_full))

    # start walkers inside the computed grid box (5% inset) so none begin
    # on the flat filler plateau
    init_box = []
    for dim in range(ndim):
        lo = max(grid_box[dim][0], prior_box[dim][0])
        hi = min(grid_box[dim][1], prior_box[dim][1])
        pad = 0.05 * (hi - lo)
        init_box.append((lo + pad, hi - pad))

    sampler_kwargs = dict(nwalkers=args.nwalkers, warmup=args.warmup,
                          max_steps=args.max_steps,
                          check_interval=args.check_interval, seed=args.seed)
    sampler = run_sampler(log_post, init_box, ndim, **sampler_kwargs)

    save_outputs(sampler, param_order, tag, args.samples_dir, args.plots_dir,
                 usetex=not args.no_usetex)

    if args.no_evidence:
        return

    # ---- harmonic evidence (changes 3 + 4) ----
    harmonic_kwargs = dict(discard=args.harmonic_discard,
                           temperature=args.harmonic_temperature,
                           epochs=args.harmonic_epochs)
    ln_z, ln_z_err = harmonic_ln_evidence(sampler, **harmonic_kwargs)

    print(f"\nln Z ({tag}) = {ln_z:.4f}  "
          f"({ln_z_err[0]:+.4f} / {ln_z_err[1]:+.4f})")
    print("  [ln Z = -ln_evidence_inv; errors are the ln(1/Z) errors "
          "swapped and negated]")

    result = {
        "tag": tag,
        "model": "baseline" if meta["baseline"] else "selection",
        "grid_file": meta["name"],
        "param_order": param_order,
        "priors": {p: list(PRIORS[p]) for p in param_order},
        "V_full": float(v_full),
        "ln_prior_const": float(-np.log(v_full)),
        "filler": float(args.filler),
        "ln_z": float(ln_z),
        "ln_z_err": [float(ln_z_err[0]), float(ln_z_err[1])],
    }

    out_json = Path(args.samples_dir) / f"evidence_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] evidence summary -> {out_json}")


if __name__ == "__main__":
    main()
