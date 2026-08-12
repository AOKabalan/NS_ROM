import os
import sys

import numpy as np

from nsrom.navier_stokes import (
    setup_navier_stokes_problem,
    load_solution,
    solve_with_continuation,
)
from nsrom.snapshot_collection import (
    SnapshotCollection,
    BranchedSnapshotCollection,
    save_snapshots,
    save_snapshot_dofs,
)

MESH_FILE = "mesh/mid_pinball.msh"
OUTPUT_DIR = "snapshots_graded"
DOFS_DIR = "snapshots_graded_dofs"

MODE = "2p"
BRANCH_MODE = "all"
SINGLE_BRANCH = "upper"

IC_SYMMETRIC = "states/velocity_checkpoint_symm"
IC_UPPER = "states/velocity_checkpoint_up"
IC_LOWER = "states/velocity_checkpoint_down"

LIFT_TOL = 1e-6

BRANCHES = {
    "symmetric": (0, IC_SYMMETRIC),
    "upper": (1, IC_UPPER),
    "lower": (2, IC_LOWER),
}


def sweep(high, low, step=None, n=None):
    if step is not None:
        return np.arange(high, low - 0.5 * abs(step), -abs(step))
    return np.linspace(high, low, n)


RE_HIGH, RE_LOW = 100.0, 40.0
RE_STEP, RE_N = None, 40

AMP_HIGH, AMP_LOW = 1.0, -1.0
AMP_STEP, AMP_N = None, 19
AMP_FIXED = 0.0

RE_VALUES = sweep(RE_HIGH, RE_LOW, RE_STEP, RE_N)
AMP_VALUES = sweep(AMP_HIGH, AMP_LOW, AMP_STEP, AMP_N)


def re_values_for(amp):
    return RE_VALUES


GRID_FIGURE = "figures/training_grid.pdf"


def plot_parameter_grid(amp_values, outfile=GRID_FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))

    total = 0
    for a in amp_values:
        a = float(a)
        r = np.asarray(re_values_for(a), float)
        total += r.size
        ax.scatter(r, np.full_like(r, a), s=7, marker="o", linewidths=0,
                   color="#1f4e79")

    ax.axhline(0.0, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel("amplitude")
    ax.set_title(f"{total} parameter points, {len(amp_values)} amplitudes",
                 fontsize=8)
    ax.tick_params(labelsize=8, direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(outfile)[0] + ".png", dpi=300,
                bbox_inches="tight")
    print(f"wrote {outfile}")

    print(f"{'amp':>9}  {'n_Re':>5}  {'Re range':>16}  {'min dRe':>8}  {'max dRe':>8}")
    for a in amp_values:
        a = float(a)
        r = np.asarray(re_values_for(a), float)
        d = np.abs(np.diff(r))
        print(f"{a:9.4f}  {r.size:5d}  "
              f"[{r[-1]:6.2f}, {r[0]:6.2f}]  {d.min():8.3f}  {d.max():8.3f}")
    print(f"{'total':>9}  {total:5d}")


def branch_configs():
    if BRANCH_MODE == "all":
        return sorted((bid, name, ic) for name, (bid, ic) in BRANCHES.items())
    bid, ic = BRANCHES[SINGLE_BRANCH]
    return [(0, SINGLE_BRANCH, ic)]


def collect(problem, amp_values):
    configs = branch_configs()
    names = {bid: name for bid, name, _ in configs}
    solutions, params, bids, lifts, drags = [], [], [], [], []
    critical, grids = {}, {}

    for amp in amp_values:
        amp = float(amp)
        re_values = np.asarray(re_values_for(amp), float)
        if not np.all(np.diff(re_values) < 0.0):
            raise ValueError(f"re_values_for({amp}) must be strictly descending")
        grids[amp] = re_values.tolist()

        print(f"\n=== amplitude = {amp:.4f}  ({re_values.size} Re values, "
              f"{re_values[0]:.2f} -> {re_values[-1]:.2f}) ===")
        problem.amplitude.assign(amp)

        runs = {}
        for bid, name, ic in configs:
            print(f"--- branch {name} ---")
            sols, forces = solve_with_continuation(
                problem,
                parameter_name="reynolds",
                parameter_values=re_values,
                initial_solution=load_solution(ic, problem),
            )
            runs[bid] = (sols, forces)

        if len(configs) == 1:
            bid = configs[0][0]
            sols, forces = runs[bid]
            for i, re in enumerate(re_values):
                solutions.append(sols[i].solution)
                params.append([re, amp])
                bids.append(bid)
                lifts.append(forces[i].lift)
                drags.append(forces[i].drag)
            continue

        f_sym, f_up, f_dn = runs[0][1], runs[1][1], runs[2][1]
        gaps = np.array([abs(a.lift - b.lift) for a, b in zip(f_up, f_dn)])
        scale = max(float(np.max([abs(f.drag) for f in f_sym])), 1.0)
        below = np.flatnonzero(gaps <= LIFT_TOL * scale)
        i_end = int(below[0]) if below.size else re_values.size
        critical[amp] = float(re_values[i_end - 1]) if i_end > 0 else None
        print(f"bifurcated for the first {i_end}/{re_values.size} Reynolds values")

        for i, re in enumerate(re_values):
            sols, forces = runs[0]
            solutions.append(sols[i].solution)
            params.append([re, amp])
            bids.append(0)
            lifts.append(forces[i].lift)
            drags.append(forces[i].drag)

            if i < i_end:
                order = sorted([1, 2], key=lambda k: -runs[k][1][i].lift)
                for tgt, src in enumerate(order, start=1):
                    sols, forces = runs[src]
                    solutions.append(sols[i].solution)
                    params.append([re, amp])
                    bids.append(tgt)
                    lifts.append(forces[i].lift)
                    drags.append(forces[i].drag)

    metadata = {
        "amplitude_values": np.asarray(amp_values, float).tolist(),
        "reynolds_values_by_amplitude": {str(k): v for k, v in grids.items()},
        "reynolds_values": sorted({r for g in grids.values() for r in g}, reverse=True),
        "tensor_grid": len({tuple(g) for g in grids.values()}) == 1,
        "mode": MODE,
        "branch_mode": BRANCH_MODE,
        "n_snapshots": len(solutions),
        "n_parameters": 2,
        "parameter_names": ["reynolds", "amplitude"],
        "critical_reynolds": {str(k): v for k, v in critical.items()},
        "initial_conditions": {name: ic for _, name, ic in configs},
    }

    args = dict(
        solutions=solutions,
        parameters=np.array(params),
        problem=problem,
        lift_coefficients=np.array(lifts),
        drag_coefficients=np.array(drags),
        metadata=metadata,
    )

    if len(configs) == 1:
        return SnapshotCollection(**args)
    return BranchedSnapshotCollection(branch_ids=np.array(bids), branch_names=names, **args)


def main():
    preview_only = "--preview" in sys.argv

    amp_values = AMP_VALUES if MODE == "2p" else np.array([AMP_FIXED])
    re0 = np.asarray(re_values_for(float(amp_values[0])), float)

    total = sum(np.asarray(re_values_for(float(a))).size for a in amp_values)
    print(f"mode={MODE}  branches={BRANCH_MODE}")
    print(f"amp: {amp_values[0]:.3f} -> {amp_values[-1]:.3f} "
          f"({amp_values.size} values)")
    print(f"parameter points: {total} (x branches where bifurcated)\n")

    plot_parameter_grid(amp_values)

    if preview_only:
        print("\npreview only, no solves performed")
        return

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=float(re0[0]),
        amplitude_init=float(amp_values[0]),
    )

    collection = collect(problem, amp_values)
    save_snapshots(collection, OUTPUT_DIR)
    save_snapshot_dofs(collection, DOFS_DIR)
    print(f"\n{collection.n_snapshots} snapshots -> {OUTPUT_DIR}/, {DOFS_DIR}/")


if __name__ == "__main__":
    main()
