import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add scripts directory to path so state_store can be imported
sys.path.append("scripts")
from state_store import StateStore


def filter_valid_branches(df):
    """Removes false asymmetric legs at large amplitudes (|a| >= 0.8)
    where only the symmetric branch physically exists.
    """
    is_asym = df["branch"].str.contains("asym")
    is_large_amp = df["amp"].abs() >= 0.8
    return df[~(is_asym & is_large_amp)].copy()


def load_pressure_bases(local_rom_dir, k_clusters=4):
    """Loads the pressure POD bases for each cluster k."""
    bases = {}
    for k in range(k_clusters):
        basis_path = local_rom_dir / f"cluster_{k}" / "pod_basis" / "pod_basis.npz"
        if not basis_path.exists():
            raise FileNotFoundError(f"Missing basis file: {basis_path}")

        with np.load(basis_path) as data:
            # Extract the correct key based on your .npz structure
            bases[k] = data["pressure_modes"]

    return bases

def compute_pressure_errors(store, valid_indices, local_rom_dir, k_clusters=4):
    """Computes relative pressure error on the fly using StateStore and POD bases."""
    print("Loading pressure POD bases...")
    bases = load_pressure_bases(local_rom_dir, k_clusters)

    print("Computing pressure errors from StateStore (with nullspace centering)...")
    err_p_map = {}

    for pt in tqdm(store):
        if pt.idx not in valid_indices:
            continue

        # Skip points without stored FOM fields or diverged points
        if not pt.get("has_fom_field", False) or pt.p_out is None:
            continue

        p_fom = pt.p_fom
        p_out = pt.p_out
        cluster_k = pt.cluster

        V_p = bases[cluster_k]

        # 1. Dimension alignment: If the saved basis has more modes than p_out,
        # we must truncate it to match the reduced dimension r_p.
        r_p = p_out.shape[0]
        if V_p.shape[1] > r_p:
            V_p = V_p[:, :r_p]

        # Reconstruct full-order ROM pressure: p_rom = V_p * p_out
        p_rom_full = V_p @ p_out

        # 2. Nullspace centering: Remove the arbitrary gauge constant
        # so we are comparing the physical pressure distributions.
        p_fom_centered = p_fom - np.mean(p_fom)
        p_rom_full_centered = p_rom_full - np.mean(p_rom_full)

        # Compute relative L2 error on the centered fields
        norm_fom = np.linalg.norm(p_fom_centered)
        if norm_fom > 0:
            err = np.linalg.norm(p_fom_centered - p_rom_full_centered) / norm_fom
            err_p_map[pt.idx] = err

    return err_p_map


def main():
    parser = argparse.ArgumentParser(description="Generate fig:branch_errors")
    parser.add_argument("--outdir", default=".", help="Output directory for the figure")
    parser.add_argument("--width", type=float, default=6.0, help="Figure width in inches")
    parser.add_argument("--usetex", action="store_true", help="Enable LaTeX text rendering")
    args = parser.parse_args()

    if args.usetex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    state_dir = Path("states/E1_K4_tensor")
    local_rom_dir = Path("local_rom/K4_H1_tol1e-08")
    csv_path = state_dir / "sweep_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}.")

    # 1. Load CSV and filter invalid branches
    df = pd.read_csv(csv_path)
    df = filter_valid_branches(df)

    # 2. Compute missing pressure errors via StateStore
    store = StateStore(str(state_dir))

    idx_col = "idx" if "idx" in df.columns else df.index

    err_p_map = compute_pressure_errors(
        store, set(idx_col), local_rom_dir, k_clusters=4
    )

    # Map computed pressure errors back to dataframe
    df["err_p"] = idx_col.map(err_p_map)

    # Drop rows where pressure error computation failed (e.g. no FOM field saved)
    df = df.dropna(subset=['err_p'])
    # 3. Setup Plot
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(args.width, args.width * 0.45), sharex=True)

    # Create a clear flag for branch type
    df["is_asym"] = df["branch"].str.contains("asym")

    # Group strictly by branch type (Symmetric vs Asymmetric) to get exactly two aggregate curves
    for is_asym, group_data in df.groupby("is_asym"):
        # Average across all branches and all amplitudes for each Reynolds number
        mean_data = group_data.groupby("Re")[["err_u", "err_p"]].mean().reset_index()
        mean_data = mean_data.sort_values("Re")

        color = "tab:red" if is_asym else "gray"
        linewidth = 1.5 if is_asym else 1.2
        label = "Asymmetric branches" if is_asym else "Symmetric branches"

        ax_u.plot(
            mean_data["Re"],
            mean_data["err_u"],
            color=color,
            lw=linewidth,
            label=label,
        )
        if "err_p" in mean_data:
            ax_p.plot(
                mean_data["Re"],
                mean_data["err_p"],
                color=color,
                lw=linewidth,
                label=label,
            )

    # Formatting
    for ax in (ax_u, ax_p):
        ax.set_xlim(20, 110)
        ax.set_xlabel(r"Reynolds number ($\mathrm{Re}$)")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.6)

    ax_u.set_title("Mean Velocity Relative Error")
    ax_u.set_ylabel(r"Mean $\|u_{\mathrm{fom}} - u_{\mathrm{rom}}\| / \|u_{\mathrm{fom}}\|$")

    ax_p.set_title("Mean Pressure Relative Error")
    ax_p.set_ylabel(r"Mean $\|p_{\mathrm{fom}} - p_{\mathrm{rom}}\| / \|p_{\mathrm{fom}}\|$")

    # Add legend to the pressure subplot (or velocity, either works)
    ax_p.legend(loc="best")

    plt.tight_layout()
    out_path = Path(args.outdir) / "fig_branch_errors.pdf"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()