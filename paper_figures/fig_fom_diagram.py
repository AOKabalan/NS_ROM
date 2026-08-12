import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def filter_valid_branches(df):
    """
    Removes false asymmetric legs at large amplitudes (|a| >= 0.8) 
    where only the symmetric branch physically exists.
    """
    is_asym = df['branch'].str.contains('asym')
    is_large_amp = df['amp'].abs() >= 0.8
    
    # Keep rows that are NOT (asymmetric AND large amplitude)
    return df[~(is_asym & is_large_amp)].copy()

def main():
    parser = argparse.ArgumentParser(description="Generate fig:fom_diagram")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for the figure")
    parser.add_argument("--width", type=float, default=6.0, help="Figure width in inches")
    parser.add_argument("--usetex", action="store_true", help="Enable LaTeX text rendering")
    args = parser.parse_args()

    if args.usetex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })

    # 1. Load the data
    data_path = Path("states/E1_K4_tensor/sweep_results.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}. Run from repo root.")
    
    df = pd.read_csv(data_path)
    
    # 2. Clean and filter for the a=0 slice
    df = filter_valid_branches(df)
    df_a0 = df[df['amp'] == 0.0].copy()

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.6))
    
    # Group by branch to plot symmetric and asymmetric legs separately
    for branch_name, branch_data in df_a0.groupby('branch'):
        branch_data = branch_data.sort_values('Re')
        
        # Style based on branch type
        if 'sym' in branch_name:
            label = "Symmetric" if "Symmetric" not in ax.get_legend_handles_labels()[1] else "_nolegend_"
            ax.plot(branch_data['Re'], branch_data['C_L_fom'], 'k-', lw=1.5, label=label)
        else:
            label = "Asymmetric" if "Asymmetric" not in ax.get_legend_handles_labels()[1] else "_nolegend_"
            ax.plot(branch_data['Re'], branch_data['C_L_fom'], 'b--', lw=1.5, label=label)

    ax.set_xlabel(r"Reynolds number ($\mathrm{Re}$)")
    ax.set_ylabel(r"Full-order lift coefficient ($C_L$)")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Note: The inset with velocity fields from E1_K4_tensor/points/ 
    # will require loading the HDF5/NPZ fields and using tricontourf.
    
    plt.tight_layout()
    out_path = Path(args.outdir) / "fig_fom_diagram.pdf"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()