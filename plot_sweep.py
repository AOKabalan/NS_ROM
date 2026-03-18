"""Visualize sweep results: clusters, convergence, iterations."""

import numpy as np
import matplotlib.pyplot as plt
import csv


def load_sweep_csv(path):
    with open(path) as f:
        reader = list(csv.DictReader(f))
    records = []
    for r in reader:
        records.append({
            'Re': float(r['Re']),
            'amp': float(r['amp']),
            'cluster': int(r['cluster']),
            'converged': r['converged'] == 'True',
            'iterations': int(r['iterations']),
            't_rom': float(r['t_rom']),
            'cluster_switched': r['cluster_switched'] == 'True',
            'u_err': float(r['u_err']) if r.get('u_err') else None,
            'p_err': float(r['p_err']) if r.get('p_err') else None,
            'C_D': float(r['C_D']) if r.get('C_D') else None,
            'C_L': float(r['C_L']) if r.get('C_L') else None,
        })
    return records


def plot_sweep_clusters(records, figsize=(12, 5)):
    """2D scatter: (Re, amp) colored by cluster, X for failed points."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    Re = np.array([r['Re'] for r in records])
    amp = np.array([r['amp'] for r in records])
    cluster = np.array([r['cluster'] for r in records])
    converged = np.array([r['converged'] for r in records])
    iterations = np.array([r['iterations'] for r in records])

    # --- Left: cluster map with convergence ---
    ax = axes[0]
    conv_mask = converged
    fail_mask = ~converged

    sc = ax.scatter(Re[conv_mask], amp[conv_mask], c=cluster[conv_mask],
                    cmap='tab10', s=40, alpha=0.8, edgecolors='k', linewidth=0.3)
    ax.scatter(Re[fail_mask], amp[fail_mask], c='red',
               marker='x', s=80, linewidths=2, zorder=5, label='Failed')

    ax.set_xlabel('Re')
    ax.set_ylabel('amp')
    ax.set_title('Cluster Assignment')
    ax.legend()
    plt.colorbar(sc, ax=ax, label='Cluster')

    # --- Right: iterations heatmap ---
    ax = axes[1]
    sc = ax.scatter(Re[conv_mask], amp[conv_mask], c=iterations[conv_mask],
                    cmap='viridis', s=40, alpha=0.8, edgecolors='k', linewidth=0.3)
    ax.scatter(Re[fail_mask], amp[fail_mask], c='red',
               marker='x', s=80, linewidths=2, zorder=5, label='Failed')

    ax.set_xlabel('Re')
    ax.set_ylabel('amp')
    ax.set_title('Newton Iterations')
    ax.legend()
    plt.colorbar(sc, ax=ax, label='Iterations')

    plt.tight_layout()
    plt.savefig('sweep_overview.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_sweep_errors(records, figsize=(12, 5)):
    """2D scatter of velocity and pressure errors (FOM-compared points only)."""
    fom_records = [r for r in records if r['u_err'] is not None]
    if not fom_records:
        print("No FOM comparison points found.")
        return

    Re = np.array([r['Re'] for r in fom_records])
    amp = np.array([r['amp'] for r in fom_records])
    u_err = np.array([r['u_err'] for r in fom_records]) * 100
    p_err = np.array([r['p_err'] for r in fom_records]) * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    sc = ax.scatter(Re, amp, c=u_err, cmap='hot_r', s=50, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Re')
    ax.set_ylabel('amp')
    ax.set_title('Velocity Error (%)')
    plt.colorbar(sc, ax=ax, label='H1 error %')

    ax = axes[1]
    sc = ax.scatter(Re, amp, c=p_err, cmap='hot_r', s=50, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Re')
    ax.set_ylabel('amp')
    ax.set_title('Pressure Error (%)')
    plt.colorbar(sc, ax=ax, label='L2 error %')

    plt.tight_layout()
    plt.savefig('sweep_errors.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_sweep_3d(records, figsize=(14, 10)):
    """3D scatter: (Re, amp, iterations) colored by cluster, failed marked."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    Re = np.array([r['Re'] for r in records])
    amp = np.array([r['amp'] for r in records])
    cluster = np.array([r['cluster'] for r in records])
    converged = np.array([r['converged'] for r in records])
    iterations = np.array([r['iterations'] for r in records])

    conv_mask = converged
    fail_mask = ~converged

    sc = ax.scatter(Re[conv_mask], amp[conv_mask], iterations[conv_mask],
                    c=cluster[conv_mask], cmap='tab10',
                    s=40, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.scatter(Re[fail_mask], amp[fail_mask], iterations[fail_mask],
               c='red', marker='x', s=100, linewidths=2, zorder=5, label='Failed')

    ax.set_xlabel('Re', labelpad=10)
    ax.set_ylabel('amp', labelpad=10)
    ax.set_zlabel('Iterations', labelpad=10)
    ax.set_title('Sweep: Iterations vs (Re, amp)\nColored by Cluster')
    ax.legend()
    plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7, label='Cluster')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('sweep_3d.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_forces_3d(records, figsize=(14, 10)):
    """3D scatter: (Re, amp, C_L) and (Re, amp, C_D) colored by cluster."""
    force_records = [r for r in records if r['C_L'] is not None]
    if not force_records:
        print("No force data found.")
        return

    Re = np.array([r['Re'] for r in force_records])
    amp = np.array([r['amp'] for r in force_records])
    cluster = np.array([r['cluster'] for r in force_records])
    C_L = np.array([r['C_L'] for r in force_records])
    C_D = np.array([r['C_D'] for r in force_records])

    fig = plt.figure(figsize=(16, 7))

    ax = fig.add_subplot(121, projection='3d')
    sc = ax.scatter(Re, amp, C_L, c=cluster, cmap='tab10',
                    s=40, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Re', labelpad=10)
    ax.set_ylabel('amp', labelpad=10)
    ax.set_zlabel('$C_L$', labelpad=10)
    ax.set_title('Lift Coefficient\nColored by Cluster')
    plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7, label='Cluster')
    ax.view_init(elev=20, azim=45)

    ax = fig.add_subplot(122, projection='3d')
    sc = ax.scatter(Re, amp, C_D, c=cluster, cmap='tab10',
                    s=40, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Re', labelpad=10)
    ax.set_ylabel('amp', labelpad=10)
    ax.set_zlabel('$C_D$', labelpad=10)
    ax.set_title('Drag Coefficient\nColored by Cluster')
    plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7, label='Cluster')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('sweep_forces_3d.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_forces_2d(records, figsize=(12, 5)):
    """2D scatter: (Re, amp) colored by C_L and C_D, X for failed."""
    force_records = [r for r in records if r['C_L'] is not None]
    fail_records = [r for r in records if not r['converged']]
    if not force_records:
        print("No force data found.")
        return

    Re = np.array([r['Re'] for r in force_records])
    amp = np.array([r['amp'] for r in force_records])
    C_L = np.array([r['C_L'] for r in force_records])
    C_D = np.array([r['C_D'] for r in force_records])
    Re_fail = np.array([r['Re'] for r in fail_records])
    amp_fail = np.array([r['amp'] for r in fail_records])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    sc = ax.scatter(Re, amp, c=C_L, cmap='coolwarm', s=50, edgecolors='k', linewidth=0.3)
    if len(Re_fail) > 0:
        ax.scatter(Re_fail, amp_fail, c='red', marker='x', s=80, linewidths=2, zorder=5, label='Failed')
    ax.set_xlabel('Re')
    ax.set_ylabel('amp')
    ax.set_title('Lift Coefficient $C_L$')
    ax.legend()
    plt.colorbar(sc, ax=ax, label='$C_L$')

    ax = axes[1]
    sc = ax.scatter(Re, amp, c=C_D, cmap='viridis', s=50, edgecolors='k', linewidth=0.3)
    if len(Re_fail) > 0:
        ax.scatter(Re_fail, amp_fail, c='red', marker='x', s=80, linewidths=2, zorder=5, label='Failed')
    ax.set_xlabel('Re')
    ax.set_ylabel('amp')
    ax.set_title('Drag Coefficient $C_D$')
    ax.legend()
    plt.colorbar(sc, ax=ax, label='$C_D$')

    plt.tight_layout()
    plt.savefig('sweep_forces_2d.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'local_rom/K3_H1/sweep_results.csv'
    records = load_sweep_csv(path)

    n_total = len(records)
    n_conv = sum(1 for r in records if r['converged'])
    print(f"Loaded {n_total} points, {n_conv} converged ({n_conv/n_total*100:.1f}%)")

    plot_sweep_clusters(records)
    plot_sweep_errors(records)
    plot_sweep_3d(records)
    plot_forces_3d(records)
    plot_forces_2d(records)
