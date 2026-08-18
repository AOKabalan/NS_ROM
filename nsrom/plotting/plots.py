import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np


def plot_lift_3d(
    parameters,
    lift_coefficients,
    labels,
    figsize=(14, 10),
    cmap='tab10',
    legend=True,
    elev=20,
    azim=45,
    title='Lift Coefficients vs Re and Amplitude',
):
    """
    3D scatter of lift coefficients over (Re, amp), colored by cluster.

    Parameters
    ----------
    parameters        : (N, 2) array of [Re, amp]
    lift_coefficients : (N,) array
    labels            : (N,) cluster labels (integer)
    legend            : if True, draw a per-cluster legend; if False, draw a
                        colorbar (better when there are many clusters)
    elev, azim        : initial 3D view angles
    title             : plot title
    """
    Re  = parameters[:, 0]
    amp = parameters[:, 1]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if legend:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                Re[mask], amp[mask], lift_coefficients[mask],
                label=f'Cluster {label}',
                color=color,
                s=60, alpha=0.7, edgecolors='k', linewidth=0.3,
            )
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    else:
        scatter = ax.scatter(
            Re, amp, lift_coefficients,
            c=labels, cmap=cmap,
            s=60, alpha=0.7, edgecolors='k', linewidth=0.3,
        )
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Cluster Label', fontsize=11)

    ax.set_xlabel('Reynolds Number (Re)', fontsize=12, labelpad=10)
    ax.set_ylabel('Amplitude (amp)',     fontsize=12, labelpad=10)
    ax.set_zlabel('Lift Coefficient',    fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(f'lift_in_{n_clusters}_clusters.png', dpi=150, bbox_inches='tight')

    print(f" lift_in_{n_clusters}_clusters.png")
    plt.show()

def plot_sweep_lift_3d(csv_path, figsize=(14, 10), cmap='tab10', elev=20, azim=45):
    import pandas as pd

    df = pd.read_csv(csv_path).dropna(subset=['C_L'])
    unique_clusters = sorted(df['cluster'].unique())
    n_clusters = len(unique_clusters)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_clusters))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for cluster, color in zip(unique_clusters, colors):
        mask = df['cluster'] == cluster
        ax.scatter(df.loc[mask, 'Re'], df.loc[mask, 'amp'], df.loc[mask, 'C_L'],
                   color=color, label=f'Cluster {cluster}',
                   s=60, alpha=0.8, edgecolors='k', linewidth=0.3)

    ax.set_xlabel('Re', fontsize=12, labelpad=10)
    ax.set_ylabel('Amplitude', fontsize=12, labelpad=10)
    ax.set_zlabel('C_L', fontsize=12, labelpad=10)
    ax.set_title(f'Lift surface colored by cluster (K={n_clusters})', fontsize=13, pad=20)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    plt.tight_layout()
    out = csv_path.replace('.csv', '_lift_clusters_3d.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved to {out}")
    plt.show()
