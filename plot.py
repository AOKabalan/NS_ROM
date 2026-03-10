import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_lift_3d(parameters, lift_coefficients, labels, figsize=(14, 10), cmap='tab10'):
    """
    Plot lift coefficients in 3D scatter plot.

    Args:
        parameters: (N, 2) array with [Re, amp] pairs
        lift_coefficients: (N,) array of lift values
        labels: (N,) array of cluster labels for color coding
        figsize: tuple for figure size (default: (14, 10))
        cmap: colormap name (default: 'tab10' works well for discrete clusters)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Extract Re and amp from parameters
    Re = parameters[:, 0]
    amp = parameters[:, 1]

    # Create scatter plot with color coding by labels
    scatter = ax.scatter(Re, amp, lift_coefficients,
                        c=labels, cmap=cmap,
                        s=60, alpha=0.7, edgecolors='k', linewidth=0.3)

    # Labels and title
    ax.set_xlabel('Reynolds Number (Re)', fontsize=12, labelpad=10)
    ax.set_ylabel('Amplitude (amp)', fontsize=12, labelpad=10)
    ax.set_zlabel('Lift Coefficient', fontsize=12, labelpad=10)
    ax.set_title('Lift Coefficients vs Re and Amplitude\n(Color-coded by Cluster)',
                 fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Cluster Label', fontsize=11)

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


def plot_lift_3d_interactive(parameters, lift_coefficients, labels, figsize=(14, 10)):
    """
    Enhanced version with rotation capability.
    Use mouse to rotate the plot.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    Re = parameters[:, 0]
    amp = parameters[:, 1]

    # Use different colors for each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label_idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(Re[mask], amp[mask], lift_coefficients[mask],
                  label=f'Cluster {label}',
                  s=60, alpha=0.7, edgecolors='k', linewidth=0.3,
                  color=colors[label_idx])

    ax.set_xlabel('Reynolds Number (Re)', fontsize=12, labelpad=10)
    ax.set_ylabel('Amplitude (amp)', fontsize=12, labelpad=10)
    ax.set_zlabel('Lift Coefficient', fontsize=12, labelpad=10)
    ax.set_title('Lift Coefficients vs Re and Amplitude', fontsize=14, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


# Usage example:
# plot_lift_3d(parameters, lift_coefficients, labels)
# or for better legend:
# plot_lift_3d_interactive(parameters, lift_coefficients, labels)