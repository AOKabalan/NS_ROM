# task_a_centered_vs_raw.py
import matplotlib.pyplot as plt
from setup_investigation import load_everything

d = load_everything()

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
for i in range(4):
    axes[0, i].scatter(range(d['A'].shape[1]), d['A'][i, :], c=d['branch_ids'], cmap='tab10', s=8)
    axes[0, i].set_title(f'Mode {i+1} (raw)')
    axes[1, i].scatter(range(d['A_c'].shape[1]), d['A_c'][i, :], c=d['branch_ids'], cmap='tab10', s=8)
    axes[1, i].set_title(f'Mode {i+1} (centered)')
plt.tight_layout()
plt.show()