"""
Snapshot collection for Reduced Order Modeling.

Provides two workflows:
- Simple: Single branch, parameter sweeps (collect_snapshots)
- Bifurcating: Multiple branches for fluidic pinball (collect_snapshots_with_branches)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from firedrake import Function, CheckpointFile
import os
import json
import csv

from ..navier_stokes import (
    NavierStokesProblem,
    NavierStokesSolution,
    setup_navier_stokes_problem,
    solve_with_continuation,
    load_solution,
    compute_forces
)

__all__ = [
    # Dataclasses
    'SnapshotCollection',
    'BranchedSnapshotCollection',
    # Collection functions
    'collect_snapshots',
    'collect_reynolds_snapshots',
    'collect_snapshots_with_branches',
    # I/O
    'save_snapshots',
    'load_snapshots',
    'save_snapshot_dofs',
    'load_snapshot_dofs',
    # Visualization
    'plot_bifurcation_diagram',
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SnapshotCollection:
    """Collection of solution snapshots for ROM construction (single branch)."""
    solutions: List[Function]
    parameters: np.ndarray              # (n_snapshots, n_params)
    problem: NavierStokesProblem
    lift_coefficients: np.ndarray       # (n_snapshots,)
    drag_coefficients: np.ndarray       # (n_snapshots,)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.solutions)
        assert self.parameters.shape[0] == n, \
            f"Parameter count {self.parameters.shape[0]} != solution count {n}"
        assert len(self.lift_coefficients) == n
        assert len(self.drag_coefficients) == n

    @property
    def n_snapshots(self) -> int:
        return len(self.solutions)


@dataclass
class BranchedSnapshotCollection:
    """Collection of solution snapshots with multiple branches (e.g., fluidic pinball)."""
    solutions: List[Function]
    parameters: np.ndarray              # (n_snapshots, n_params)
    branch_ids: np.ndarray              # (n_snapshots,)
    branch_names: Dict[int, str]
    problem: NavierStokesProblem
    lift_coefficients: np.ndarray       # (n_snapshots,)
    drag_coefficients: np.ndarray       # (n_snapshots,)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.solutions)
        assert self.parameters.shape[0] == n
        assert len(self.branch_ids) == n
        assert len(self.lift_coefficients) == n
        assert len(self.drag_coefficients) == n
        assert set(self.branch_ids).issubset(self.branch_names.keys())

    @property
    def n_snapshots(self) -> int:
        return len(self.solutions)


# Type alias for functions accepting either type
AnySnapshotCollection = Union[SnapshotCollection, BranchedSnapshotCollection]


def _is_branched(collection: AnySnapshotCollection) -> bool:
    """Check if collection has branch information."""
    return isinstance(collection, BranchedSnapshotCollection)


# =============================================================================
# SNAPSHOT COLLECTION - SIMPLE
# =============================================================================

def collect_snapshots(
    problem: NavierStokesProblem,
    reynolds_values: np.ndarray,
    amplitude_values: np.ndarray,
    initial_checkpoint: str,
    reference_length: float = 1.0,
    reference_velocity: float = 1.0
) -> SnapshotCollection:
    """
    Collect snapshots across parameter space (single branch).

    Strategy:
    - Outer loop: amplitude
    - Inner loop: Reynolds with continuation
    - Uses last solution of each amplitude sweep as IC for next

    Args:
        problem: Problem definition
        reynolds_values: Array of Reynolds numbers to sweep
        amplitude_values: Array of amplitudes to sweep
        initial_checkpoint: Path to initial condition (without .h5)
        reference_length: For force computation
        reference_velocity: For force computation

    Returns:
        SnapshotCollection with all snapshots
    """
    solutions = []
    parameters = []
    lift_coefficients = []
    drag_coefficients = []

    print("=" * 80)
    print("SNAPSHOT COLLECTION")
    print("=" * 80)
    print(f"Reynolds range: [{reynolds_values[0]}, {reynolds_values[-1]}]")
    print(f"Amplitude range: [{amplitude_values[0]}, {amplitude_values[-1]}]")
    print(f"Total parameter combinations: {len(reynolds_values) * len(amplitude_values)}")
    print("=" * 80)

    initial_solution = load_solution(initial_checkpoint, problem)

    for amp in amplitude_values:
        print(f"\n{'='*60}")
        print(f"AMPLITUDE = {amp:.4f}")
        print(f"{'='*60}")

        problem.amplitude.assign(amp)

        re_solutions, forces = solve_with_continuation(
            problem,
            parameter_name='reynolds',
            parameter_values=reynolds_values,
            initial_solution=initial_solution
        )

        for i, (sol, force) in enumerate(zip(re_solutions, forces)):
            re = reynolds_values[i]
            solutions.append(sol.solution)
            parameters.append(np.array([re, amp]))
            lift_coefficients.append(force.lift)
            drag_coefficients.append(force.drag)
            print(f"  Re={re:.1f}: C_D={force.drag:.6f}, C_L={force.lift:.6f}")

        # Use first solution as IC for next amplitude
        initial_solution = re_solutions[0]

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total snapshots collected: {len(solutions)}")

    return SnapshotCollection(
        solutions=solutions,
        parameters=np.array(parameters),
        problem=problem,
        lift_coefficients=np.array(lift_coefficients),
        drag_coefficients=np.array(drag_coefficients),
        metadata={
            'reynolds_values': reynolds_values.tolist(),
            'amplitude_values': amplitude_values.tolist(),
            'reference_length': reference_length,
            'reference_velocity': reference_velocity,
            'n_snapshots': len(solutions),
            'n_parameters': 2,
            'parameter_names': ['reynolds', 'amplitude']
        }
    )

def collect_reynolds_snapshots(
    problem: NavierStokesProblem,
    reynolds_values: np.ndarray,
    amplitude: float,
    initial_checkpoint: str,
    reference_length: float = 1.0,
    reference_velocity: float = 1.0
) -> SnapshotCollection:
    """
    Collect snapshots across Reynolds number range at fixed amplitude.

    Args:
        problem: Problem definition
        reynolds_values: Array of Reynolds numbers to sweep
        amplitude: Fixed amplitude value
        initial_checkpoint: Path to initial condition (without .h5)
        reference_length: For force computation
        reference_velocity: For force computation

    Returns:
        SnapshotCollection with all snapshots
    """
    solutions = []
    parameters = []
    lift_coefficients = []
    drag_coefficients = []

    print("=" * 80)
    print("SNAPSHOT COLLECTION")
    print("=" * 80)
    print(f"Reynolds range: [{reynolds_values[0]}, {reynolds_values[-1]}]")
    print(f"Amplitude: {amplitude}")
    print(f"Total parameter combinations: {len(reynolds_values)}")
    print("=" * 80)

    initial_solution = load_solution(initial_checkpoint, problem)

    problem.amplitude.assign(amplitude)

    re_solutions, forces = solve_with_continuation(
        problem,
        parameter_name='reynolds',
        parameter_values=reynolds_values,
        initial_solution=initial_solution
    )

    for i, (sol, force) in enumerate(zip(re_solutions, forces)):
        re = reynolds_values[i]
        solutions.append(sol.solution)
        parameters.append(np.array([re, amplitude]))
        lift_coefficients.append(force.lift)
        drag_coefficients.append(force.drag)
        print(f"  Re={re:.1f}: C_D={force.drag:.6f}, C_L={force.lift:.6f}")

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total snapshots collected: {len(solutions)}")

    return SnapshotCollection(
        solutions=solutions,
        parameters=np.array(parameters),
        problem=problem,
        lift_coefficients=np.array(lift_coefficients),
        drag_coefficients=np.array(drag_coefficients),
        metadata={
            'reynolds_values': reynolds_values.tolist(),
            'amplitude': amplitude,
            'reference_length': reference_length,
            'reference_velocity': reference_velocity,
            'n_snapshots': len(solutions),
            'n_parameters': 2,
            'parameter_names': ['reynolds', 'amplitude']
        }
    )
# =============================================================================
# SNAPSHOT COLLECTION - BIFURCATING (FLUIDIC PINBALL)
# =============================================================================

def collect_snapshots_with_branches(
    problem: NavierStokesProblem,
    reynolds_values: np.ndarray,
    amplitude_values: np.ndarray,
    branch_configs: List[dict],
    reference_length: float = 1.0,
    reference_velocity: float = 1.0
) -> BranchedSnapshotCollection:
    """
    Collect snapshots for fluidic pinball with bifurcation handling.

    Requires exactly 3 branches: symmetric, asymmetric_up, asymmetric_down.

    Strategy:
    - Outer loop: amplitude (restart with IC each time)
    - Inner loop: Reynolds (continuation between Re values)
    - At each (Re, amp): solve all branches
    - Pre-bifurcation (lift_up * lift_down > 0): keep only symmetric
    - Post-bifurcation (lift_up * lift_down < 0): keep all 3

    Args:
        problem: Problem definition
        reynolds_values: Array of Reynolds numbers to sweep
        amplitude_values: Array of amplitudes to sweep
        branch_configs: List of 3 dicts with keys:
            - 'id': int (0=symmetric, 1=up, 2=down)
            - 'name': str (branch name)
            - 'ic_file': str (path to checkpoint, without .h5)
        reference_length: For force computation
        reference_velocity: For force computation

    Returns:
        BranchedSnapshotCollection with all snapshots
    """
    assert len(branch_configs) == 3, \
        "Bifurcating collection requires exactly 3 branches (symmetric, up, down)"

    solutions = []
    parameters = []
    branch_ids = []
    lift_coefficients = []
    drag_coefficients = []
    branch_names = {cfg['id']: cfg['name'] for cfg in branch_configs}

    print("=" * 80)
    print("SNAPSHOT COLLECTION (BIFURCATING)")
    print("=" * 80)
    print(f"Reynolds range: [{reynolds_values[0]}, {reynolds_values[-1]}]")
    print(f"Amplitude range: [{amplitude_values[0]}, {amplitude_values[-1]}]")
    print(f"Branches: {branch_names}")
    print(f"Total parameter combinations: {len(reynolds_values) * len(amplitude_values)}")
    print("=" * 80)

    for amp in amplitude_values:
        print(f"\n{'='*80}")
        print(f"AMPLITUDE = {amp:.4f}")
        print(f"{'='*80}")

        problem.amplitude.assign(amp)

        branch_solutions_at_amp = []
        forces_at_amp = []

        for branch_cfg in branch_configs:
            branch_id = branch_cfg['id']
            branch_name = branch_cfg['name']
            ic_file = branch_cfg['ic_file']

            print(f"\n--- Branch: {branch_name} (id={branch_id}) ---")

            initial_solution = load_solution(ic_file, problem)

            branch_solutions, forces = solve_with_continuation(
                problem,
                parameter_name='reynolds',
                parameter_values=reynolds_values,
                initial_solution=initial_solution
            )

            branch_solutions_at_amp.append(branch_solutions)
            forces_at_amp.append(forces)

        # Decide which snapshots to keep at each Reynolds
        for i, re in enumerate(reynolds_values):
            sol_symm = branch_solutions_at_amp[0][i]
            sol_up = branch_solutions_at_amp[1][i]
            sol_down = branch_solutions_at_amp[2][i]

            forces_symm = forces_at_amp[0][i]
            forces_up = forces_at_amp[1][i]
            forces_down = forces_at_amp[2][i]

            param_point = np.array([re, amp])

            # Bifurcation detection: opposite signs of lift
            is_bifurcated = forces_up.lift * forces_down.lift < 0.0

            if is_bifurcated:
                solutions.extend([sol_symm.solution, sol_up.solution, sol_down.solution])
                parameters.extend([param_point, param_point, param_point])
                branch_ids.extend([0, 1, 2])
                lift_coefficients.extend([forces_symm.lift, forces_up.lift, forces_down.lift])
                drag_coefficients.extend([forces_symm.drag, forces_up.drag, forces_down.drag])
                print(f"  Re={re:.1f}: BIFURCATED - "
                      f"C_L: symm={forces_symm.lift:.4f}, up={forces_up.lift:.4f}, down={forces_down.lift:.4f}")
            else:
                solutions.append(sol_symm.solution)
                parameters.append(param_point)
                branch_ids.append(0)
                lift_coefficients.append(forces_symm.lift)
                drag_coefficients.append(forces_symm.drag)
                print(f"  Re={re:.1f}: Pre-bifurcation - C_L={forces_symm.lift:.4f}")

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total snapshots collected: {len(solutions)}")
    for bid, bname in branch_names.items():
        count = np.sum(np.array(branch_ids) == bid)
        print(f"  {bname}: {count}")

    return BranchedSnapshotCollection(
        solutions=solutions,
        parameters=np.array(parameters),
        branch_ids=np.array(branch_ids),
        branch_names=branch_names,
        problem=problem,
        lift_coefficients=np.array(lift_coefficients),
        drag_coefficients=np.array(drag_coefficients),
        metadata={
            'reynolds_values': reynolds_values.tolist(),
            'amplitude_values': amplitude_values.tolist(),
            'reference_length': reference_length,
            'reference_velocity': reference_velocity,
            'n_snapshots': len(solutions),
            'n_parameters': 2,
            'parameter_names': ['reynolds', 'amplitude']
        }
    )


# =============================================================================
# I/O - SAVE
# =============================================================================

def save_snapshots(
    collection: AnySnapshotCollection,
    output_dir: str = "snapshots"
):
    """
    Save snapshot collection to disk.

    Structure:
    output_dir/
    ├── metadata.npz
    ├── metadata.json
    ├── metadata.csv
    ├── snapshot_0000.h5
    └── ...

    Args:
        collection: SnapshotCollection or BranchedSnapshotCollection
        output_dir: Directory to save snapshots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build arrays dict
    arrays = {
        'parameters': collection.parameters,
        'lift_coefficients': collection.lift_coefficients,
        'drag_coefficients': collection.drag_coefficients,
    }

    if _is_branched(collection):
        arrays['branch_ids'] = collection.branch_ids
        arrays['branch_names_keys'] = np.array(list(collection.branch_names.keys()))
        arrays['branch_names_values'] = np.array(list(collection.branch_names.values()))

    np.savez(os.path.join(output_dir, "metadata.npz"), **arrays)

    # Save JSON metadata
    json_metadata = {
        'n_snapshots': collection.n_snapshots,
        'is_branched': _is_branched(collection),
    }
    if _is_branched(collection):
        json_metadata['branch_names'] = {str(k): v for k, v in collection.branch_names.items()}

    for key, value in collection.metadata.items():
        if isinstance(value, np.ndarray):
            json_metadata[key] = value.tolist()
        else:
            json_metadata[key] = value

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(json_metadata, f, indent=2)

    # Save CSV (fixed: proper columns)
    csv_path = os.path.join(output_dir, "metadata.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['reynolds', 'amplitude', 'lift_coefficient', 'drag_coefficient']
        if _is_branched(collection):
            header.append('branch_id')
        writer.writerow(header)

        for i in range(collection.n_snapshots):
            row = [
                collection.parameters[i, 0],
                collection.parameters[i, 1],
                collection.lift_coefficients[i],
                collection.drag_coefficients[i]
            ]
            if _is_branched(collection):
                row.append(collection.branch_ids[i])
            writer.writerow(row)

    # Save solution files
    print(f"Saving {collection.n_snapshots} snapshots to {output_dir}/")
    mesh = collection.problem.mesh

    for i, sol in enumerate(collection.solutions):
        filepath = os.path.join(output_dir, f"snapshot_{i:04d}.h5")
        u, p = sol.subfunctions

        with CheckpointFile(filepath, 'w') as afile:
            afile.save_mesh(mesh)
            afile.save_function(u, name="velocity")
            afile.save_function(p, name="pressure")

        if (i + 1) % 10 == 0 or i == collection.n_snapshots - 1:
            print(f"  Saved {i + 1}/{collection.n_snapshots}")

    print(f"Snapshot collection saved to {output_dir}/")


def save_snapshot_dofs(
    collection: AnySnapshotCollection,
    output_dir: str = "snapshots_dofs"
):
    """
    Save snapshot DOFs as numpy arrays (no Firedrake dependency to load).

    Args:
        collection: SnapshotCollection or BranchedSnapshotCollection
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    u_dofs_list = []
    p_dofs_list = []
    for sol in collection.solutions:
        u, p = sol.subfunctions
        u_dofs_list.append(u.dat.data_ro.flatten().copy())
        p_dofs_list.append(p.dat.data_ro.flatten().copy())

    velocity_matrix = np.vstack(u_dofs_list)
    pressure_matrix = np.vstack(p_dofs_list)

    save_dict = {
        'velocity': velocity_matrix,
        'pressure': pressure_matrix,
        'parameters': collection.parameters,
        'lift_coefficients': collection.lift_coefficients,
        'drag_coefficients': collection.drag_coefficients,
    }

    if _is_branched(collection):
        save_dict['branch_ids'] = collection.branch_ids

    np.savez_compressed(os.path.join(output_dir, "snapshot_dofs.npz"), **save_dict)

    print(f"Saved {collection.n_snapshots} snapshots to {output_dir}/snapshot_dofs.npz")
    print(f"  Velocity shape: {velocity_matrix.shape}")
    print(f"  Pressure shape: {pressure_matrix.shape}")


# =============================================================================
# I/O - LOAD
# =============================================================================

def load_snapshots(
    input_dir: str,
    problem: NavierStokesProblem
) -> AnySnapshotCollection:
    """
    Load snapshot collection from disk.

    Automatically detects if collection is branched or simple.

    Args:
        input_dir: Directory containing snapshots
        problem: Problem definition (for function spaces)

    Returns:
        SnapshotCollection or BranchedSnapshotCollection
    """
    print(f"Loading snapshots from {input_dir}/")

    data = np.load(os.path.join(input_dir, "metadata.npz"), allow_pickle=True)

    parameters = data['parameters']
    lift_coefficients = data['lift_coefficients']
    drag_coefficients = data['drag_coefficients']

    is_branched = 'branch_ids' in data

    if is_branched:
        branch_ids = data['branch_ids']
        branch_names = dict(zip(
            data['branch_names_keys'].astype(int),
            data['branch_names_values']
        ))

    with open(os.path.join(input_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    n_snapshots = len(parameters)
    solutions = []

    print(f"Loading {n_snapshots} snapshots...")

    for i in range(n_snapshots):
        filepath = os.path.join(input_dir, f"snapshot_{i:04d}.h5")
        sol = Function(problem.mixed_space, name="solution")

        with CheckpointFile(filepath, 'r') as afile:
            mesh_in = afile.load_mesh()
            u_loaded = afile.load_function(mesh_in, name="velocity")
            p_loaded = afile.load_function(mesh_in, name="pressure")

        sol.sub(0).dat.data[:] = u_loaded.dat.data[:]
        sol.sub(1).dat.data[:] = p_loaded.dat.data[:]
        solutions.append(sol)

        if (i + 1) % 10 == 0 or i == n_snapshots - 1:
            print(f"  Loaded {i + 1}/{n_snapshots}")

    print(f"Loaded {n_snapshots} snapshots")

    if is_branched:
        return BranchedSnapshotCollection(
            solutions=solutions,
            parameters=parameters,
            branch_ids=branch_ids,
            branch_names=branch_names,
            problem=problem,
            lift_coefficients=lift_coefficients,
            drag_coefficients=drag_coefficients,
            metadata=metadata
        )
    else:
        return SnapshotCollection(
            solutions=solutions,
            parameters=parameters,
            problem=problem,
            lift_coefficients=lift_coefficients,
            drag_coefficients=drag_coefficients,
            metadata=metadata
        )


def load_snapshot_dofs(input_dir: str) -> dict:
    """
    Load snapshot DOFs (no Firedrake dependency).

    Args:
        input_dir: Directory containing snapshot_dofs.npz

    Returns:
        Dict with 'velocity', 'pressure', 'parameters', 'lift_coefficients',
        'drag_coefficients', and optionally 'branch_ids', 'metadata'
    """
    data = np.load(os.path.join(input_dir, "snapshot_dofs.npz"))

    result = {
        'velocity': data['velocity'],
        'pressure': data['pressure'],
        'parameters': data['parameters'],
        'lift_coefficients': data['lift_coefficients'],
        'drag_coefficients': data['drag_coefficients'],
    }

    if 'branch_ids' in data:
        result['branch_ids'] = data['branch_ids']

    metadata_path = os.path.join(input_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            result['metadata'] = json.load(f)
    else:
        result['metadata'] = {}

    print(f"Loaded from {input_dir}/snapshot_dofs.npz")
    print(f"  Velocity shape: {result['velocity'].shape}")
    print(f"  Pressure shape: {result['pressure'].shape}")

    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_bifurcation_diagram(
    collection: BranchedSnapshotCollection,
    save_file: str = None
):
    """
    Plot bifurcation diagram: lift vs Reynolds for each amplitude.

    Args:
        collection: BranchedSnapshotCollection
        save_file: Optional filename to save plot
    """
    reynolds = collection.parameters[:, 0]
    amplitudes = collection.parameters[:, 1]
    lifts = collection.lift_coefficients

    unique_amps = np.unique(amplitudes)
    n_amps = len(unique_amps)

    fig, axes = plt.subplots(1, n_amps, figsize=(6*n_amps, 5), squeeze=False)
    axes = axes.flatten()

    colors = {0: 'blue', 1: 'red', 2: 'green'}
    markers = {0: 'o', 1: '^', 2: 'v'}

    for idx, amp in enumerate(unique_amps):
        ax = axes[idx]
        mask = amplitudes == amp

        for branch_id, branch_name in collection.branch_names.items():
            branch_mask = mask & (collection.branch_ids == branch_id)
            if np.any(branch_mask):
                ax.plot(reynolds[branch_mask], lifts[branch_mask],
                       color=colors[branch_id], marker=markers[branch_id],
                       linestyle='-', linewidth=2, markersize=6, label=branch_name)

        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Reynolds Number', fontsize=12)
        ax.set_ylabel('Lift Coefficient', fontsize=12)
        ax.set_title(f'Amplitude = {amp:.2f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Bifurcation diagram saved to {save_file}")

    plt.show()