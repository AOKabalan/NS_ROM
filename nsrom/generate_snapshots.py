"""
Generates Snapshots for ROM and DEIM training.
Supports both single-parameter (Reynolds only) and two-parameter (Reynolds + Amplitude) modes.
"""

import numpy as np
import os

from .navier_stokes import (
    setup_navier_stokes_problem,
    solve_steady_navier_stokes,
    load_solution
)
from .snapshot_collection import *
from .lifting_functions import load_lifting_functions
from .helper_functions import dofs_to_functions

from rom import (
    compute_pod_basis,
    save_pod_basis,
    load_pod_basis,
    build_reduced_operators,
    save_reduced_operators,
    solve_rom,
    reconstruct_solution,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
MESH_FILE = "mesh/mid_pinball.msh"
INITIAL_CHECKPOINT = "newlygenerated_high"
LIFTING_DIR = "lifting"

# =============================================================================
# MODE SELECTION: Set to True for two-parameter mode, False for one-parameter
# =============================================================================
TWO_PARAMETER_MODE = True

# =============================================================================
# ONE-PARAMETER CONFIGURATION (Reynolds only, fixed amplitude)
# =============================================================================
POD_RE_VALUES_1P = np.linspace(100.0, 40.0, 40)
POD_AMPLITUDE_1P = 0.0
POD_OUTPUT_DIR_1P = "one_param_snapshots_pod"
POD_DOFS_DIR_1P = "one_param_snapshots_pod_dofs"

DEIM_RE_VALUES_1P = np.linspace(100.0, 40.0, 40)
DEIM_AMPLITUDE_1P = 0.0
DEIM_OUTPUT_DIR_1P = "one_param_snapshots_rom_deim"
DEIM_DOFS_DIR_1P = "one_param_snapshots_rom_deim_dofs"

# =============================================================================
# TWO-PARAMETER CONFIGURATION (Reynolds + Amplitude grid)
# =============================================================================
POD_RE_VALUES_2P = np.linspace(100.0, 40.0, 20)
POD_AMP_VALUES_2P = np.linspace(-1.0, 1.0, 19)
POD_OUTPUT_DIR_2P = "two_param_snapshots_pod"
POD_DOFS_DIR_2P = "two_param_snapshots_pod_dofs"

DEIM_RE_VALUES_2P = np.linspace(100.0, 40.0, 24)
DEIM_AMP_VALUES_2P = np.linspace(-1.0, 1.0, 23)
DEIM_OUTPUT_DIR_2P = "two_param_snapshots_rom_deim"
DEIM_DOFS_DIR_2P = "two_param_snapshots_rom_deim_dofs"

# =============================================================================
# ROM-BASED DEIM OPTIONS
# =============================================================================
USE_ROM_FOR_DEIM = False
ROM_POD_DIR_1P = "pod_basis_one_param"
ROM_POD_DIR_2P = "pod_basis_two_param"
# =============================================================================


# =============================================================================
# Validation Functions
# =============================================================================

def load_snapshot_metadata(dofs_dir: str) -> dict | None:
    """
    Load metadata from existing snapshot DOFs.

    Returns None if metadata cannot be loaded.
    """
    metadata_path = os.path.join(dofs_dir, "snapshot_dofs.npz")
    if not os.path.exists(metadata_path):
        return None

    try:
        data = np.load(metadata_path, allow_pickle=True)

        metadata = {
            'n_snapshots': data['velocity'].shape[0],
            'parameters': data['parameters'] if 'parameters' in data else None,
        }

        if metadata['parameters'] is not None:
            metadata['reynolds_values'] = metadata['parameters'][:, 0]
            metadata['amplitude_values'] = metadata['parameters'][:, 1]

        return metadata
    except Exception as e:
        print(f"  Warning: Could not load metadata from {dofs_dir}: {e}")
        return None


def needs_snapshot_regeneration_1p(
    dofs_dir: str,
    reynolds_values: np.ndarray,
    amplitude: float,
    tol: float = 1e-6,
) -> tuple[bool, str]:
    """
    Check if one-parameter snapshots need regeneration.
    """
    if not os.path.exists(dofs_dir):
        return True, "Directory does not exist"

    snapshot_file = os.path.join(dofs_dir, "snapshot_dofs.npz")
    if not os.path.exists(snapshot_file):
        return True, "Snapshot file not found"

    metadata = load_snapshot_metadata(dofs_dir)
    if metadata is None:
        return True, "Could not load metadata"

    n_requested = len(reynolds_values)
    n_existing = metadata['n_snapshots']
    if n_existing != n_requested:
        return True, f"Snapshot count mismatch: existing={n_existing}, requested={n_requested}"

    if metadata.get('reynolds_values') is not None:
        re_existing = metadata['reynolds_values']

        if len(re_existing) != len(reynolds_values):
            return True, f"Reynolds array length mismatch"

        if not np.allclose(re_existing, reynolds_values, rtol=tol, atol=tol):
            return True, f"Reynolds values changed: [{re_existing[0]:.1f}, {re_existing[-1]:.1f}] → [{reynolds_values[0]:.1f}, {reynolds_values[-1]:.1f}]"

    if metadata.get('amplitude_values') is not None:
        amp_existing = metadata['amplitude_values']

        if np.allclose(amp_existing, amplitude, rtol=tol, atol=tol):
            pass
        else:
            unique_amps = np.unique(amp_existing)
            if len(unique_amps) == 1:
                if not np.isclose(unique_amps[0], amplitude, rtol=tol, atol=tol):
                    return True, f"Amplitude changed: {unique_amps[0]:.4f} → {amplitude:.4f}"
            else:
                return True, f"Amplitude configuration changed (was two-param, now one-param)"

    return False, "Snapshots are up-to-date"


def needs_snapshot_regeneration_2p(
    dofs_dir: str,
    reynolds_values: np.ndarray,
    amplitude_values: np.ndarray,
    tol: float = 1e-6,
) -> tuple[bool, str]:
    """
    Check if two-parameter snapshots need regeneration.
    """
    if not os.path.exists(dofs_dir):
        return True, "Directory does not exist"

    snapshot_file = os.path.join(dofs_dir, "snapshot_dofs.npz")
    if not os.path.exists(snapshot_file):
        return True, "Snapshot file not found"

    metadata = load_snapshot_metadata(dofs_dir)
    if metadata is None:
        return True, "Could not load metadata"

    # For two-parameter: total snapshots = len(re) * len(amp)
    n_requested = len(reynolds_values) * len(amplitude_values)
    n_existing = metadata['n_snapshots']
    if n_existing != n_requested:
        return True, f"Snapshot count mismatch: existing={n_existing}, requested={n_requested}"

    if metadata.get('reynolds_values') is not None and metadata.get('amplitude_values') is not None:
        re_existing_unique = np.unique(metadata['reynolds_values'])
        amp_existing_unique = np.unique(metadata['amplitude_values'])

        if len(re_existing_unique) != len(reynolds_values):
            return True, f"Reynolds grid size changed: {len(re_existing_unique)} → {len(reynolds_values)}"
        if not np.allclose(np.sort(re_existing_unique), np.sort(reynolds_values), rtol=tol, atol=tol):
            return True, f"Reynolds values changed"

        if len(amp_existing_unique) != len(amplitude_values):
            return True, f"Amplitude grid size changed: {len(amp_existing_unique)} → {len(amplitude_values)}"
        if not np.allclose(np.sort(amp_existing_unique), np.sort(amplitude_values), rtol=tol, atol=tol):
            return True, f"Amplitude values changed"

    return False, "Snapshots are up-to-date"


# =============================================================================
# FOM Snapshot Collection
# =============================================================================

def collect_fom_snapshots_1p(
    problem,
    reynolds_values: np.ndarray,
    amplitude: float,
    initial_checkpoint: str,
) -> SnapshotCollection:
    """
    Collect FOM snapshots for one-parameter case (Reynolds sweep, fixed amplitude).
    """
    print("=" * 80)
    print("SNAPSHOT COLLECTION (ONE-PARAMETER: Reynolds)")
    print("=" * 80)
    print(f"Reynolds range: [{reynolds_values[0]}, {reynolds_values[-1]}]")
    print(f"Number of Reynolds values: {len(reynolds_values)}")
    print(f"Amplitude (fixed): {amplitude}")

    return collect_reynolds_snapshots(
        problem=problem,
        reynolds_values=reynolds_values,
        amplitude=amplitude,
        initial_checkpoint=initial_checkpoint,
    )


def collect_fom_snapshots_2p(
    problem,
    reynolds_values: np.ndarray,
    amplitude_values: np.ndarray,
    initial_checkpoint: str,
) -> SnapshotCollection:
    """
    Collect FOM snapshots for two-parameter case (Reynolds x Amplitude grid).
    """
    print("=" * 80)
    print("SNAPSHOT COLLECTION (TWO-PARAMETER: Reynolds x Amplitude)")
    print("=" * 80)
    print(f"Reynolds range: [{reynolds_values[0]}, {reynolds_values[-1]}]")
    print(f"Number of Reynolds values: {len(reynolds_values)}")
    print(f"Amplitude range: [{amplitude_values[0]}, {amplitude_values[-1]}]")
    print(f"Number of Amplitude values: {len(amplitude_values)}")
    print(f"Total snapshots: {len(reynolds_values) * len(amplitude_values)}")

    return collect_snapshots(
        problem=problem,
        reynolds_values=reynolds_values,
        amplitude_values=amplitude_values,
        initial_checkpoint=initial_checkpoint,
    )


# =============================================================================
# ROM-based Snapshot Collection (for DEIM)
# =============================================================================

def collect_rom_snapshots_for_deim_1p(
    problem,
    reynolds_values: np.ndarray,
    amplitude: float,
    pod_dir: str,
    lifting_dir: str,
    initial_checkpoint: str,
) -> SnapshotCollection:
    """
    Collect ROM-reconstructed snapshots for DEIM training (one-parameter).
    """
    print("=" * 80)
    print("DEIM TRAINING SNAPSHOT COLLECTION (ROM-based, ONE-PARAMETER)")
    print("=" * 80)
    print(f"Reynolds range: [{reynolds_values[0]}, {reynolds_values[-1]}]")
    print(f"Number of Reynolds values: {len(reynolds_values)}")
    print(f"Amplitude: {amplitude}")

    # Load POD basis and build operators
    print("\n--- Loading POD basis and operators ---")
    basis = load_pod_basis(pod_dir)
    lifting = load_lifting_functions(lifting_dir, problem)
    operators = build_reduced_operators(problem, basis, lifting)

    # Get initial guess
    initial_solution = load_solution(initial_checkpoint, problem)
    velocity_dofs = initial_solution.velocity.dat.data_ro.flatten()
    pressure_dofs = initial_solution.pressure.dat.data_ro.flatten()

    # Project to ROM coefficients
    from rom import project_to_rom_coefficients, assemble_inner_product_matrix
    M_u = assemble_inner_product_matrix(problem.velocity_space, 'H1_semi')
    M_p = assemble_inner_product_matrix(problem.pressure_space, 'L2')

    u_coeffs_init, _ = project_to_rom_coefficients(
        velocity_dofs, pressure_dofs, operators, amp=amplitude, M_u=M_u, M_p=M_p
    )

    # Collect ROM solutions
    solutions = []
    parameters = []
    lift_coefficients = []
    drag_coefficients = []

    prev_guess = u_coeffs_init

    print("\n--- Solving ROM at each parameter value ---")
    for i, re in enumerate(reynolds_values):
        nu = 1.0 / re

        solution = solve_rom(
            operators=operators,
            nu=nu,
            amp=amplitude,
            problem=problem,
            initial_guess=prev_guess,
            store_full_dofs=True,
        )

        u_rom, p_rom = reconstruct_solution(solution, operators, problem)

        from firedrake import Function
        sol = Function(problem.mixed_space)
        sol.sub(0).dat.data[:] = u_rom.dat.data[:]
        sol.sub(1).dat.data[:] = p_rom.dat.data[:]

        solutions.append(sol)
        parameters.append(np.array([re, amplitude]))

        from .navier_stokes import compute_forces
        try:
            forces = compute_forces(u_rom, p_rom, problem)
            lift_coefficients.append(forces.lift)
            drag_coefficients.append(forces.drag)
        except:
            lift_coefficients.append(0.0)
            drag_coefficients.append(0.0)

        print(f"  Re={re:.1f}: converged={solution.converged}, iter={solution.iterations}")

        prev_guess = solution.velocity_coeffs

    print("\n" + "=" * 80)
    print("ROM SNAPSHOT COLLECTION COMPLETE")
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
            'n_snapshots': len(solutions),
            'source': 'ROM',
            'pod_dir': pod_dir,
        }
    )


def collect_rom_snapshots_for_deim_2p(
    problem,
    reynolds_values: np.ndarray,
    amplitude_values: np.ndarray,
    pod_dir: str,
    lifting_dir: str,
    initial_checkpoint: str,
) -> SnapshotCollection:
    """
    Collect ROM-reconstructed snapshots for DEIM training (two-parameter).
    """
    print("=" * 80)
    print("DEIM TRAINING SNAPSHOT COLLECTION (ROM-based, TWO-PARAMETER)")
    print("=" * 80)
    print(f"Reynolds range: [{reynolds_values[0]}, {reynolds_values[-1]}]")
    print(f"Number of Reynolds values: {len(reynolds_values)}")
    print(f"Amplitude range: [{amplitude_values[0]}, {amplitude_values[-1]}]")
    print(f"Number of Amplitude values: {len(amplitude_values)}")
    print(f"Total snapshots: {len(reynolds_values) * len(amplitude_values)}")

    # Load POD basis and build operators
    print("\n--- Loading POD basis and operators ---")
    basis = load_pod_basis(pod_dir)
    lifting = load_lifting_functions(lifting_dir, problem)
    operators = build_reduced_operators(problem, basis, lifting)

    # Get initial guess
    initial_solution = load_solution(initial_checkpoint, problem)
    velocity_dofs = initial_solution.velocity.dat.data_ro.flatten()
    pressure_dofs = initial_solution.pressure.dat.data_ro.flatten()

    from rom import project_to_rom_coefficients, assemble_inner_product_matrix
    M_u = assemble_inner_product_matrix(problem.velocity_space, 'H1_semi')
    M_p = assemble_inner_product_matrix(problem.pressure_space, 'L2')

    solutions = []
    parameters = []
    lift_coefficients = []
    drag_coefficients = []

    print("\n--- Solving ROM across parameter grid ---")
    for amp in amplitude_values:
        print(f"\n  Amplitude = {amp:.4f}")

        u_coeffs_init, _ = project_to_rom_coefficients(
            velocity_dofs, pressure_dofs, operators, amp=amp, M_u=M_u, M_p=M_p
        )
        prev_guess = u_coeffs_init

        for re in reynolds_values:
            nu = 1.0 / re

            solution = solve_rom(
                operators=operators,
                nu=nu,
                amp=amp,
                problem=problem,
                initial_guess=prev_guess,
                store_full_dofs=True,
            )

            u_rom, p_rom = reconstruct_solution(solution, operators, problem)

            from firedrake import Function
            sol = Function(problem.mixed_space)
            sol.sub(0).dat.data[:] = u_rom.dat.data[:]
            sol.sub(1).dat.data[:] = p_rom.dat.data[:]

            solutions.append(sol)
            parameters.append(np.array([re, amp]))

            from .navier_stokes import compute_forces
            try:
                forces = compute_forces(u_rom, p_rom, problem)
                lift_coefficients.append(forces.lift)
                drag_coefficients.append(forces.drag)
            except:
                lift_coefficients.append(0.0)
                drag_coefficients.append(0.0)

            print(f"    Re={re:.1f}: converged={solution.converged}, iter={solution.iterations}")

            prev_guess = solution.velocity_coeffs

    print("\n" + "=" * 80)
    print("ROM SNAPSHOT COLLECTION COMPLETE")
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
            'n_snapshots': len(solutions),
            'source': 'ROM',
            'pod_dir': pod_dir,
        }
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("SNAPSHOT GENERATION")
    print("=" * 80)
    print(f"Mode: {'TWO-PARAMETER (Reynolds + Amplitude)' if TWO_PARAMETER_MODE else 'ONE-PARAMETER (Reynolds only)'}")

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=100.0,
        amplitude_init=0.0
    )

    if TWO_PARAMETER_MODE:
        # =================================================================
        # TWO-PARAMETER MODE
        # =================================================================
        pod_re = POD_RE_VALUES_2P
        pod_amp = POD_AMP_VALUES_2P
        pod_output = POD_OUTPUT_DIR_2P
        pod_dofs = POD_DOFS_DIR_2P

        deim_re = DEIM_RE_VALUES_2P
        deim_amp = DEIM_AMP_VALUES_2P
        deim_output = DEIM_OUTPUT_DIR_2P
        deim_dofs = DEIM_DOFS_DIR_2P

        rom_pod_dir = ROM_POD_DIR_2P

        # Step 1: POD snapshots
        print("\n" + "=" * 60)
        print("Step 1: POD Training Snapshots (Two-Parameter)")
        print("=" * 60)

        regenerate_pod, reason_pod = needs_snapshot_regeneration_2p(
            pod_dofs, pod_re, pod_amp
        )

        if regenerate_pod:
            print(f"Generating POD training snapshots: {reason_pod}")
            pod_collection = collect_fom_snapshots_2p(
                problem=problem,
                reynolds_values=pod_re,
                amplitude_values=pod_amp,
                initial_checkpoint=INITIAL_CHECKPOINT,
            )
            save_snapshots(pod_collection, pod_output)
            save_snapshot_dofs(pod_collection, pod_dofs)
        else:
            print(f"POD snapshots up-to-date at {pod_dofs}")
            metadata = load_snapshot_metadata(pod_dofs)
            if metadata:
                print(f"  Existing: {metadata['n_snapshots']} snapshots")

        # Step 2: DEIM snapshots
        print("\n" + "=" * 60)
        print("Step 2: DEIM Training Snapshots (Two-Parameter)")
        print("=" * 60)

        regenerate_deim, reason_deim = needs_snapshot_regeneration_2p(
            deim_dofs, deim_re, deim_amp
        )

        if regenerate_deim:
            print(f"Generating DEIM training snapshots: {reason_deim}")

            if USE_ROM_FOR_DEIM:
                print("Source: ROM solutions")
                deim_collection = collect_rom_snapshots_for_deim_2p(
                    problem=problem,
                    reynolds_values=deim_re,
                    amplitude_values=deim_amp,
                    pod_dir=rom_pod_dir,
                    lifting_dir=LIFTING_DIR,
                    initial_checkpoint=INITIAL_CHECKPOINT,
                )
            else:
                print("Source: FOM solutions")
                deim_collection = collect_fom_snapshots_2p(
                    problem=problem,
                    reynolds_values=deim_re,
                    amplitude_values=deim_amp,
                    initial_checkpoint=INITIAL_CHECKPOINT,
                )

            save_snapshots(deim_collection, deim_output)
            save_snapshot_dofs(deim_collection, deim_dofs)
        else:
            print(f"DEIM snapshots up-to-date at {deim_dofs}")
            metadata = load_snapshot_metadata(deim_dofs)
            if metadata:
                print(f"  Existing: {metadata['n_snapshots']} snapshots")

        # Summary
        print("\n" + "=" * 80)
        print("SNAPSHOT GENERATION COMPLETE (TWO-PARAMETER)")
        print("=" * 80)
        print(f"POD snapshots:  {pod_dofs}")
        print(f"  Grid: {len(pod_re)} Re x {len(pod_amp)} Amp = {len(pod_re)*len(pod_amp)} snapshots")
        print(f"  Re: [{pod_re[0]:.1f}, {pod_re[-1]:.1f}], Amp: [{pod_amp[0]:.2f}, {pod_amp[-1]:.2f}]")
        print(f"DEIM snapshots: {deim_dofs}")
        print(f"  Grid: {len(deim_re)} Re x {len(deim_amp)} Amp = {len(deim_re)*len(deim_amp)} snapshots")
        print(f"  Re: [{deim_re[0]:.1f}, {deim_re[-1]:.1f}], Amp: [{deim_amp[0]:.2f}, {deim_amp[-1]:.2f}]")

    else:
        # =================================================================
        # ONE-PARAMETER MODE
        # =================================================================
        pod_re = POD_RE_VALUES_1P
        pod_amp = POD_AMPLITUDE_1P
        pod_output = POD_OUTPUT_DIR_1P
        pod_dofs = POD_DOFS_DIR_1P

        deim_re = DEIM_RE_VALUES_1P
        deim_amp = DEIM_AMPLITUDE_1P
        deim_output = DEIM_OUTPUT_DIR_1P
        deim_dofs = DEIM_DOFS_DIR_1P

        rom_pod_dir = ROM_POD_DIR_1P

        # Step 1: POD snapshots
        print("\n" + "=" * 60)
        print("Step 1: POD Training Snapshots (One-Parameter)")
        print("=" * 60)

        regenerate_pod, reason_pod = needs_snapshot_regeneration_1p(
            pod_dofs, pod_re, pod_amp
        )

        if regenerate_pod:
            print(f"Generating POD training snapshots: {reason_pod}")
            pod_collection = collect_fom_snapshots_1p(
                problem=problem,
                reynolds_values=pod_re,
                amplitude=pod_amp,
                initial_checkpoint=INITIAL_CHECKPOINT,
            )
            save_snapshots(pod_collection, pod_output)
            save_snapshot_dofs(pod_collection, pod_dofs)
        else:
            print(f"POD snapshots up-to-date at {pod_dofs}")
            metadata = load_snapshot_metadata(pod_dofs)
            if metadata:
                print(f"  Existing: {metadata['n_snapshots']} snapshots")
                if metadata.get('reynolds_values') is not None:
                    re = metadata['reynolds_values']
                    print(f"  Reynolds: [{re[0]:.1f}, {re[-1]:.1f}]")

        # Step 2: DEIM snapshots
        print("\n" + "=" * 60)
        print("Step 2: DEIM Training Snapshots (One-Parameter)")
        print("=" * 60)

        regenerate_deim, reason_deim = needs_snapshot_regeneration_1p(
            deim_dofs, deim_re, deim_amp
        )

        if regenerate_deim:
            print(f"Generating DEIM training snapshots: {reason_deim}")

            if USE_ROM_FOR_DEIM:
                print("Source: ROM solutions")
                deim_collection = collect_rom_snapshots_for_deim_1p(
                    problem=problem,
                    reynolds_values=deim_re,
                    amplitude=deim_amp,
                    pod_dir=rom_pod_dir,
                    lifting_dir=LIFTING_DIR,
                    initial_checkpoint=INITIAL_CHECKPOINT,
                )
            else:
                print("Source: FOM solutions")
                deim_collection = collect_fom_snapshots_1p(
                    problem=problem,
                    reynolds_values=deim_re,
                    amplitude=deim_amp,
                    initial_checkpoint=INITIAL_CHECKPOINT,
                )

            save_snapshots(deim_collection, deim_output)
            save_snapshot_dofs(deim_collection, deim_dofs)
        else:
            print(f"DEIM snapshots up-to-date at {deim_dofs}")
            metadata = load_snapshot_metadata(deim_dofs)
            if metadata:
                print(f"  Existing: {metadata['n_snapshots']} snapshots")
                if metadata.get('reynolds_values') is not None:
                    re = metadata['reynolds_values']
                    print(f"  Reynolds: [{re[0]:.1f}, {re[-1]:.1f}]")

        # Summary
        print("\n" + "=" * 80)
        print("SNAPSHOT GENERATION COMPLETE (ONE-PARAMETER)")
        print("=" * 80)
        print(f"POD snapshots:  {pod_dofs}")
        print(f"  {len(pod_re)} snapshots, Re=[{pod_re[0]:.1f}, {pod_re[-1]:.1f}], amp={pod_amp}")
        print(f"DEIM snapshots: {deim_dofs}")
        print(f"  {len(deim_re)} snapshots, Re=[{deim_re[0]:.1f}, {deim_re[-1]:.1f}], amp={deim_amp}")

    print("\nNext steps:")
    print("  1. Compute POD basis from POD snapshots")
    print("  2. Compute DEIM basis from DEIM snapshots")
    print("  3. Build ROM operators and DEIM online operators")


if __name__ == "__main__":
    main()