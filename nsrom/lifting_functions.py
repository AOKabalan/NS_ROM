# lifting_functions.py

from dataclasses import dataclass, field
from typing import List
import os
import json
import numpy as np
from firedrake import (
    Function, Constant, DirichletBC, CheckpointFile,
    SpatialCoordinate, as_vector, sin, cos, atan2, norm
)

from .navier_stokes import (
    NavierStokesProblem,
    NavierStokesSolution,
    setup_navier_stokes_problem,
    solve_steady_navier_stokes,
    load_solution
)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LiftingFunctions:
    """Lifting functions for non-homogeneous boundary conditions"""
    u_base: Function          # Base flow with inlet BC, no rotation
    u_control: Function       # Control flow with unit rotation, zero inlet
    problem: NavierStokesProblem
    reynolds_ref: float
    metadata: dict = field(default_factory=dict)

# =============================================================================
# LIFTING FUNCTION COMPUTATION
# =============================================================================

def create_lifting_bcs(
    problem: NavierStokesProblem,
    lift_type: str  # 'base' or 'control'
) -> List:
    """
    Create boundary conditions for lifting functions.

    Args:
        problem: Existing problem definition
        lift_type: 'base' (inlet BC, no rotation) or 'control' (unit rotation, zero inlet)

    Returns:
        List of DirichletBC for lifting problem
    """
    Z = problem.mixed_space
    x = SpatialCoordinate(problem.mesh)

    cx_top, cy_top = 0.0, 0.75
    cx_bottom, cy_bottom = 0.0, -0.75

    if lift_type == 'base':
        # Base flow: inlet BC, no rotation
        inflow = Constant((1.0, 0.0))
        noslip = Constant((0.0, 0.0))
        amp = Constant(0.0)

    elif lift_type == 'control':
        # Control flow: zero inlet, unit rotation
        inflow = Constant((0.0, 0.0))
        noslip = Constant((0.0, 0.0))
        amp = Constant(1.0)

    else:
        raise ValueError(f"Unknown lift_type: {lift_type}")

    # Rotation expressions (same as original)
    top_expr = as_vector([
        amp * sin(atan2(x[1] - cy_top, x[0] - cx_top)),
        -amp * cos(atan2(x[1] - cy_top, x[0] - cx_top))
    ])
    bottom_expr = as_vector([
        -amp * sin(atan2(x[1] - cy_bottom, x[0] - cx_bottom)),
        amp * cos(atan2(x[1] - cy_bottom, x[0] - cx_bottom))
    ])

    bcs = [
        DirichletBC(Z.sub(0), inflow, (1, 3)),
        DirichletBC(Z.sub(0), noslip, 4),
        DirichletBC(Z.sub(0), top_expr, 6),
        DirichletBC(Z.sub(0), bottom_expr, 5)
    ]

    return bcs


def compute_lifting_functions(
    problem: NavierStokesProblem,
    reynolds_ref: float = 100.0,
    initial_guess_base: NavierStokesSolution = None,
    initial_guess_control: NavierStokesSolution = None
) -> LiftingFunctions:
    """
    Compute lifting functions for non-homogeneous velocity BCs.

    Only velocity needs lifting since only velocity has non-homogeneous
    Dirichlet BCs. Pressure has no Dirichlet BCs to homogenize.

    Args:
        problem: Problem definition
        reynolds_ref: Reference Reynolds number
        initial_guess_base: Optional initial guess for u_base
        initial_guess_control: Optional initial guess for u_control

    Returns:
        LiftingFunctions containing u_base and u_control
    """
    print("=" * 80)
    print("COMPUTING LIFTING FUNCTIONS")
    print("=" * 80)
    print(f"Reference Reynolds number: {reynolds_ref}")

    # Save original BCs and parameters
    original_bcs = problem.boundary_conditions
    original_re = float(problem.reynolds)
    original_amp = float(problem.amplitude)

    # Set Reynolds number for lifting computation
    problem.reynolds.assign(reynolds_ref)

    # --- Compute u_base (inlet BC, no rotation) ---
    print("\n--- Computing u_base (inlet BC, no rotation) ---")
    problem.boundary_conditions = create_lifting_bcs(problem, 'base')
    problem.amplitude.assign(0.0)

    sol_base = solve_steady_navier_stokes(
        problem,
        initial_guess=initial_guess_base.solution if initial_guess_base else None
    )
    print(f"u_base: ||u|| = {norm(sol_base.velocity):.6f}")

    # --- Compute u_control (unit rotation, zero inlet) ---
    print("\n--- Computing u_control (unit rotation, zero inlet) ---")
    problem.boundary_conditions = create_lifting_bcs(problem, 'control')
    problem.amplitude.assign(1.0)

    sol_control = solve_steady_navier_stokes(
        problem,
        initial_guess=initial_guess_control.solution if initial_guess_control else None
    )
    print(f"u_control: ||u|| = {norm(sol_control.velocity):.6f}")

    # Restore original BCs and parameters
    problem.boundary_conditions = original_bcs
    problem.reynolds.assign(original_re)
    problem.amplitude.assign(original_amp)

    print("\n" + "=" * 80)
    print("LIFTING FUNCTIONS COMPUTED")
    print("=" * 80)

    return LiftingFunctions(
        u_base=sol_base.velocity,
        u_control=sol_control.velocity,
        problem=problem,
        reynolds_ref=reynolds_ref,
        metadata={
            'u_base_norm': float(norm(sol_base.velocity)),
            'u_control_norm': float(norm(sol_control.velocity))
        }
    )

# =============================================================================
# I/O FUNCTIONS
# =============================================================================

def save_lifting_functions(
    lifting: LiftingFunctions,
    output_dir: str = "lifting"
):
    """
    Save lifting functions to disk.

    Args:
        lifting: LiftingFunctions to save
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)

    mesh = lifting.problem.mesh

    # Save u_base
    with CheckpointFile(os.path.join(output_dir, "u_base.h5"), 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(lifting.u_base, name="velocity")

    # Save u_control
    with CheckpointFile(os.path.join(output_dir, "u_control.h5"), 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(lifting.u_control, name="velocity")

    # Save metadata
    metadata_file = os.path.join(output_dir, "lifting_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'reynolds_ref': lifting.reynolds_ref,
            **lifting.metadata
        }, f, indent=2)

    print(f"\nLifting functions saved to {output_dir}/")
    print(f"  - u_base.h5")
    print(f"  - u_control.h5")
    print(f"  - lifting_metadata.json")


def load_lifting_functions(
    input_dir: str,
    problem: NavierStokesProblem
) -> LiftingFunctions:
    """
    Load lifting functions from disk.

    Args:
        input_dir: Directory containing lifting functions
        problem: Problem definition (for function spaces)

    Returns:
        LiftingFunctions
    """
    V = problem.velocity_space

    # Load u_base
    u_base = Function(V, name="u_base")
    with CheckpointFile(os.path.join(input_dir, "u_base.h5"), 'r') as afile:
        mesh_in = afile.load_mesh()
        u_base_loaded = afile.load_function(mesh_in, name="velocity")
    u_base.dat.data[:] = u_base_loaded.dat.data[:]

    # Load u_control
    u_control = Function(V, name="u_control")
    with CheckpointFile(os.path.join(input_dir, "u_control.h5"), 'r') as afile:
        mesh_in = afile.load_mesh()
        u_control_loaded = afile.load_function(mesh_in, name="velocity")
    u_control.dat.data[:] = u_control_loaded.dat.data[:]

    # Load metadata
    metadata_file = os.path.join(input_dir, "lifting_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    reynolds_ref = metadata.pop('reynolds_ref')

    print(f"\nLifting functions loaded from {input_dir}/")
    print(f"  Reynolds reference: {reynolds_ref}")
    print(f"  ||u_base|| = {metadata.get('u_base_norm', 'N/A')}")
    print(f"  ||u_control|| = {metadata.get('u_control_norm', 'N/A')}")

    return LiftingFunctions(
        u_base=u_base,
        u_control=u_control,
        problem=problem,
        reynolds_ref=reynolds_ref,
        metadata=metadata
    )

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LIFTING FUNCTIONS - TEST")
    print("=" * 80)

    # Setup problem
    problem = setup_navier_stokes_problem(
        mesh_file='mesh/mid_pinball.msh',
        reynolds_init=100.0,
        amplitude_init=0.0
    )
    loaded_solution = load_solution("states/velocity_checkpoint_symm", problem)

    # Compute lifting functions at Re=100
    lifting = compute_lifting_functions(
        problem,
        reynolds_ref=100.0,
        initial_guess_base = loaded_solution
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Reference Reynolds: {lifting.reynolds_ref}")
    print(f"||u_base||:    {lifting.metadata['u_base_norm']:.6f}")
    print(f"||u_control||: {lifting.metadata['u_control_norm']:.6f}")

    # Save to disk
    save_lifting_functions(lifting, output_dir="lifting")

    # Test loading
    print("\n" + "=" * 80)
    print("TESTING LOAD")
    print("=" * 80)

    lifting_loaded = load_lifting_functions("lifting", problem)

    # Verify
    print("\nVerification:")
    print(f"Loaded ||u_base||:    {norm(lifting_loaded.u_base):.6f}")
    print(f"Loaded ||u_control||: {norm(lifting_loaded.u_control):.6f}")
    print(f"Match original: {np.allclose(norm(lifting.u_base), norm(lifting_loaded.u_base))}")

