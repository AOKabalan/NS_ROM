"""
Reduced Order Modeling package.

Modules:
    data_structures: Dataclasses (PODBasis, ReducedOperators, OnlineSystem, ROMSolution) and I/O
    pod: POD basis computation and supremizer enrichment
    operators: Reduced operator construction (with optional affine convection)
    rom_callbacks: Callback factory for residual/Jacobian (internal)
    rom_solver: Unified ROM solver (exact or DEIM, auto-detected)

Typical workflow:
    1. Compute POD basis from snapshots: compute_pod_basis()
    2. Build reduced operators: build_reduced_operators()
    3. Solve ROM: solve_rom(operators, nu, amp, problem)
    4. Reconstruct full-order solution: reconstruct_solution()

DEIM workflow:
    1. Build operators with affine convection:
       build_reduced_operators(..., compute_affine_convection=True)
    2. Prepare DEIM online operators externally
    3. Solve ROM with DEIM: solve_rom(operators, nu, amp, problem, deim_ops=my_deim)
"""

# Data structures
from .data_structures import (
    PODBasis,
    ReducedOperators,
    OnlineSystem,
    ROMSolution,
    save_pod_basis,
    load_pod_basis,
    save_reduced_operators,
    load_reduced_operators,
    save_rom_solution,
    load_rom_solution,
    save_rom_solutions_batch,
    load_rom_solutions_batch,
)

# POD computation
from .pod import (
    # Inner products
    assemble_inner_product_matrix,
    # Snapshot processing
    homogenize_snapshots,
    # POD
    build_correlation_matrix,
    solve_eigenvalue_problem,
    truncate_modes,
    compute_pod_modes,
    compute_pod_basis,
    # Supremizer
    compute_supremizer_snapshots,
    compute_supremizer_modes,
    # Utilities
    project_to_basis,
    reconstruct_from_basis,
    check_orthonormality,
    compute_reconstruction_error,
)

# Reduced operators
from .operators import (
    assemble_full_order_operators,
    build_reduced_operators,
    assemble_convection_matrix,
    assemble_convection_jacobian,
)

# Unified solver
from .rom_solver import (
    solve_rom,
    solve_rom_parameter_sweep,
    reconstruct_solution,
    reconstruct_velocity,
    reconstruct_pressure,
    project_to_rom_coefficients,
    compute_rom_error,
)
from .hyper_reduction import SubMeshDEIM

# Callbacks (internal, but exposed for advanced use / debugging)
from .rom_callbacks import make_callbacks


__all__ = [
    # ==========================================================================
    # DATA STRUCTURES
    # ==========================================================================
    'PODBasis',
    'ReducedOperators',
    'OnlineSystem',
    'ROMSolution',

    # ==========================================================================
    # I/O
    # ==========================================================================
    'save_pod_basis',
    'load_pod_basis',
    'save_reduced_operators',
    'load_reduced_operators',
    'save_rom_solution',
    'load_rom_solution',
    'save_rom_solutions_batch',
    'load_rom_solutions_batch',

    # ==========================================================================
    # POD
    # ==========================================================================
    'assemble_inner_product_matrix',
    'homogenize_snapshots',
    'build_correlation_matrix',
    'solve_eigenvalue_problem',
    'truncate_modes',
    'compute_pod_modes',
    'compute_pod_basis',
    'compute_supremizer_snapshots',
    'compute_supremizer_modes',
    'project_to_basis',
    'reconstruct_from_basis',
    'check_orthonormality',
    'compute_reconstruction_error',

    # ==========================================================================
    # OPERATORS
    # ==========================================================================
    'assemble_full_order_operators',
    'build_reduced_operators',
    'assemble_convection_matrix',
    'assemble_convection_jacobian',

    # ==========================================================================
    # SOLVER (unified — exact or DEIM)
    # ==========================================================================
    'solve_rom',
    'solve_rom_parameter_sweep',
    'reconstruct_solution',
    'reconstruct_velocity',
    'reconstruct_pressure',
    'project_to_rom_coefficients',
    'compute_rom_error',

    # ==========================================================================
    # CALLBACKS (internal / advanced)
    # ==========================================================================
    'make_callbacks',

    ############################################################################
    'SubMeshDEIM',
]