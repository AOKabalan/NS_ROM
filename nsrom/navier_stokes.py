from firedrake import *
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NavierStokesProblem:
    """Problem definition with mutable parameters"""
    mesh: Mesh
    velocity_space: FunctionSpace
    pressure_space: FunctionSpace
    mixed_space: MixedFunctionSpace
    boundary_conditions: List
    # Mutable parameters as Constants
    reynolds: Constant
    amplitude: Constant
    mesh_file: str = None

@dataclass
class NavierStokesSolution:
    """Solution container"""
    solution: Function  # Mixed function
    velocity: Function
    pressure: Function
    reynolds: float
    amplitude: float
    converged: Optional[bool] = None
    num_iterations: Optional[int] = None

@dataclass
class ForceCoefficients:
    """Drag and lift coefficients"""
    drag: float
    lift: float
    reynolds: float
    amplitude: float

# =============================================================================
# PROBLEM SETUP
# =============================================================================

def setup_navier_stokes_problem(
    mesh_file: str,
    reynolds_init: float = 100.0,
    amplitude_init: float = 0.0,
    comm=COMM_WORLD
) -> NavierStokesProblem:
    """
    Setup problem once. Reynolds and amplitude can be changed later.

    Args:
        mesh_file: Path to mesh file
        reynolds_init: Initial Reynolds number
        amplitude_init: Initial amplitude for boundary actuation
        comm: MPI communicator

    Returns:
        NavierStokesProblem with mutable parameters
    """
    # Load mesh (expensive - do once)
    mesh = Mesh(mesh_file, comm=comm)

    # Create function spaces (expensive - do once)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, Q])

    # Parameters as Constants (cheap to update)
    Re = Constant(reynolds_init)
    amp = Constant(amplitude_init)

    # Boundary conditions with Constants
    inflow = Constant((1.0, 0.0))
    noslip = Constant((0.0, 0.0))
    x = SpatialCoordinate(mesh)

    cx_top, cy_top = 0.0, 0.75
    cx_bottom, cy_bottom = 0.0, -0.75

    top_expr = as_vector([
        amp * sin(atan2(x[1] - cy_top, x[0] - cx_top)),
        -amp * cos(atan2(x[1] - cy_top, x[0] - cx_top))
    ])
    bottom_expr = as_vector([
        -amp * sin(atan2(x[1] - cy_bottom, x[0] - cx_bottom)),
        amp * cos(atan2(x[1] - cy_bottom, x[0] - cx_bottom))
    ])
    bcs = [
        DirichletBC(Z.sub(0), inflow, (1, 3)),      # ← Use Z.sub(0)
        DirichletBC(Z.sub(0), noslip, 4),
        DirichletBC(Z.sub(0), top_expr, 6),
        DirichletBC(Z.sub(0), bottom_expr, 5)
    ]
    return NavierStokesProblem(
        mesh=mesh,
        velocity_space=V,
        pressure_space=Q,
        mixed_space=Z,
        boundary_conditions=bcs,
        reynolds=Re,
        amplitude=amp,
        mesh_file=mesh_file
    )

# =============================================================================
# PHYSICS
# =============================================================================

def navier_stokes_residual(
    solution: Function,
    test_function: Function,
    reynolds: Constant
) -> Form:
    """Navier-Stokes residual form"""
    u, p = split(solution)
    v, q = split(test_function)

    f = Constant((0, 0))

    F = (
        (1/reynolds) * inner(grad(u), grad(v)) * dx
        + inner(grad(u) * u, v) * dx
        - div(v) * p * dx
        - inner(f, v) * dx
        + q * div(u) * dx
    )
    return F

# =============================================================================
# SOLVER
# =============================================================================
def run_picard_iterations(problem, u_init, num_picard=10):
    Z = problem.mixed_space
    Re = problem.reynolds
    V = problem.velocity_space
    Q = problem.pressure_space

    u_k = Function(V)
    p_k = Function(Q)

    u0, p0 = u_init.subfunctions
    u_k.assign(u0)
    p_k.assign(p0)

    u_trial, p_trial = TrialFunctions(Z)
    v_test, q_test = TestFunctions(Z)

    f = Constant((0.0, 0.0))

    # Nullspace for pressure
    nullspace = MixedVectorSpaceBasis(
        Z,
        [
            Z.sub(0),
            VectorSpaceBasis(constant=True)
        ]
    )

    u_next = Function(Z)

    # Simple direct solver - most robust for debugging
    solver_params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    print(f"Running {num_picard} Picard iterations...")

    for it in range(num_picard):
        a = (
            (1.0/Re) * inner(grad(u_trial), grad(v_test))
            + inner(dot(grad(u_trial),u_k), v_test)
            - div(v_test) * p_trial
            + q_test * div(u_trial)
        ) * dx

        L = inner(f, v_test) * dx

        solve(
            a == L,
            u_next,
            bcs=problem.boundary_conditions,
            nullspace=nullspace,
            solver_parameters=solver_params
        )

        u_new, p_new = u_next.subfunctions
        diff = norm(u_new - u_k)
        print(f"  Picard iteration {it+1}/{num_picard}, ||u - u_k|| = {diff:.6e}")

        # Apply relaxation
        u_k.assign( u_new)
        p_k.assign( p_new)

        if diff < 1e-8:
            print("  Converged!")
            break

    final_mixed = Function(Z)
    final_mixed.sub(0).assign(u_k)
    final_mixed.sub(1).assign(p_k)

    return final_mixed


def solve_steady_navier_stokes(
    problem: NavierStokesProblem,
    initial_guess: Function = None,
    solver_parameters: dict = None,
    use_picard: bool = False
) -> NavierStokesSolution:
    """
    Solve steady Navier-Stokes with current parameter values.

    Args:
        problem: Problem definition
        initial_guess: Optional initial guess (for continuation)
        solver_parameters: Optional solver parameters

    Returns:
        NavierStokesSolution
    """
    # Initial guess
    solution = Function(problem.mixed_space)
    if initial_guess is not None:
        # solution = initial_guess.copy(deepcopy=True)
        solution.assign(initial_guess)  #
    if use_picard:
        print("Running Picard iterations before Newton...")
        solution = run_picard_iterations(problem, solution, num_picard=10)
    # Test function
    test = TestFunction(problem.mixed_space)

    # Residual and Jacobian
    F = navier_stokes_residual(solution, test, problem.reynolds)
    J = derivative(F, solution)

    # Default solver parameters
    if solver_parameters is None:
        solver_parameters = {
            "mat_type": "aij",
            "snes_type":"newtonls",
            "snes_monitor": None,
            "snes_linesearch_type": "bt",
            "snes_max_it": 100,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }

    # Setup and solve
    nvp = NonlinearVariationalProblem(F, solution, problem.boundary_conditions, J)
    solver = NonlinearVariationalSolver(nvp, solver_parameters=solver_parameters)
    try:
        solver.solve()
    except ConvergenceError as e:
        reason = solver.snes.getConvergedReason()
        num_iterations = solver.snes.getIterationNumber()
        converged = int(reason) > 0
        print(f"Solver diverged with reason {reason} after {num_iterations} iterations: {e}")
    else:
        reason = solver.snes.getConvergedReason()
        num_iterations = solver.snes.getIterationNumber()
        converged = int(reason) > 0
    # Extract velocity and pressure
    u, p = solution.subfunctions

    return NavierStokesSolution(
        solution=solution,
        velocity=u,
        pressure=p,
        reynolds=float(problem.reynolds),
        amplitude=float(problem.amplitude),
        converged=converged,
        num_iterations=num_iterations
    )


def solve_with_continuation(
    problem: NavierStokesProblem,
    parameter_name: str,
    parameter_values: np.ndarray,
    initial_solution: NavierStokesSolution = None
) -> Tuple[List[NavierStokesSolution],List[ForceCoefficients]]:
    """
    Solve for multiple parameter values using continuation.
    Uses previous solution as initial guess.

    Args:
        problem: Problem definition
        parameter_name: 'reynolds' or 'amplitude'
        parameter_values: Array of parameter values
        initial_solution: Optional initial solution for first parameter

    Returns:
        List of solutions
    """
    solutions = []
    forces = []
    prev_solution = initial_solution.solution if initial_solution else None
    use_picard = True

    for param_value in parameter_values:
        print(f"Solving for {parameter_name} = {param_value:.4f}")

        # Update parameter
        if parameter_name == 'reynolds':
            problem.reynolds.assign(param_value)
        elif parameter_name == 'amplitude':
            problem.amplitude.assign(param_value)
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")

        # Solve with continuation
        solution = solve_steady_navier_stokes(problem, initial_guess=prev_solution, use_picard=use_picard)
        solutions.append(solution)

        force = compute_forces(solution, problem,
                                        reference_length=1.0,
                                        reference_velocity=1.0)
        forces.append(force)



        # Use this solution as initial guess for next
        prev_solution = solution.solution
        use_picard = False

    return solutions, forces

# =============================================================================
# I/O FUNCTIONS
# =============================================================================

def save_solution(
    solution: NavierStokesSolution,
    filename: str,
    mesh: Mesh = None
):
    """
    Save solution to HDF5 file.

    Args:
        solution: Solution to save
        filename: Output filename (without .h5 extension)
        mesh: Optional mesh (if not provided, extracted from solution)
    """
    # os.makedirs(os.path.dirname(filename), exist_ok=True)

    if mesh is None:
        mesh = solution.solution.function_space().mesh()
    u, p = solution.solution.subfunctions

    with CheckpointFile(f'{filename}'+ '.h5', 'w') as afile:
        afile.save_mesh(mesh)

        afile.save_function(u, name="velocity")
        afile.save_function(p, name="pressure")

    print(f"Solution saved to {filename}.h5")

def save_to_paraview(
    solution: NavierStokesSolution,
    filename: str
):
    """
    Save solution to VTK format for ParaView visualization.
    Writes both velocity and pressure fields.

    Args:
        solution: Solution to export
        filename: Output filename (e.g., 'output/solution.pvd')
    """
    # Extract velocity and pressure
    u, p = solution.solution.subfunctions

    # Rename for clarity in ParaView
    u.rename("Velocity", "Velocity")
    p.rename("Pressure", "Pressure")

    # Write to VTK file
    pvd = VTKFile(filename)
    pvd.write(u, p)

    print(f"Solution exported to ParaView: {filename}")

def load_solution(
    filename: str,
    problem: NavierStokesProblem,
    with_pressure: bool = False
) -> NavierStokesSolution:
    """
    Load solution from HDF5 file.

    Args:
        filename: Input filename (without .h5 extension)
        problem: Problem definition (for function space)

    Returns:
        NavierStokesSolution
    """
    V = problem.mixed_space.sub(0).collapse()
    sol = Function(problem.mixed_space,name="solution")

    with CheckpointFile(filename + '.h5', 'r') as afile:
        mesh_in = afile.load_mesh()
        u_loaded = afile.load_function(mesh_in, name="velocity")

        if with_pressure:
            p_loaded = afile.load_function(mesh_in, name="pressure")

    sol.sub(0).dat.data[:] = u_loaded.dat.data[:]
    if with_pressure:
        sol.sub(1).dat.data[:] = p_loaded.dat.data[:]
    u, p = sol.subfunctions
    reynolds = 0.0
    amplitude = 0.0

    print(f"Solution loaded from {filename}.h5")
    return NavierStokesSolution(
        solution=sol,
        velocity=u,
        pressure=p,
        reynolds=problem.reynolds.values()[0],
        amplitude=problem.amplitude.values()[0]
    )

# =============================================================================
# FORCE COMPUTATION
# =============================================================================

def compute_forces(
    solution: NavierStokesSolution,
    problem: NavierStokesProblem,
    reference_length: float = 1.0,
    reference_velocity: float = 1.0
) -> ForceCoefficients:
    """
    Compute drag and lift coefficients on cylinder.

    Args:
        solution: Solution to analyze
        problem: Problem definition
        cylinder_marker: Boundary marker for cylinder
        reference_length: Reference length for non-dimensionalization
        reference_velocity: Reference velocity

    Returns:
        ForceCoefficients with drag and lift
    """
    u, p = split(solution.solution)
    Re = solution.reynolds

    # Normal and tangent vectors
    n = FacetNormal(problem.mesh)

    # Stress tensor: σ = -pI + (1/Re)∇u
    stress = -p * Identity(2) + (1/Re) * (grad(u) + grad(u).T)

    # Force vector
    force = dot(stress, n)



    ds_cylinder = ds(4) + ds(5) + ds(6)

    # Integrate over cylinder surface
    drag_form = force[0] * ds_cylinder
    lift_form = force[1] * ds_cylinder

    drag = assemble(drag_form)
    lift = assemble(lift_form)

    # Non-dimensionalization: C_D = F_D / (0.5 * ρ * U^2 * L)
    # Here assuming ρ=1, dynamic pressure = 0.5 * U^2
    dynamic_pressure = 0.5 * reference_velocity**2
    drag_coeff = drag / (dynamic_pressure * reference_length)
    lift_coeff = lift / (dynamic_pressure * reference_length)

    return ForceCoefficients(
        drag=drag_coeff,
        lift=lift_coeff,
        reynolds=solution.reynolds,
        amplitude=solution.amplitude
    )

# def save_as_numpy(
#     solution: NavierStokesSolution,
#     save_path: str,


# )
# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # EXAMPLE 1: Single solve
    # =========================================================================
    print("=" * 60)
    print("EXAMPLE 1: Single solve")
    print("=" * 60)

    # Setup problem (expensive - do once)
    problem = setup_navier_stokes_problem(
        mesh_file='mesh/mid_pinball.msh',
        reynolds_init=80.0,
        amplitude_init=0.0
    )

    # Solve

    # loaded_solution = load_solution("states/velocity_checkpoint_up", problem)

    print("EXAMPLE 3: Parameter sweep with continuation")
    print("=" * 60)
    loaded_solution = load_solution("states/velocity_checkpoint_down",problem)

    # Reset parameters
    problem.reynolds.assign(100.0)
    problem.amplitude.assign(0.0)
    Sol = solve_steady_navier_stokes(
        problem,
        initial_guess = loaded_solution.solution,
        use_picard = True
    )
    print(f'Solved for paramtere Re: {Sol.reynolds}, and amp: {Sol.amplitude}')
    save_solution(Sol,"newlygenerated_high")
    save_to_paraview(Sol,"newlygenerated_high.pvd")


    err =errornorm(Sol.velocity, loaded_solution.velocity, norm_type='L2')
    print(f' The error btween calculated and read : {err}')
    # # Solve for multiple Reynolds numbers
    # re_values = np.array([10.0,  20.0, 30.0, 60.0,65.0,70.0,75.0,80.0, 100.0])
    # reverse_re_values = np.flip(re_values)
    # solutions,forces = solve_with_continuation(
    #     problem,
    #     parameter_name='reynolds',
    #     parameter_values=reverse_re_values,
    #     initial_solution=loaded_solution
    # )

    # # Analyze all solutions
    # print("\nResults:")
    # print(f"{'Re':<10} {'C_D':<15} {'C_L':<15}")
    # print("-" * 55)

    # for sol in forces:
    #     print(f"{sol.reynolds:<10.1f}"
    #           f"{sol.drag:<15.6f} {sol.lift:<15.6f}")
