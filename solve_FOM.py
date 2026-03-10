import numpy as np
import time

from Navier_Stokes import (
    setup_navier_stokes_problem,
    solve_steady_navier_stokes,
    load_solution,
    save_solution,
    compute_forces,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

MESH_FILE      = "mesh/mid_pinball.msh"
REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

TEST_RE  = 100.0
TEST_AMP = 0.0
CHECKPOINT = "velocity_checkpoint_symm"
# Optional: path to a checkpoint to compare against (set to None to skip)
FOM_CHECKPOINT = f"states/{CHECKPOINT}"

# =============================================================================
# MAIN
# =============================================================================

def main():

    # --- Setup problem ---
    print("--- Setting up Navier-Stokes problem ---")
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )

    # --- Assign test parameters ---
    problem.reynolds.assign(TEST_RE)
    problem.amplitude.assign(TEST_AMP)
    print(f"  Re  = {TEST_RE}")
    print(f"  amp = {TEST_AMP}")
    fom_solution = load_solution(FOM_CHECKPOINT, problem)


    # --- Solve ---
    print("\n--- Solving steady Navier-Stokes ---")
    t0 = time.perf_counter()
    solution = solve_steady_navier_stokes(problem,fom_solution.solution)
    t_fom = time.perf_counter() - t0

    u_fom = solution.velocity.dat.data_ro.flatten()
    p_fom = solution.pressure.dat.data_ro.flatten()

    print(f"  FOM solve time: {t_fom:.3f}s")
    print(f"  Velocity DOFs:  {u_fom.shape[0]}")
    print(f"  Pressure DOFs:  {p_fom.shape[0]}")
    print(f"  |u|_2 = {np.linalg.norm(u_fom):.6e}")
    print(f"  |p|_2 = {np.linalg.norm(p_fom):.6e}")
    forces = compute_forces(solution,problem)
    print(f' THE LIFT FORCE = {forces.lift}')

    save_solution(solution,f'initial_states/{CHECKPOINT}')




if __name__ == "__main__":
    main()
