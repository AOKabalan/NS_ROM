"""Phase A3: per-iterate DEIM/MDEIM error along the Newton path.

For each chosen amp, both solves are seeded from the PREVIOUS amp's exact
solution -- the same warm start the real continuation uses, so iterate 1
of each trace is exactly "the first iteration" of a continuation step.

  Run 1: EXACT solve, spy armed with deim_ops.
         Logs, at every iterate of the healthy 3-iteration path, what
         DEIM/MDEIM *would* have returned from the same full assembly.
         -> shows whether the DEIM error grows off-manifold or stays at
            the ~2e-4 solution-level floor along the whole path.

  Run 2: DEIM solve, spy armed plain.
         Logs, at every iterate of the (possibly dying) DEIM path, the
         exact full-mesh quantities at the same states.
         -> shows what Newton is actually being fed as it fails: watch
            rel err and cos(exact, deim) of F as the line search dies.

Wiring: the spy lives inside the callbacks, so solve_rom must import the
instrumented module. Replace the contents of your rom_callbacks.py with
rom_callbacks_spy.py (identical behavior when the spy is disabled), then:

    from nsrom.rom.rom_callbacks import (      # adjust to your layout
        enable_spy, disable_spy, print_spy_trace)

and call trace_newton_paths(...) after Phase A1 in main_local.
"""

import numpy as np

from nsrom.rom import solve_rom, project_to_rom_coefficients, SubMeshDEIM

# adjust this import to wherever the instrumented callbacks module lives:
from nsrom.rom.rom_callbacks_spy import enable_spy, disable_spy, print_spy_trace


def trace_newton_paths(
    exact_states,          # from diagnose_spine Phase A1
    amps_of_interest,      # e.g. [-0.1, -0.3, -0.5]
    Re,
    operators, deim_ops_dict, problem, M_u, M_p,
    cluster=1,
):
    nu = 1.0 / Re
    problem.reynolds.assign(Re)
    k = cluster
    sm = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    amp_list = [s[0] for s in exact_states]
    amp_arr = np.array(amp_list)
    for amp in amps_of_interest:
        i = int(np.argmin(np.abs(amp_arr - amp)))
        # optional safety check that it's actually a match, not just closest
        if abs(amp_arr[i] - amp) > 1e-9:
            raise ValueError(f"no state near amp={amp} (closest {amp_arr[i]})")
    # for amp in amps_of_interest:
    #     i = amp_list.index(amp)
        if i == 0:
            print(f"amp={amp} has no predecessor state; skipping")
            continue
        # warm start = previous amp's exact solution (as the continuation does)
        _, _, full_prev, _, _ = exact_states[i - 1]
        pv = full_prev.velocity.dat.data_ro.flatten()
        pp = full_prev.pressure.dat.data_ro.flatten()
        u_g, p_g = project_to_rom_coefficients(
            pv, pp, operators[k], amp=exact_states[i - 1][0], M_u=M_u, M_p=M_p)

        print("\n" + "=" * 72)
        print(f"TRACE amp={amp:+.3f} (warm start from amp="
              f"{exact_states[i-1][0]:+.3f})")
        print("=" * 72)

        # ---- Run 1: exact solve, DEIM shadow --------------------------
        enable_spy(deim_ops=deim_ops_dict[k])
        sol, _, _ = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=None,
            u_initial_guess=u_g, p_initial_guess=p_g,
            submesh_deim=None, return_internals=True)
        print_spy_trace(f"EXACT path (converged={sol.converged}) "
                        f"-- what DEIM would have said")
        disable_spy()

        # ---- Run 2: DEIM solve, exact shadow --------------------------
        enable_spy()
        sol_d, _, _ = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_g, p_initial_guess=p_g,
            submesh_deim=sm, return_internals=True)
        print_spy_trace(f"DEIM path (converged={sol_d.converged}) "
                        f"-- exact evaluated at DEIM iterates")
        disable_spy()


if __name__ == "__main__":
    # After diagnose_spine in main_local.py:
    #
    # from newton_trace import trace_newton_paths
    # trace_newton_paths(
    #     exact_states,
    #     amps_of_interest=[-0.1, -0.3, -0.5],
    #     Re=99.0,
    #     operators=operators, deim_ops_dict=deim_ops_dict,
    #     problem=problem, M_u=M_u, M_p=M_p,
    #     cluster=1,
    # )
    pass
