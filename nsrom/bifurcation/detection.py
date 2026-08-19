"""
Pitchfork / zero-crossing detection along a Reynolds sweep.

This module provides:

* Indicator computation at the FOM level
  (`compute_fom_indicator`; the ROM counterpart lives in `branch_jump`).
* Parameter sweeps that record the leftmost real eigenvalue mu(Re)
  for the ROM (`rom_sweep`) and FOM (`fom_sweep`).
* A generic bisection driver (`bisect_zero_crossing`) that refines a
  mu(Re) sign change to a critical Reynolds number, plus thin pitchfork
  wrappers that consume sweep records directly
  (`bisect_pitchfork_rom_from_records`, `bisect_pitchfork_fom_from_records`).
* `build_reduced_velocity_l2_mass_matrices`, the L2 velocity mass matrix
  projected onto each local POD basis, used as the right-hand side of
  the reduced generalized eigenvalue problem driving the ROM indicator.

Single-point evaluators (`evaluate_rom_point`, `evaluate_fom_point`)
are exposed so callers can reuse the same eval logic for custom
bracketing strategies.
"""

import numpy as np

from nsrom.navier_stokes import solve_steady_navier_stokes
from nsrom.rom import assemble_inner_product_matrix
from nsrom.rom.local import solve_at_parameter, solve_at_parameter_with_internals
from .branch_jump import compute_rom_indicator
from .eigen_solver import solve_leftmost_real_eigenpairs
from .jacobian import build_state_jacobian_pencil


__all__ = [
    "DEFAULT_SYMMETRY_RECOVERY_TOL",
    "normalized_h1_distance",
    "classify_symmetry_recovery",
    "compute_fom_indicator",
    "build_reduced_velocity_l2_mass_matrices",
    "find_mu_bracket",
    "evaluate_rom_point",
    "evaluate_fom_point",
    "rom_sweep",
    "fom_sweep",
    "bisect_zero_crossing",
    "bisect_pitchfork_rom_from_records",
    "bisect_pitchfork_fom_from_records",
]


DEFAULT_SYMMETRY_RECOVERY_TOL = 1e-5


def normalized_h1_distance(candidate_velocity_dofs,
                           symmetric_reference_velocity_dofs,
                           M_u, *, epsilon=None):
    """Return the candidate's normalized H1 distance from a reference state.

    The normalization is ``max(||u_sym||_H1, epsilon)``. ``M_u`` may be a
    sparse matrix; it is applied directly and is never densified. Missing or
    nonfinite states produce ``nan`` rather than physical classification.
    """
    if (candidate_velocity_dofs is None
            or symmetric_reference_velocity_dofs is None):
        return float("nan")

    candidate = np.asarray(candidate_velocity_dofs, dtype=float).reshape(-1)
    reference = np.asarray(
        symmetric_reference_velocity_dofs, dtype=float).reshape(-1)
    if candidate.shape != reference.shape:
        raise ValueError(
            "candidate and symmetric-reference velocity states must have "
            f"the same shape, got {candidate.shape} and {reference.shape}"
        )
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        return float("nan")

    delta = candidate - reference
    distance_sq = float(delta @ (M_u @ delta))
    reference_norm_sq = float(reference @ (M_u @ reference))
    if (not np.isfinite(distance_sq)
            or not np.isfinite(reference_norm_sq)
            or distance_sq < 0.0
            or reference_norm_sq < 0.0):
        return float("nan")

    if epsilon is None:
        epsilon = np.finfo(float).eps
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon!r}")

    denominator = max(float(np.sqrt(reference_norm_sq)), epsilon)
    return float(np.sqrt(distance_sq) / denominator)


def classify_symmetry_recovery(candidate_velocity_dofs,
                               symmetric_reference_velocity_dofs,
                               M_u,
                               tolerance=DEFAULT_SYMMETRY_RECOVERY_TOL,
                               *, converged):
    """Classify numerical recovery of an independent symmetric solution.

    The boundary is inclusive: a finite normalized H1 distance equal to
    ``tolerance`` is classified as recovered. A failed solve, missing
    reference, or nonfinite distance always returns ``(False, nan)`` (or the
    measured nonfinite value) and never falls back to branch labels or lift.
    """
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"symmetry-recovery tolerance must be finite and nonnegative, "
            f"got {tolerance!r}"
        )
    if not converged:
        return False, float("nan")

    distance = normalized_h1_distance(
        candidate_velocity_dofs,
        symmetric_reference_velocity_dofs,
        M_u,
    )
    recovered = bool(np.isfinite(distance) and distance <= tolerance)
    return recovered, distance


# ---------------------------------------------------------------------------
# Indicator and reduced mass matrices
# ---------------------------------------------------------------------------

def compute_fom_indicator(problem, solution, n_eigenvalues=6, target=0.0):
    """
    Compute the FOM pitchfork indicator: the real eigenvalue of the
    state-Jacobian generalized eigenproblem closest to zero from the
    leftmost real spectrum.

    Returns a dict with keys ``mu``, ``eigvals``, ``eigvec``, ``nconv``,
    or ``None`` if no real eigenvalue was recovered.
    """
    J, M = build_state_jacobian_pencil(problem, solution)

    eigvals, vrs, vis, nconv = solve_leftmost_real_eigenpairs(
        J,
        M,
        n_eigenvalues=n_eigenvalues,
        target=target,
    )

    if nconv == 0:
        return None

    imag_tol = 1e-6 * max(abs(e) for e in eigvals)
    real_mask = np.abs(eigvals.imag) < imag_tol
    real_idx = np.where(real_mask)[0]

    if len(real_idx) == 0:
        return None

    # For pitchfork tracking: real eigenvalue closest to zero.
    i = int(real_idx[np.argmin(eigvals[real_mask].real)])
    return {
        "mu": float(eigvals[i].real),
        "eigvals": eigvals,
        "eigvec": vrs[i].copy(),
        "nconv": int(nconv),
    }


def build_reduced_velocity_l2_mass_matrices(problem, clustering, operators):
    """
    Build the L2 velocity mass matrix projected onto each cluster's
    POD velocity basis: ``M_red[k] = Z_u_k.T @ M_L2 @ Z_u_k``,
    symmetrized to compensate for numerical asymmetry.

    Returned as a dict keyed by cluster index.
    """
    M_L2 = assemble_inner_product_matrix(problem.velocity_space, "L2")

    M_uu_red = {}
    for k in range(clustering.n_clusters):
        Z_u_k = operators[k].Z_u
        M_red = Z_u_k.T @ (M_L2 @ Z_u_k)
        M_uu_red[k] = 0.5 * (M_red + M_red.T)

    return M_uu_red


def build_global_reduced_velocity_l2_mass_matrices(problem, operators):
    """
    Build the L2 velocity mass matrix projected onto each cluster's
    POD velocity basis: ``M_red = Z_u.T @ M_L2 @ Z_u``,
    symmetrized to compensate for numerical asymmetry.

    Returned as a dict keyed by cluster index.
    """
    M_L2 = assemble_inner_product_matrix(problem.velocity_space, "L2")

    Z_u = operators.Z_u
    M_red = Z_u.T @ (M_L2 @ Z_u)
    M_uu_red = 0.5 * (M_red + M_red.T)

    return M_uu_red
# ---------------------------------------------------------------------------
# Bracket location
# ---------------------------------------------------------------------------

def find_mu_bracket(records, mu_key="mu"):
    """
    Find neighbouring records where ``mu`` changes sign.

    Returns
    -------
    left, right : dict, dict
        Consecutive records sorted by Re such that
        ``mu_left * mu_right <= 0``. If a record has ``mu == 0`` exactly,
        ``left is right`` and points to that record. Returns
        ``(None, None)`` if no sign change is found.
    """
    clean = [
        r for r in records
        if r.get(mu_key) is not None and np.isfinite(r[mu_key])
    ]
    clean = sorted(clean, key=lambda r: r["Re"])

    for left, right in zip(clean[:-1], clean[1:]):
        mu_left = left[mu_key]
        mu_right = right[mu_key]

        if mu_left == 0.0:
            return left, left

        if mu_left * mu_right < 0.0:
            return left, right

    return None, None


# ---------------------------------------------------------------------------
# Single-point evaluators
# ---------------------------------------------------------------------------

def evaluate_rom_point(
    *,
    Re,
    amp,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    mode,
    n_eigenvalues=6,
    target=0.0,
):
    """
    Solve the ROM at (Re, amp) starting from ``initial_solution`` and
    compute the ROM pitchfork indicator on success. Never raises:
    failures are returned as ``converged=False`` records with a
    ``failed_reason`` string.

    Notes
    -----
    The ROM solve always starts from the same ``initial_solution`` so
    that bisection points are deterministic and independent of the
    order in which they are evaluated.
    """
    try:
        rom_res = solve_at_parameter(
            Re=Re,
            amp=amp,
            clustering=clustering,
            operators=operators,
            deim_ops_dict=deim_ops_dict,
            problem=problem,
            initial_solution=initial_solution,
            M_u=M_u,
            M_p=M_p,
            mode=mode,
            return_internals=True,
        )

    except Exception as e:
        return {
            "Re": float(Re),
            "mu": None,
            "rom_res": None,
            "converged": False,
            "cluster": None,
            "iters": None,
            "residual_norm": None,
            "failed_reason": f"solve_exception: {e}",
        }

    rom_solution = rom_res.solution

    if not rom_solution.converged:
        return {
            "Re": float(Re),
            "mu": None,
            "rom_res": rom_res,
            "converged": False,
            "cluster": getattr(rom_res, "cluster", None),
            "iters": rom_solution.iterations,
            "residual_norm": rom_solution.residual_norm,
            "failed_reason": "rom_solve_not_converged",
        }

    k_active = rom_res.cluster

    indicator = compute_rom_indicator(
        rom_res.w_rom,
        rom_res.callback_data,
        operators[k_active],
        M_uu_red[k_active],
        n_eigenvalues=n_eigenvalues,
        target=target,
    )

    if indicator is None:
        return {
            "Re": float(Re),
            "mu": None,
            "rom_res": rom_res,
            "converged": False,
            "cluster": int(k_active),
            "iters": rom_solution.iterations,
            "residual_norm": rom_solution.residual_norm,
            "failed_reason": "indicator_failed",
        }

    return {
        "Re": float(Re),
        "mu": float(indicator["mu"]),
        "rom_res": rom_res,
        "converged": True,
        "cluster": int(k_active),
        "iters": rom_solution.iterations,
        "residual_norm": rom_solution.residual_norm,
        "failed_reason": None,
    }


def evaluate_fom_point(
    *,
    problem,
    Re,
    amp,
    base_initial_solution,
    n_eigenvalues=6,
    target=0.0,
    use_picard=False,
):
    """
    Solve the steady Navier-Stokes problem at (Re, amp) starting from
    ``base_initial_solution`` and compute the FOM pitchfork indicator
    on success. Never raises: failures are returned as
    ``converged=False`` records with a ``failed_reason`` string.
    """
    problem.reynolds.assign(Re)
    problem.amplitude.assign(amp)

    try:
        sol = solve_steady_navier_stokes(
            problem,
            initial_guess=base_initial_solution.solution,
            use_picard=use_picard,
        )
    except Exception as e:
        return {
            "Re": float(Re),
            "mu": None,
            "solution": None,
            "converged": False,
            "iters": None,
            "failed_reason": f"solve_exception: {type(e).__name__}: {e}",
        }

    if not sol.converged:
        return {
            "Re": float(Re),
            "mu": None,
            "solution": None,
            "converged": False,
            "iters": sol.num_iterations,
            "failed_reason": "snes_not_converged",
        }

    indicator = compute_fom_indicator(
        problem,
        sol,
        n_eigenvalues=n_eigenvalues,
        target=target,
    )

    if indicator is None:
        return {
            "Re": float(Re),
            "mu": None,
            "solution": sol,
            "converged": False,
            "iters": sol.num_iterations,
            "failed_reason": "indicator_failed",
        }

    return {
        "Re": float(Re),
        "mu": float(indicator["mu"]),
        "solution": sol,
        "converged": True,
        "iters": sol.num_iterations,
        "failed_reason": None,
    }


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------

def rom_sweep(
    Re_range,
    amp_range,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    *,
    mode,
    n_eigenvalues=6,
    target=0.0,
    verbose=True,
):
    """
    Sweep over a (Re, amp) grid, solving the ROM and recording the
    pitchfork indicator at each point.

    Parameters
    ----------
    amp_range : float or sequence of float
        Scalar is accepted as shorthand for a single amplitude.

    Returns
    -------
    records : list of dict
        Flat list of all (amp, Re) records.
    by_amp : dict
        Same records grouped by amplitude.
    """
    if np.isscalar(amp_range):
        amp_values = [float(amp_range)]
    else:
        amp_values = [float(a) for a in amp_range]

    Re_values = [float(Re) for Re in Re_range]

    results = []
    by_amp = {}

    if verbose:
        print("\n" + "=" * 80)
        print("ROM SWEEP")
        print("=" * 80)
        print(f"  {'amp':>10} {'Re':>10} {'clus':>5} {'it':>4} {'mu':>14}")
        print("  " + "-" * 50)

    for amp in amp_values:
        amp_records = []

        for Re in Re_values:
            rom_res = solve_at_parameter_with_internals(
                Re=Re,
                amp=amp,
                clustering=clustering,
                operators=operators,
                deim_ops_dict=deim_ops_dict,
                problem=problem,
                initial_solution=initial_solution,
                M_u=M_u,
                M_p=M_p,
                mode= mode,
            )

            k_active = rom_res.cluster

            rom_ind = compute_rom_indicator(
                rom_res.w_rom,
                rom_res.callback_data,
                operators[k_active],
                M_uu_red[k_active],
                n_eigenvalues=n_eigenvalues,
                target=target,
            )

            if rom_ind is None:
                mu = float("nan")
                nu_red = None
                eigvals = None
            else:
                mu = rom_ind["mu"]
                nu_red = rom_ind["nu_red"]
                eigvals = rom_ind["eigvals"]

            record = {
                "amp": float(amp),
                "Re": float(Re),
                "cluster": int(k_active),
                "iters": int(rom_res.iterations),
                "mu": mu,
                "nu_red": nu_red,
                "eigvals": eigvals,
                "rom_res": rom_res,
            }

            results.append(record)
            amp_records.append(record)

            if verbose:
                print(
                    f"  {amp:>+10.4e} {Re:>10.4f} "
                    f"{k_active:>5d} {rom_res.iterations:>4d} "
                    f"{mu:>+14.6e}"
                )

        by_amp[float(amp)] = amp_records

    return results, by_amp


def fom_sweep(
    Re_range,
    amp_range,
    problem,
    initial_solution,
    *,
    n_eigenvalues=6,
    target=0.0,
    verbose=True,
):
    """
    Sweep over a (Re, amp) grid, solving the FOM Navier-Stokes problem
    and recording the pitchfork indicator at each point.

    Solver failures and indicator failures are tolerated: they appear
    as records with ``mu = nan`` and ``nconv = 0``.

    Parameters
    ----------
    amp_range : float or sequence of float
        Scalar is accepted as shorthand for a single amplitude.

    Returns
    -------
    records : list of dict
        Flat list of all (amp, Re) records.
    by_amp : dict
        Same records grouped by amplitude.
    """
    if np.isscalar(amp_range):
        amp_values = [float(amp_range)]
    else:
        amp_values = [float(a) for a in amp_range]

    Re_values = [float(Re) for Re in Re_range]

    records = []
    by_amp = {}

    if verbose:
        print("\n" + "=" * 80)
        print("FOM SWEEP")
        print("=" * 80)
        print(f"  {'amp':>10} {'Re':>10} {'nconv':>6} {'mu_FOM':>14}")
        print("  " + "-" * 46)

    for amp in amp_values:
        amp_records = []

        for Re in Re_values:
            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)

            try:
                fom_solution = solve_steady_navier_stokes(
                    problem,
                    initial_solution.solution,
                )

                indicator = compute_fom_indicator(
                    problem,
                    fom_solution,
                    n_eigenvalues=n_eigenvalues,
                    target=target,
                )

            except Exception as e:
                if verbose:
                    print(
                        f"  {amp:>+10.4e} {Re:>10.4f} "
                        f"{'FAIL':>6} {'nan':>14}  {type(e).__name__}"
                    )

                record = {
                    "amp": float(amp),
                    "Re": float(Re),
                    "mu": float("nan"),
                    "eigvals": None,
                    "eigvec": None,
                    "nconv": 0,
                    "solution": None,
                    "error": e,
                }

                records.append(record)
                amp_records.append(record)
                continue

            if indicator is None:
                mu_fom = float("nan")
                eigvals = None
                eigvec = None
                nconv = 0
            else:
                mu_fom = indicator["mu"]
                eigvals = indicator["eigvals"]
                eigvec = indicator["eigvec"]
                nconv = indicator["nconv"]

            record = {
                "amp": float(amp),
                "Re": float(Re),
                "mu": mu_fom,
                "eigvals": eigvals,
                "eigvec": eigvec,
                "nconv": nconv,
                "solution": fom_solution,
            }

            records.append(record)
            amp_records.append(record)

            if verbose:
                print(
                    f"  {amp:>+10.4e} {Re:>10.4f} "
                    f"{nconv:>6d} {mu_fom:>+14.6e}"
                )

        by_amp[float(amp)] = amp_records

    return records, by_amp


# ---------------------------------------------------------------------------
# Generic bisection on mu(Re)
# ---------------------------------------------------------------------------

def bisect_zero_crossing(
    records,
    evaluate_fn,
    *,
    mu_key="mu",
    state_key,
    Re_tol=1e-2,
    max_iter=20,
    newton_safe_mu=None,
    label="",
):
    """
    Refine a sign change in mu(Re) by bisection.

    Parameters
    ----------
    records : list of dict
        Sweep records with at least ``Re`` and ``mu_key`` entries.
    evaluate_fn : callable
        ``evaluate_fn(Re) -> dict`` returning a record with at least
        ``converged``, ``mu_key`` and ``state_key``. Should never raise;
        return ``converged=False`` instead.
    state_key : str
        Key of the state object (e.g. ``"rom_res"`` or ``"solution"``)
        to extract from the records.
    newton_safe_mu : float or None
        If not ``None``, stop early once one bracket endpoint has
        ``|mu| < newton_safe_mu`` so a Newton continuation pass can take
        over from a numerically safe distance to criticality.

    Returns
    -------
    dict or None
        ``None`` if the initial records contain no sign change.
        Otherwise a dict with the linearly extrapolated ``Re_c``, the
        nearest converged state, the bisection history, and the
        termination reason.
    """
    left, right = find_mu_bracket(records, mu_key=mu_key)

    if left is None:
        print(f"No {label} sign change found in sweep records.")
        return None

    if left is right:
        Re_c = float(left["Re"])
        mu_c = float(left[mu_key])
        state_c = left[state_key]
        print(f"{label} critical point already found in sweep: Re_c = {Re_c:.6f}")
        return {
            "Re_c": Re_c,
            "Re_state_c": Re_c,
            "mu_c": mu_c,
            "state_c": state_c,
            "history": [{"Re": Re_c, "mu": mu_c, "converged": True,
                         "source": "exact_sweep_point"}],
            "bracket": {"Re_left": Re_c, "mu_left": mu_c,
                        "Re_right": Re_c, "mu_right": mu_c},
            "terminated_reason": "exact_sweep_point",
        }

    Re_left = float(left["Re"])
    mu_left = float(left[mu_key])
    state_left = left[state_key]

    Re_right = float(right["Re"])
    mu_right = float(right[mu_key])
    state_right = right[state_key]

    history = [
        {"Re": Re_left,  "mu": mu_left,  "converged": True, "source": "initial_left"},
        {"Re": Re_right, "mu": mu_right, "converged": True, "source": "initial_right"},
    ]

    print("\n" + "=" * 80)
    print(f"{label} PITCHFORK BISECTION")
    print("=" * 80)
    print(f"Initial bracket: [{Re_left:.6f}, {Re_right:.6f}]")
    print(f"mu_left={mu_left:+.6e}, mu_right={mu_right:+.6e}")

    terminated_reason = "max_iter_reached"

    for it in range(max_iter):
        if newton_safe_mu is not None and min(abs(mu_left), abs(mu_right)) < newton_safe_mu:
            terminated_reason = "newton_safe_mu_reached"
            break

        Re_mid = 0.5 * (Re_left + Re_right)
        result_mid = evaluate_fn(Re_mid)
        history.append(result_mid)

        if not result_mid["converged"]:
            print(f"  [it {it+1:2d}] Re={Re_mid:.6f} "
                  f"{label} failed; rejecting midpoint.")
            print(f"       reason: {result_mid.get('failed_reason', 'unknown')}")
            terminated_reason = "nonconverged_midpoint_rejected"
            break

        mu_mid    = result_mid[mu_key]
        state_mid = result_mid[state_key]

        print(f"  [it {it+1:2d}] Re={Re_mid:.6f}  "
              f"mu={mu_mid:+.6e}  iters={result_mid.get('iters', '?')}")

        if mu_left * mu_mid > 0.0:
            Re_left, mu_left, state_left = Re_mid, mu_mid, state_mid
        else:
            Re_right, mu_right, state_right = Re_mid, mu_mid, state_mid

        if abs(Re_right - Re_left) < Re_tol:
            terminated_reason = "Re_tol_reached"
            break

    if abs(mu_right - mu_left) > 0.0:
        Re_c = Re_left - mu_left * (Re_right - Re_left) / (mu_right - mu_left)
    else:
        Re_c = 0.5 * (Re_left + Re_right)

    if abs(mu_left) <= abs(mu_right):
        Re_state_c, mu_c, state_c = Re_left,  mu_left,  state_left
    else:
        Re_state_c, mu_c, state_c = Re_right, mu_right, state_right

    print(f"{label} Re_c ≈ {Re_c:.6f}")
    print(f"Nearest converged state: Re={Re_state_c:.6f}, mu={mu_c:+.6e}")
    print(f"Terminated reason: {terminated_reason}")

    return {
        "Re_c": Re_c,
        "Re_state_c": Re_state_c,
        "mu_c": mu_c,
        "state_c": state_c,
        "history": history,
        "bracket": {"Re_left": Re_left, "mu_left": mu_left,
                    "Re_right": Re_right, "mu_right": mu_right},
        "terminated_reason": terminated_reason,
    }


# ---------------------------------------------------------------------------
# High-level pitchfork bisection wrappers
# ---------------------------------------------------------------------------

def bisect_pitchfork_rom_from_records(
    rom_records,
    *,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    mode,
    amp=0.0,
    Re_tol=1e-2,
    newton_safe_mu=1e-4,
    max_iter=20,
    n_eigenvalues=6,
    target=0.0,
):
    """
    Refine a ROM pitchfork on the symmetric branch given a ROM sweep.

    Wraps `bisect_zero_crossing` with `evaluate_rom_point` as the
    midpoint evaluator.
    """
    def eval_fn(Re):
        return evaluate_rom_point(
            Re=Re, amp=amp,
            clustering=clustering, operators=operators,
            deim_ops_dict=deim_ops_dict, problem=problem,
            initial_solution=initial_solution,
            M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
            mode=mode,
            n_eigenvalues=n_eigenvalues, target=target,
        )

    return bisect_zero_crossing(
        rom_records, eval_fn,
        mu_key="mu", state_key="rom_res",
        Re_tol=Re_tol, max_iter=max_iter,
        newton_safe_mu=newton_safe_mu,
        label="ROM",
    )


def bisect_pitchfork_fom_from_records(
    fom_records,
    *,
    problem,
    initial_solution,
    amp=0.0,
    Re_tol=1e-2,
    max_iter=20,
    n_eigenvalues=6,
    target=0.0,
):
    """
    Refine a FOM pitchfork on the symmetric branch given a FOM sweep.

    Wraps `bisect_zero_crossing` with `evaluate_fom_point` as the
    midpoint evaluator.
    """
    def eval_fn(Re):
        return evaluate_fom_point(
            problem=problem, Re=Re, amp=amp,
            base_initial_solution=initial_solution,
            n_eigenvalues=n_eigenvalues, target=target,
        )

    return bisect_zero_crossing(
        fom_records, eval_fn,
        mu_key="mu", state_key="solution",
        Re_tol=Re_tol, max_iter=max_iter,
        newton_safe_mu=None,
        label="FOM",
    )
def find_critical_curve(
    amp_values,
    Re_scan,
    *,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    mode,
    Re_tol=1e-2,
    max_iter=20,
    newton_safe_mu=None,
    n_eigenvalues=6,
    target=0.0,
    verbose=True,
):
    """
    Reduced critical curve Re_c(amp) across the (Re, amp) domain.

    For each amp:
      1. Coarse-sweep the ROM pitchfork indicator mu(Re) on the PRIMARY
         branch. Every solve starts from the symmetric `initial_solution`,
         so the cluster selector lands on the symmetric/primary cluster and
         mu is the primary branch's stability eigenvalue.
      2. Bracket the mu sign change and bisect to Re_c (reusing
         bisect_pitchfork_rom_from_records).

    Only evaluates mu near its zero-crossing, where shift-invert
    (TARGET_REAL at sigma=0) is reliable -- so the mu=nan seen far out on
    developed asymmetric branches never arises here. Re_c(amp) is the
    primary branch's stability crossing; the asymmetric branches merge into
    it, they don't define it.

    Parameters
    ----------
    amp_values : sequence of amp values (any sign) to locate Re_c at.
    Re_scan    : ascending coarse Re grid used to bracket the crossing;
                 must straddle Re_c for every amp in amp_values.

    Returns
    -------
    curve : list of dict (sorted by amp), each with keys
        'amp', 'Re_c' (None if not found), 'mu_c', 'found' (bool),
        'sweep' (the coarse mu(Re) records for inspection/plotting).
    """
    # walk amps outward from 0 so the crossing moves smoothly between solves
    amps_sorted = sorted(amp_values, key=lambda a: abs(a))

    curve = []
    for amp in amps_sorted:
        if verbose:
            print("\n" + "=" * 80)
            print(f"CRITICAL CURVE  amp = {amp:+.4f}")
            print("=" * 80)

        # 1. coarse mu(Re) sweep at this amp (primary branch)
        records = []
        for Re in Re_scan:
            rec = evaluate_rom_point(
                Re=Re, amp=amp,
                clustering=clustering, operators=operators,
                deim_ops_dict=deim_ops_dict, problem=problem,
                initial_solution=initial_solution,
                M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
                mode=mode,
                n_eigenvalues=n_eigenvalues, target=target,
            )
            records.append(rec)
            if verbose:
                mu_str = f"{rec['mu']:+.6e}" if rec['mu'] is not None else "   nan    "
                print(f"  Re={Re:7.3f}  mu={mu_str}  "
                      f"k={rec['cluster']}  conv={rec['converged']}")

        # 2. bracket + bisect to Re_c (returns None if no sign change in records)
        result = bisect_pitchfork_rom_from_records(
            records,
            clustering=clustering, operators=operators,
            deim_ops_dict=deim_ops_dict, problem=problem,
            initial_solution=initial_solution,
            M_u=M_u, M_p=M_p, M_uu_red=M_uu_red, mode = mode,
            amp=amp,
            Re_tol=Re_tol, max_iter=max_iter,
            newton_safe_mu=newton_safe_mu,
            n_eigenvalues=n_eigenvalues, target=target,
        )

        if result is None:
            curve.append({'amp': float(amp), 'Re_c': None, 'mu_c': None,
                          'found': False, 'sweep': records})
            if verbose:
                print(f"  -> no mu sign change in Re_scan for amp={amp:+.4f}")
        else:
            curve.append({'amp': float(amp), 'Re_c': float(result['Re_c']),
                          'mu_c': float(result['mu_c']), 'found': True,
                          'sweep': records})
            if verbose:
                print(f"  -> Re_c({amp:+.4f}) = {result['Re_c']:.4f}")

    curve = sorted(curve, key=lambda c: c['amp'])
    if verbose:
        print("\n" + "=" * 80)
        print("REDUCED CRITICAL CURVE  Re_c(amp)")
        print("=" * 80)
        for c in curve:
            tag = f"{c['Re_c']:.4f}" if c['found'] else "(not found)"
            print(f"  amp={c['amp']:+.4f}   Re_c={tag}")

    return curve

def save_critical_curve(curve, path, *, sweep_path=None):
    """
    Write the reduced critical curve to CSV.

    curve : list of dict from find_critical_curve[_continuation]; each has
        'amp', 'Re_c', 'mu_c', 'found', optional 'kind', and 'sweep'.
    path : curve CSV, one row per amp:
        amp, Re_c, mu_c, found, kind
        (Re_c/mu_c written as 'nan' where not found; kind '' if absent.)
    sweep_path : optional; all coarse mu(Re) step points across amps:
        amp, Re, mu, cluster, converged, iters
    """
    import csv

    rows = sorted(curve, key=lambda c: c['amp'])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["amp", "Re_c", "mu_c", "found", "kind"])
        for c in rows:
            re_c = c["Re_c"] if c["Re_c"] is not None else float("nan")
            mu_c = c["mu_c"] if c["mu_c"] is not None else float("nan")
            w.writerow([f"{c['amp']:.6f}", f"{re_c:.6f}", f"{mu_c:.6e}",
                        bool(c["found"]), c.get("kind") or ""])
    print(f"Critical curve saved to {path}  ({len(rows)} amps)")

    if sweep_path is not None:
        n = 0
        with open(sweep_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["amp", "Re", "mu", "cluster", "converged", "iters"])
            for c in rows:
                for rec in c.get("sweep", []):
                    mu  = rec["mu"] if rec.get("mu") is not None else float("nan")
                    clu = rec["cluster"] if rec.get("cluster") is not None else ""
                    itr = rec.get("iters")
                    w.writerow([f"{c['amp']:.6f}", f"{rec['Re']:.6f}",
                                f"{mu:.6e}", clu, bool(rec["converged"]),
                                "" if itr is None else itr])
                    n += 1
        print(f"Sweep points saved to {sweep_path}  ({n} rows)")
