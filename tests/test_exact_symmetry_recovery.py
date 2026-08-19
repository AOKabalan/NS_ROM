from dataclasses import FrozenInstanceError
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from nsrom.bifurcation import diagram
from nsrom.bifurcation.detection import (
    classify_symmetry_recovery,
    normalized_h1_distance,
)


class _Dat:
    def __init__(self, values):
        self.data_ro = np.asarray(values, dtype=float)


class _Field:
    def __init__(self, values):
        self.dat = _Dat(values)


class _Assignable:
    def __init__(self):
        self.value = None

    def assign(self, value):
        self.value = value


class _Writer:
    def __init__(self):
        self.calls = []

    def write(self, **kwargs):
        self.calls.append(kwargs)


def _solution(values, *, converged=True, lift=0.1, coefficient=1.0):
    return SimpleNamespace(
        converged=converged,
        iterations=2,
        velocity_coeffs=np.array([coefficient]),
        pressure_coeffs=np.array([0.0]),
        full_velocity=np.asarray(values, dtype=float),
        lift=float(lift),
    )


def _run_continue(monkeypatch, solutions, *, path, branch_id="asym+1@amp+0.10",
                  references=None, terminate=True, tolerance=1e-5):
    pending = list(solutions)
    solve_guesses = []
    force_calls = []

    def solve_rom(**kwargs):
        solve_guesses.append(np.array(kwargs["u_initial_guess"], copy=True))
        return pending.pop(0), object(), {}

    def reconstruct_solution(solution, operators, problem, amp):
        return SimpleNamespace(
            velocity=_Field(solution.full_velocity),
            pressure=_Field([0.0]),
            lift=solution.lift,
        )

    def compute_forces(solution_rom, problem):
        force_calls.append(solution_rom)
        return SimpleNamespace(lift=solution_rom.lift, drag=1.0)

    monkeypatch.setattr(diagram, "select_cluster", lambda *args, **kwargs: 0)
    monkeypatch.setattr(diagram, "select_nearest_cluster", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        diagram,
        "project_to_rom_coefficients",
        lambda *args, **kwargs: (np.array([0.0]), np.array([0.0])),
    )
    monkeypatch.setattr(diagram, "solve_rom", solve_rom)
    monkeypatch.setattr(diagram, "reconstruct_solution", reconstruct_solution)
    monkeypatch.setattr(diagram, "compute_rom_indicator", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagram, "compute_forces", compute_forces)

    problem = SimpleNamespace(reynolds=_Assignable(), amplitude=_Assignable())
    initial_solution = SimpleNamespace(
        velocity=_Field([0.0, 0.0]),
        pressure=_Field([0.0]),
    )
    clustering = SimpleNamespace(change_of_basis=None)
    operators = {0: SimpleNamespace(
        n_lifting=2,
        lifting_dofs=np.zeros((2, 2)),
        Z_u=np.eye(2),
    )}
    writer = _Writer()

    compact_references = references
    if references is not None:
        compact_references = {}
        for key, values in references.items():
            coefficients = np.array(values, dtype=float, copy=True)
            coefficients.setflags(write=False)
            compact_references[key] = diagram._SymmetricReference(
                cluster=0,
                amplitude=key[1],
                velocity_coeffs=coefficients,
            )

    result = diagram._continue(
        path,
        clustering,
        operators,
        {0: None},
        {},
        problem,
        initial_solution,
        csr_matrix(np.eye(2)),
        csr_matrix(np.eye(1)),
        {0: np.eye(1)},
        mode="tensor",
        can_bifurcate=False,
        branch_id=branch_id,
        terminate_on_collapse=terminate,
        symmetric_references=compact_references,
        symmetry_recovery_tol=tolerance,
        state_writer=writer,
        verbose=False,
    )
    return result, writer, solve_guesses, force_calls


def test_identical_states_recover_with_zero_distance():
    state = np.array([1.0, -2.0, 3.0])
    recovered, distance = classify_symmetry_recovery(
        state, state.copy(), csr_matrix(np.eye(3)), 1e-5, converged=True)
    assert recovered
    assert distance == 0.0


def test_clearly_separated_states_remain_asymmetric():
    recovered, distance = classify_symmetry_recovery(
        [1.0, 1.0], [1.0, 0.0], csr_matrix(np.eye(2)), 1e-5,
        converged=True)
    assert not recovered
    assert distance == pytest.approx(1.0)


def test_tolerance_boundary_is_inclusive():
    tolerance = 1e-5
    recovered, distance = classify_symmetry_recovery(
        [1.0, tolerance], [1.0, 0.0], csr_matrix(np.eye(2)), tolerance,
        converged=True)
    assert distance == pytest.approx(tolerance)
    assert recovered


def test_distance_just_above_tolerance_is_not_recovered():
    tolerance = 1e-5
    recovered, distance = classify_symmetry_recovery(
        [1.0, tolerance * (1.0 + 1e-8)],
        [1.0, 0.0],
        csr_matrix(np.eye(2)),
        tolerance,
        converged=True,
    )
    assert distance > tolerance
    assert not recovered


def test_missing_reference_and_failed_solve_do_not_classify():
    missing, missing_distance = classify_symmetry_recovery(
        [1.0], None, csr_matrix([[1.0]]), 1e-5, converged=True)
    failed, failed_distance = classify_symmetry_recovery(
        [1.0], [1.0], csr_matrix([[1.0]]), 1e-5, converged=False)
    assert not missing and np.isnan(missing_distance)
    assert not failed and np.isnan(failed_distance)


@pytest.mark.parametrize("bad_state", ([np.nan, 0.0], [np.inf, 0.0]))
def test_nonfinite_states_do_not_classify(bad_state):
    recovered, distance = classify_symmetry_recovery(
        bad_state, [1.0, 0.0], csr_matrix(np.eye(2)), 1e-5,
        converged=True)
    assert not recovered
    assert np.isnan(distance)


def test_sparse_h1_mass_matrix_is_applied_without_densifying():
    mass = diags([2.0, 8.0], format="csr")
    distance = normalized_h1_distance([2.0, 1.0], [2.0, 0.0], mass)
    assert distance == pytest.approx(1.0)


@pytest.mark.parametrize("branch_id", ["asym+1@amp+0.10", "asym-1@amp+0.10"])
def test_asymmetric_merger_stops_before_all_mutations(monkeypatch, branch_id):
    tolerance = 2e-5
    seen_tolerances = []
    real_classifier = diagram.classify_symmetry_recovery

    def recording_classifier(*args, **kwargs):
        seen_tolerances.append(args[3])
        return real_classifier(*args, **kwargs)

    monkeypatch.setattr(diagram, "classify_symmetry_recovery", recording_classifier)
    result, writer, guesses, force_calls = _run_continue(
        monkeypatch,
        [_solution([1.0, 0.0], coefficient=7.0)],
        path=[(70.0, 0.1), (69.0, 0.1)],
        branch_id=branch_id,
        references={(70.0, 0.1): np.array([1.0, 0.0])},
        tolerance=tolerance,
    )
    records, spawns, last_converged, converged_states = result
    assert records == []
    assert spawns == []
    assert last_converged is None
    assert converged_states == []
    assert writer.calls == []
    assert len(guesses) == 1
    assert force_calls == []
    assert seen_tolerances == [tolerance]


def test_genuine_low_lift_state_is_retained_with_exact_reference(monkeypatch):
    result, writer, guesses, force_calls = _run_continue(
        monkeypatch,
        [
            _solution([1.0, 1.0], lift=0.1, coefficient=3.0),
            _solution([1.0, 1.0], lift=8e-4, coefficient=4.0),
        ],
        path=[(90.0, -0.4), (89.0, -0.4)],
        branch_id="asym+1@amp-0.40",
        references={
            (90.0, -0.4): np.array([1.0, 0.0]),
            (89.0, -0.4): np.array([1.0, 0.0]),
        },
    )
    records, _, last_converged, converged_states = result
    assert len(records) == 2
    assert records[1]["C_L"] == pytest.approx(8e-4)
    assert records[1]["symmetry_distance_H1"] == pytest.approx(1.0)
    assert last_converged["u_coeffs"] == pytest.approx([4.0])
    assert len(converged_states) == 2
    assert len(writer.calls) == 2
    assert len(guesses) == 2
    assert len(force_calls) == 2


def test_recovered_state_preserves_previous_asymmetric_anchor(monkeypatch):
    result, writer, guesses, _ = _run_continue(
        monkeypatch,
        [
            _solution([1.0, 1.0], coefficient=3.0),
            _solution([1.0, 0.0], coefficient=8.0),
            _solution([2.0, 0.0], coefficient=9.0),
        ],
        path=[(72.0, 0.1), (71.0, 0.1), (70.0, 0.1)],
        references={
            (72.0, 0.1): np.array([1.0, 0.0]),
            (71.0, 0.1): np.array([1.0, 0.0]),
            (70.0, 0.1): np.array([2.0, 0.0]),
        },
    )
    records, _, last_converged, converged_states = result
    assert len(records) == 1
    assert len(writer.calls) == 1
    assert len(guesses) == 2
    assert guesses[1] == pytest.approx([3.0])
    assert last_converged["u_coeffs"] == pytest.approx([3.0])
    assert len(converged_states) == 1


def test_symmetric_branch_never_uses_recovery_stop(monkeypatch):
    result, writer, _, _ = _run_continue(
        monkeypatch,
        [_solution([1.0, 0.0])],
        path=[(70.0, 0.1)],
        branch_id="sym@amp+0.10",
        references={(70.0, 0.1): np.array([1.0, 0.0])},
        terminate=False,
    )
    records, _, last_converged, converged_states = result
    assert len(records) == 1
    assert last_converged is not None
    assert len(converged_states) == 1
    assert len(writer.calls) == 1


def test_missing_exact_reference_raises_without_lift_fallback(monkeypatch):
    with pytest.raises(RuntimeError, match="missing exact symmetric reference"):
        _run_continue(
            monkeypatch,
            [_solution([1.0, 0.0], lift=0.0)],
            path=[(70.0, 0.1)],
            references={},
        )


def test_nonconverged_merge_stop_breaks_before_persistence(monkeypatch):
    result, writer, guesses, _ = _run_continue(
        monkeypatch,
        [
            _solution([1.0, 1.0], lift=0.1, coefficient=3.0),
            _solution([9.0, 9.0], converged=False, coefficient=8.0),
            _solution([2.0, 2.0], coefficient=9.0),
        ],
        path=[(72.0, 0.1), (71.0, 0.1), (70.0, 0.1)],
        references={
            (72.0, 0.1): np.array([1.0, 0.0]),
            (71.0, 0.1): np.array([1.0, 0.0]),
            (70.0, 0.1): np.array([1.0, 0.0]),
        },
    )
    records, _, last_converged, converged_states = result
    assert len(records) == 1
    assert len(writer.calls) == 1
    assert len(guesses) == 2
    assert guesses[1] == pytest.approx([3.0])
    assert last_converged["u_coeffs"] == pytest.approx([3.0])
    assert len(converged_states) == 1


def test_reference_free_path_retains_legacy_lift_stop(monkeypatch):
    result, writer, guesses, _ = _run_continue(
        monkeypatch,
        [
            _solution([1.0, 1.0], lift=0.1, coefficient=3.0),
            _solution([1.0, 0.5], lift=8e-4, coefficient=4.0),
            _solution([1.0, 0.0], coefficient=5.0),
        ],
        path=[(72.0, 0.1), (71.0, 0.1), (70.0, 0.1)],
        references=None,
    )
    records, _, last_converged, converged_states = result
    assert len(records) == 1
    assert len(writer.calls) == 1
    assert len(guesses) == 2
    assert last_converged["u_coeffs"] == pytest.approx([3.0])
    assert len(converged_states) == 1


def test_reference_coverage_is_checked_before_asymmetric_tracing(monkeypatch):
    anchor = {"Re": 21.0}
    calls = []
    monkeypatch.setattr(
        diagram,
        "_three_branches",
        lambda *args, **kwargs: ([], [("asym+1", anchor)], [anchor]),
    )

    def offaxis(anchor, base_id, *args, **kwargs):
        calls.append(base_id)
        return []

    monkeypatch.setattr(diagram, "_offaxis_grid", offaxis)
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        diagram.build_full_diagram_bare(
            [20.0, 21.0],
            [0.1],
            SimpleNamespace(n_clusters=1),
            {0: SimpleNamespace()},
            {0: None},
            SimpleNamespace(),
            SimpleNamespace(),
            csr_matrix(np.eye(2)),
            csr_matrix(np.eye(1)),
            {0: np.eye(1)},
            mode="tensor",
            trace_symmetric=True,
            verbose=False,
        )
    assert calls == ["sym"]


def test_stored_symmetric_reference_is_compact_and_immutable():
    source = np.array([1.0, 2.0])
    solution = SimpleNamespace(velocity_coeffs=source)
    references = {}
    diagram._store_symmetric_reference(references, 70.0, 0.1, solution, 3)
    source[:] = 9.0
    stored = references[(70.0, 0.1)]
    assert stored.cluster == 3
    assert stored.amplitude == 0.1
    assert stored.velocity_coeffs == pytest.approx([1.0, 2.0])
    assert not stored.velocity_coeffs.flags.writeable
    with pytest.raises(FrozenInstanceError):
        stored.cluster = 1


def test_compact_reference_reconstructs_lifting_and_reduced_state():
    coefficients = np.array([2.0, 5.0])
    coefficients.setflags(write=False)
    reference = diagram._SymmetricReference(0, 0.5, coefficients)
    operator = SimpleNamespace(
        n_lifting=2,
        lifting_dofs=np.array([[10.0, 20.0], [3.0, 4.0]]),
        Z_u=np.array([[1.0, 0.0], [0.0, 2.0]]),
    )
    velocity = diagram._reconstruct_symmetric_reference(
        reference, {0: operator}, Re=70.0, amp=0.5)
    assert velocity == pytest.approx([13.5, 32.0])


def test_production_diagram_default_uses_low_symmetric_spine():
    default = inspect.signature(
        diagram.build_full_diagram_bare).parameters["sym_start"].default
    assert default == "low"
