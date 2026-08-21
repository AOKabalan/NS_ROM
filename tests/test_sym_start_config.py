"""Reproducibility guards for the ``sym_start`` experiment parameter and the
fallback amplitude grid.

Context: ``build_full_diagram_bare``'s library default for ``sym_start`` was
flipped ``'high' -> 'low'`` in a later commit, but every canonical published
run in ``states/*/run_meta.json`` was generated with ``'high'``. The pipeline
never passed the value explicitly, so re-running the canonical experiments
would have silently regenerated them on the *opposite* symmetric spine.

These tests pin the fix in place:

  * ``local_pipeline`` reads ``NSROM_SYM_START`` and threads it explicitly into
    the diagram builder (so the value is never inherited from a mutable Python
    default), and
  * every canonical ``E*`` experiment in ``run.sh`` pins ``NSROM_SYM_START``
    explicitly, while replay-only ``R*`` runs (which do not build a diagram)
    do not.

They also guard the fallback amplitude grid against re-omitting ``a = 0.0``.

Everything here is pure text / AST inspection: no Firedrake, no numerics, no
state regeneration. That keeps the guard cheap and makes it hard for a future
edit to quietly remove the plumbing.
"""
import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
PIPELINE = REPO / "nsrom" / "workflows" / "local_pipeline.py"
RUN_SH = REPO / "run.sh"


def _pipeline_tree():
    return ast.parse(PIPELINE.read_text())


def _find_assign(tree, target_name):
    """Return the value node assigned to ``target_name`` at module scope."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == target_name:
                    return node.value
    return None


def _find_call(tree, func_name):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == func_name):
            return node
    return None


# --- sym_start plumbing in local_pipeline.py --------------------------------

def test_pipeline_reads_sym_start_from_env():
    """DIAGRAM_SYM_START must come from the NSROM_SYM_START env var."""
    value = _find_assign(_pipeline_tree(), "DIAGRAM_SYM_START")
    assert value is not None, "DIAGRAM_SYM_START assignment vanished"
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name) and value.func.id == "_env_str"
    assert value.args, "_env_str called with no arguments"
    env_name = value.args[0]
    assert isinstance(env_name, ast.Constant) and env_name.value == "NSROM_SYM_START"


def test_pipeline_threads_sym_start_into_diagram_builder():
    """The diagram call must forward sym_start=DIAGRAM_SYM_START explicitly.

    This is the anti-regression: it fails if the keyword is dropped (which would
    silently fall back to diagram.py's library default) or wired to a literal
    instead of the env-derived variable.
    """
    call = _find_call(_pipeline_tree(), "build_full_diagram_bare")
    assert call is not None, "build_full_diagram_bare is no longer called"
    kw = {k.arg: k.value for k in call.keywords}
    assert "sym_start" in kw, "sym_start no longer passed to build_full_diagram_bare"
    passed = kw["sym_start"]
    assert isinstance(passed, ast.Name) and passed.id == "DIAGRAM_SYM_START", (
        "sym_start must forward the env-derived DIAGRAM_SYM_START, "
        "not a hard-coded literal"
    )


# --- fallback amplitude grid ------------------------------------------------

def test_fallback_amp_grid_contains_exactly_one_zero():
    """A bare/manual run must use the symmetric grid that includes a = 0.0,
    not the retired grid that omitted it."""
    value = _find_assign(_pipeline_tree(), "DIAGRAM_AMP_VALUES")
    assert isinstance(value, ast.Call) and value.func.id == "_env_floats"
    default = value.args[1]
    assert isinstance(default, ast.List), "fallback amp grid is not a list literal"
    amps = ast.literal_eval(default)  # handles negative (UnaryOp) literals
    assert amps.count(0.0) == 1, f"expected exactly one 0.0 in fallback grid, got {amps}"
    # the grid should stay symmetric about zero
    assert sorted(amps) == sorted(-a for a in amps), "fallback amp grid is not symmetric"


# --- run.sh explicit pinning ------------------------------------------------

def _run_sh_define_blocks():
    """Split run.sh into (tag, block-text) pairs, one per ``define`` invocation.

    A block runs from a ``define ...`` line to the next blank line. Handles the
    ``E7`` loop form ``define "E7_..._${_tol}"`` too.
    """
    blocks = []
    cur, tag = None, None
    for line in RUN_SH.read_text().splitlines():
        if line.strip().startswith("define "):
            m = re.search(r'define\s+"?([A-Za-z0-9_${}]+)"?', line)
            tag = m.group(1) if m else "?"
            cur = [line]
        elif cur is not None:
            if line.strip() == "":
                blocks.append((tag, "\n".join(cur)))
                cur, tag = None, None
            else:
                cur.append(line)
    if cur is not None:
        blocks.append((tag, "\n".join(cur)))
    return blocks


def test_run_sh_pins_sym_start_for_every_canonical_experiment():
    """Every E* experiment (all build a diagram) pins NSROM_SYM_START='high';
    replay-only R* runs (no diagram) do not carry it.

    Convention guarded here: E* = diagram-building experiment, R* = replay.
    """
    blocks = _run_sh_define_blocks()
    e_tags = [(t, b) for t, b in blocks if t.startswith("E")]
    r_tags = [(t, b) for t, b in blocks if t.startswith("R")]
    assert e_tags, "no E* defines found -- run.sh parsing broke"

    for tag, block in e_tags:
        assert 'NSROM_SYM_START="high"' in block, (
            f"canonical experiment {tag} does not pin NSROM_SYM_START; it would "
            f"fall back to the mutable library default"
        )
    for tag, block in r_tags:
        assert "NSROM_SYM_START" not in block, (
            f"replay run {tag} sets NSROM_SYM_START but does not build a diagram"
        )
