"""
Capture golden values for the characterization suite.

Usage
-----
    python tests/generate_golden.py            # run everything, PRINT ONLY
    python tests/generate_golden.py --write    # also write tests/golden/*.json
    python tests/generate_golden.py --only A1,A4,B2
    python tests/generate_golden.py --tier A

The first (review) run is invoked WITHOUT --write: it prints every captured
value so the magnitudes can be sanity-checked before any golden is frozen.

Run from the repository root with the Firedrake venv active. The Firedrake
import happens once here (slow — minutes) and is shared across all cases.
"""
import argparse
import json
import os
import sys
import time
import traceback

# Ensure repo root is importable (so `import tests.characterization` works when
# invoked as `python tests/generate_golden.py` from the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, list):
        if v and isinstance(v[0], float):
            return "[" + ", ".join(f"{x:.6g}" for x in v) + "]"
        return str(v)
    return str(v)


def _print_case(name, captured, dt, status="ok", err=None):
    print("\n" + "-" * 72)
    print(f"[{name}]  status={status}  ({dt:.2f}s)")
    print("-" * 72)
    if err is not None:
        print(f"  !! {err}")
        for line in traceback.format_exc().splitlines():
            print("     " + line)
        return
    for k, v in captured.items():
        print(f"  {k:32s} = {_fmt(v)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write tests/golden/*.json (default: print only)")
    ap.add_argument("--only", default=None, help="comma list of case ids")
    ap.add_argument("--tier", default=None, choices=["A", "B"])
    args = ap.parse_args()

    print("=" * 72)
    print("CHARACTERIZATION GOLDEN CAPTURE")
    print(f"  time      : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  write     : {args.write}")
    print(f"  python    : {sys.version.split()[0]}")
    print("=" * 72)

    print("\nImporting nsrom (triggers Firedrake — this can take minutes)...")
    t0 = time.time()
    from tests import characterization as ch
    print(f"  import done in {time.time() - t0:.1f}s")

    # Select cases.
    cases = ch.TIER_A + ch.TIER_B
    if args.tier == "A":
        cases = ch.TIER_A
    elif args.tier == "B":
        cases = ch.TIER_B
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c in wanted]

    # Shared data.
    print("\nLoading shared data...")
    snap = ch.load_stratified_snapshots()
    print(f"  stratified snapshots: n_sel={snap['n_sel']}, n_dofs={snap['n_dofs']}")
    print(f"  branch composition (branch_id -> count): {snap['composition']}")
    print("  per-branch (Re, amp) coverage in the selection:")
    for bid, r in snap["per_branch_ranges"].items():
        print(f"    branch {bid}: n={r['n']}  Re[{r['Re'][0]:.3g}, {r['Re'][1]:.3g}]  "
              f"amp[{r['amp'][0]:.3g}, {r['amp'][1]:.3g}]")
    print(f"  subsample branch_ids (block order): {list(map(int, snap['branch_ids']))}")
    print(f"  held-out snapshot: idx={snap['holdout_idx']}, "
          f"(Re,amp)={snap['holdout_param']}")
    S = snap["S"]
    M_u = ch.load_Mu()
    print(f"  M_u: shape={M_u.shape}, nnz={M_u.nnz}")

    # Lazily-built Firedrake objects, shared across B-cases.
    _problem = {"obj": None}

    def problem():
        if _problem["obj"] is None:
            print("\n  [building Firedrake problem: mesh + spaces]")
            _problem["obj"] = ch.build_problem()
        return _problem["obj"]

    dispatch = {
        "A1": lambda: ch.case_A1_pod_euclidean(S),
        "A2": lambda: ch.case_A2_pod_energy(S, M_u),
        "A3": lambda: ch.case_A3_pod_energy_truncation(S, M_u),
        "A4": lambda: ch.case_A4_deim(S, snap["holdout"]),
        "A5": lambda: ch.case_A5_clustering_cholesky(S, M_u, snap["branch_ids"]),
        "A6": lambda: ch.case_A6_clustering_whitened(S, M_u),
        "A7": lambda: ch.case_A7_transform_roundtrip(S, M_u),
        "B1": lambda: ch.case_B1_lifting(problem()),
        "B2": lambda: ch.case_B2_rom_solve(problem()),
        "B3": lambda: ch.case_B3_eigen(problem()),
    }

    results = {}
    failures = []
    for name in cases:
        t = time.time()
        try:
            captured = dispatch[name]()
            dt = time.time() - t
            results[name] = captured
            _print_case(name, captured, dt, status="ok")
        except Exception as e:  # isolate; one failure must not sink the rest
            dt = time.time() - t
            nonblock = name in ch.NONBLOCKING
            _print_case(name, None, dt,
                        status="NONBLOCKING-FAIL" if nonblock else "FAIL",
                        err=f"{type(e).__name__}: {e}")
            if not nonblock:
                failures.append(name)

    # Summary.
    print("\n" + "=" * 72)
    print(f"CAPTURED {len(results)}/{len(cases)} cases; "
          f"blocking failures: {failures or 'none'}")
    print("=" * 72)

    if args.write:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        for name, captured in results.items():
            path = os.path.join(GOLDEN_DIR, f"{name}.json")
            with open(path, "w") as f:
                json.dump(captured, f, indent=2, sort_keys=True)
            print(f"  wrote {path}")
    else:
        print("\n(--write not passed: NO golden files written — review only)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
