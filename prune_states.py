#!/usr/bin/env python
"""
prune_states.py -- retract points from state stores without touching the data.

Two retractions, both decided after the runs finished:

  1. amp = -0.25 was swept by mistake. There is no +0.25, so the amplitude grid
     is asymmetric: 11 negative columns against 10 positive. Every point at
     amp = -0.25, on every branch, is dropped.

  2. At amp = +0.8, +0.9, +1.0 the pitchfork sits above Re_max, so the "asym"
     legs there are not a second solution -- they are the symmetric one under a
     different label. Only the rows whose branch starts with 'asym' are dropped
     at those amplitudes; the symmetric column stays.

How it works
------------
`StateStore` reads index.jsonl and resolves arrays through rec['idx'], so a row
removed from the index is invisible to every downstream consumer while its
points/NNNNNN.npz stays on disk under its original number. Nothing is
renumbered, nothing is recomputed, and --restore puts the row back.

  states/<TAG>/
      index.jsonl                     rewritten, surviving lines verbatim
      index.jsonl.prebak_<stamp>      the file as it was
      prune_log.json                  what was dropped and why
      points/NNNNNN.npz               untouched (unless --purge-npz)

Usage
-----
    python prune_states.py --dry-run              # report only, change nothing
    python prune_states.py                        # apply to states/*
    python prune_states.py --runs 'states/E*'     # subset
    python prune_states.py --purge-npz            # also delete orphaned arrays
    python prune_states.py --restore              # undo the most recent prune

Safety
------
* Refuses to drop the asym legs at an amplitude where the run has no symmetric
  rows to fall back on -- that would delete the whole column. --force overrides.
* Reports, per amplitude, how far the asym rows actually sit from the symmetric
  ones in C_L. If they are genuinely duplicates that number is at the solver
  tolerance; if it is not, --dup-tol makes the script refuse.
"""

import argparse
import glob
import json
import os
import shutil
import time

AMP_TOL = 1e-8

DEFAULT_DROP_ALL = [-0.25]              # every branch at these amplitudes
DEFAULT_DROP_ASYM = [0.8, 0.9, 1.0]     # only asym* branches at these


def _near(a, values, tol=AMP_TOL):
    return any(abs(float(a) - v) < tol for v in values)


def _fmt(a):
    return f"{a:+.2f}"


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def classify(rec, drop_all, drop_asym):
    """Return a reason string if this record should be retracted, else None."""
    amp = float(rec.get('amp'))
    branch = str(rec.get('branch', ''))
    if _near(amp, drop_all):
        return 'amp_retracted'
    if _near(amp, drop_asym) and branch.startswith('asym'):
        return 'asym_duplicate'
    return None


def duplicate_report(records, drop_asym):
    """For each amplitude in drop_asym, compare the asym rows against the
    symmetric rows at the same Re. Returns {amp: dict}."""
    out = {}
    for a in drop_asym:
        sym, asym = {}, []
        for r in records:
            if not _near(r.get('amp', 1e9), [a]):
                continue
            b = str(r.get('branch', ''))
            cl = r.get('C_L_rom', r.get('C_L'))
            if b.startswith('asym'):
                asym.append((round(float(r['Re']), 6), cl, r.get('converged')))
            elif b.startswith('sym'):
                sym.setdefault(round(float(r['Re']), 6), cl)
        if not asym:
            continue
        diffs = [abs(cl - sym[re]) for re, cl, _ in asym
                 if cl is not None and re in sym and sym[re] is not None]
        cls = [abs(cl) for _, cl, _ in asym if cl is not None]
        out[a] = {
            'n_asym': len(asym),
            'n_sym': len(sym),
            'n_paired': len(diffs),
            'max_dCL': max(diffs) if diffs else None,
            'med_dCL': sorted(diffs)[len(diffs) // 2] if diffs else None,
            'max_absCL_asym': max(cls) if cls else None,
        }
    return out


# ---------------------------------------------------------------------------
# per-run
# ---------------------------------------------------------------------------
def prune_run(state_dir, args, stamp):
    index_path = os.path.join(state_dir, 'index.jsonl')
    if not os.path.isfile(index_path):
        return None

    with open(index_path) as f:
        lines = [ln for ln in f if ln.strip()]
    records = [json.loads(ln) for ln in lines]
    if not records:
        return None

    amps = sorted({round(float(r['amp']), 6) for r in records})

    dup = duplicate_report(records, args.drop_asym)

    print(f"\n{state_dir}")
    print(f"  {len(records)} rows, {len(amps)} amplitudes")

    # --- the safety check: never delete a column outright -------------------
    blocked = []
    for a, d in sorted(dup.items()):
        line = (f"  a={_fmt(a)}  asym={d['n_asym']:4d}  sym={d['n_sym']:4d}  "
                f"paired={d['n_paired']:4d}")
        if d['max_dCL'] is not None:
            line += (f"  |dC_L| med={d['med_dCL']:.2e} max={d['max_dCL']:.2e}"
                     f"  max|C_L_asym|={d['max_absCL_asym']:.2e}")
        print(line)
        if d['n_sym'] == 0:
            blocked.append((a, 'no symmetric rows at this amplitude'))
        elif (args.dup_tol is not None and d['max_dCL'] is not None
              and d['max_dCL'] > args.dup_tol):
            blocked.append((a, f"max |dC_L| {d['max_dCL']:.2e} > dup_tol "
                               f"{args.dup_tol:.2e} -- not a duplicate"))

    drop_asym = list(args.drop_asym)
    if blocked and not args.force:
        for a, why in blocked:
            print(f"  BLOCKED at a={_fmt(a)}: {why}  (--force to override)")
        drop_asym = [a for a in drop_asym
                     if not _near(a, [b for b, _ in blocked])]

    keep_lines, dropped = [], []
    for ln, rec in zip(lines, records):
        reason = classify(rec, args.drop_all, drop_asym)
        if reason is None:
            keep_lines.append(ln)
        else:
            dropped.append({'idx': rec['idx'], 'Re': rec['Re'], 'amp': rec['amp'],
                            'branch': rec['branch'], 'reason': reason})

    by_reason = {}
    for d in dropped:
        by_reason[d['reason']] = by_reason.get(d['reason'], 0) + 1
    for reason, n in sorted(by_reason.items()):
        print(f"  drop {n:5d}  {reason}")
    if not dropped:
        print("  nothing to drop (already pruned, or grid never contained these)")
        return {'state_dir': state_dir, 'n_dropped': 0}

    print(f"  -> {len(keep_lines)} rows remain, "
          f"{len(set(round(float(json.loads(l)['amp']), 6) for l in keep_lines))}"
          f" amplitudes")

    if args.dry_run:
        return {'state_dir': state_dir, 'n_dropped': len(dropped), 'dry': True}

    # --- apply --------------------------------------------------------------
    backup = f"{index_path}.prebak_{stamp}"
    shutil.copy2(index_path, backup)
    tmp = index_path + '.tmp'
    with open(tmp, 'w') as f:
        f.writelines(keep_lines)
    os.replace(tmp, index_path)

    log_path = os.path.join(state_dir, 'prune_log.json')
    log = []
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log = json.load(f)
    log.append({
        'stamp': stamp,
        'backup': os.path.basename(backup),
        'drop_all_amps': args.drop_all,
        'drop_asym_amps': drop_asym,
        'n_before': len(records),
        'n_after': len(keep_lines),
        'duplicate_report': {str(k): v for k, v in dup.items()},
        'dropped': dropped,
        'npz_purged': bool(args.purge_npz),
    })
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    freed = 0
    points_dir = os.path.join(state_dir, 'points')
    for d in dropped:
        p = os.path.join(points_dir, f"{d['idx']:06d}.npz")
        if os.path.isfile(p):
            if args.purge_npz:
                freed += os.path.getsize(p)
                os.remove(p)
            else:
                freed += os.path.getsize(p)
    verb = 'freed' if args.purge_npz else 'orphaned (kept)'
    print(f"  npz {verb}: {freed / 1e6:.1f} MB")
    print(f"  backup: {os.path.basename(backup)}")

    return {'state_dir': state_dir, 'n_dropped': len(dropped),
            'n_after': len(keep_lines), 'bytes': freed}


def restore_run(state_dir):
    index_path = os.path.join(state_dir, 'index.jsonl')
    baks = sorted(glob.glob(index_path + '.prebak_*'))
    if not baks:
        return False
    latest = baks[-1]
    shutil.copy2(latest, index_path)
    print(f"{state_dir}: restored from {os.path.basename(latest)}")
    log_path = os.path.join(state_dir, 'prune_log.json')
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log = json.load(f)
        if log and log[-1].get('npz_purged'):
            print("  WARNING: that prune ran with --purge-npz. The index is "
                  "back but the arrays for those points are gone.")
    return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='states/*',
                    help="glob of state directories (default: states/*)")
    ap.add_argument('--drop-all', type=float, nargs='*', default=DEFAULT_DROP_ALL,
                    metavar='AMP', help="amplitudes to remove entirely")
    ap.add_argument('--drop-asym', type=float, nargs='*', default=DEFAULT_DROP_ASYM,
                    metavar='AMP', help="amplitudes where only asym* branches go")
    ap.add_argument('--dup-tol', type=float, default=None, metavar='X',
                    help="refuse to drop asym rows if max |C_L_asym - C_L_sym| "
                         "exceeds X (they would not be duplicates)")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--purge-npz', action='store_true',
                    help="also delete the arrays of dropped points (irreversible)")
    ap.add_argument('--force', action='store_true',
                    help="drop even where the safety check objects")
    ap.add_argument('--restore', action='store_true',
                    help="undo the most recent prune in each run")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(args.runs)
                  if os.path.isdir(d) and '.bak_' not in d)
    if not dirs:
        print(f"no state directories matched {args.runs!r}")
        return 1

    if args.restore:
        n = sum(restore_run(d) for d in dirs)
        print(f"\nrestored {n} run(s)")
        return 0

    stamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"prune  drop_all={args.drop_all}  drop_asym={args.drop_asym}  "
          f"dry_run={args.dry_run}")

    results = [r for r in (prune_run(d, args, stamp) for d in dirs) if r]
    total = sum(r.get('n_dropped', 0) for r in results)
    print(f"\n{'-' * 60}")
    print(f"{len(results)} run(s) touched, {total} rows dropped"
          + ("  (dry run -- nothing written)" if args.dry_run else ""))
    if not args.dry_run and total:
        print("\nregenerate downstream: sweep_results.csv, critical_curve.csv,")
        print("section_6_figures/*, then re-run audit_section6.py")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
