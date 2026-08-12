#!/usr/bin/env python
"""
prune_csv.py -- carry the index retraction through to sweep_results.csv.

Why not just re-join on idx
---------------------------
CSV row order and the store's `idx` do not correspond -- E1 has 4258 CSV rows
against 4295 index rows. But the retraction was never defined by index: it is a
predicate on (branch, amp). The same predicate applied to the CSV gives an
exactly consistent result with no join at all.

The rule is read per-run from prune_log.json, so runs pruned with different
amplitude lists (E11/E12 at +0.6/+0.8, E5/E6 at +0.8, the rest at +0.7..+1.0)
each get their own. --drop-all / --drop-asym override it for runs with no log.

After filtering, every surviving CSV row is looked up in the pruned index and
the amplitude sets are compared. That is the check that the CSV and the store
now agree -- and it also surfaces the pre-existing CSV/index row gap instead of
silently papering over it.

Usage
-----
    python prune_csv.py --dry-run          # report, write nothing
    python prune_csv.py                    # apply to states/*/sweep_results.csv
    python prune_csv.py --verify           # compare only, no filtering
    python prune_csv.py --restore          # put the previous CSV back
"""

import argparse
import csv
import glob
import json
import os
import shutil
import time

AMP_TOL = 1e-8
CSV_NAME = 'sweep_results.csv'


def _near(a, values, tol=AMP_TOL):
    return any(abs(float(a) - v) < tol for v in values)


def _key(branch, Re, amp):
    return (str(branch), round(float(Re), 6), round(float(amp), 6))


def load_rule(state_dir, args):
    """(drop_all, drop_asym, source) for this run."""
    log_path = os.path.join(state_dir, 'prune_log.json')
    if os.path.isfile(log_path):
        try:
            with open(log_path) as f:
                log = json.load(f)
            if log:
                last = log[-1]
                return (last.get('drop_all_amps', []),
                        last.get('drop_asym_amps', []),
                        f"prune_log.json[{len(log) - 1}] {last['stamp']}")
        except (ValueError, KeyError):
            pass
    return list(args.drop_all), list(args.drop_asym), 'command line'


def retracted(branch, amp, drop_all, drop_asym):
    if _near(amp, drop_all):
        return 'amp_retracted'
    if _near(amp, drop_asym) and str(branch).startswith('asym'):
        return 'asym_duplicate'
    return None


def process(state_dir, args, stamp):
    csv_path = os.path.join(state_dir, CSV_NAME)
    index_path = os.path.join(state_dir, 'index.jsonl')
    if not os.path.isfile(csv_path):
        return None

    idx_keys, idx_amps = set(), set()
    if os.path.isfile(index_path):
        with open(index_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    idx_keys.add(_key(r['branch'], r['Re'], r['amp']))
                    idx_amps.add(round(float(r['amp']), 6))

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if not rows or 'branch' not in fields or 'amp' not in fields:
        print(f"\n{state_dir}\n  {CSV_NAME}: no branch/amp columns, skipped")
        return None

    drop_all, drop_asym, source = load_rule(state_dir, args)

    keep, dropped = [], {}
    for row in rows:
        why = retracted(row['branch'], float(row['amp']), drop_all, drop_asym)
        if why is None:
            keep.append(row)
        else:
            dropped[why] = dropped.get(why, 0) + 1

    csv_amps = {round(float(r['amp']), 6) for r in keep}
    miss_by_amp = {}
    for r in keep:
        if _key(r['branch'], r['Re'], r['amp']) not in idx_keys:
            a = round(float(r['amp']), 6)
            miss_by_amp[a] = miss_by_amp.get(a, 0) + 1
    missing = sum(miss_by_amp.values())

    print(f"\n{state_dir}")
    print(f"  rule: drop_all={drop_all} drop_asym={drop_asym}   ({source})")
    print(f"  csv {len(rows)} -> {len(keep)} rows"
          + (f"   [{', '.join(f'{k} {v}' for k, v in sorted(dropped.items()))}]"
             if dropped else "   [nothing to drop]"))

    ok = True
    if idx_keys:
        print(f"  index {len(idx_keys)} rows,  csv amps {len(csv_amps)} vs "
              f"index amps {len(idx_amps)}")
        extra = sorted(csv_amps - idx_amps)
        gone = sorted(idx_amps - csv_amps)
        if extra:
            print(f"  MISMATCH: amps in csv but not in index: {extra}")
            ok = False
        if gone:
            print(f"  note: amps in index but not in csv: {gone}")
        if missing:
            hot = sorted((a for a, n in miss_by_amp.items() if n >= args.hot),
                         key=float)
            pct = 100.0 * missing / max(len(keep), 1)
            if hot:
                print(f"  MISMATCH: {missing} surviving csv rows have no index "
                      f"row, concentrated at amp {hot}")
                print("            the index dropped more than this rule does "
                      "-- the rule is stale")
                ok = False
            else:
                print(f"  note: {missing} surviving csv rows ({pct:.1f}%) have "
                      f"no index row, spread across "
                      f"{len(miss_by_amp)} amplitudes -- pre-existing gap, "
                      f"not caused by pruning")
    else:
        print("  no index.jsonl -- cross-check skipped")

    if args.verify or args.dry_run or not dropped:
        return {'dir': state_dir, 'dropped': sum(dropped.values()), 'ok': ok}

    if not ok and not args.force:
        print("  NOT written: the rule disagrees with the index "
              "(--force to override)")
        return {'dir': state_dir, 'dropped': 0, 'ok': False}

    backup = f"{csv_path}.prebak_{stamp}"
    shutil.copy2(csv_path, backup)
    tmp = csv_path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    os.replace(tmp, csv_path)
    print(f"  written, backup {os.path.basename(backup)}")
    return {'dir': state_dir, 'dropped': sum(dropped.values()), 'ok': ok}


def restore(state_dir):
    csv_path = os.path.join(state_dir, CSV_NAME)
    baks = sorted(glob.glob(csv_path + '.prebak_*'))
    if not baks:
        return False
    shutil.copy2(baks[-1], csv_path)
    print(f"{state_dir}: restored {CSV_NAME} from {os.path.basename(baks[-1])}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='states/*')
    ap.add_argument('--drop-all', type=float, nargs='*', default=[-0.25],
                    metavar='AMP', help="fallback rule where no prune_log exists")
    ap.add_argument('--drop-asym', type=float, nargs='*',
                    default=[0.7, 0.8, 0.9, 1.0], metavar='AMP')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--verify', action='store_true',
                    help="cross-check csv against index, change nothing")
    ap.add_argument('--hot', type=int, default=20, metavar='N',
                    help="unmatched rows at one amplitude above N mean the "
                         "rule is stale, not a pre-existing gap")
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(args.runs)
                  if os.path.isdir(d) and '.bak_' not in d)
    if args.restore:
        print(f"restored {sum(restore(d) for d in dirs)} csv(s)")
        return 0

    stamp = time.strftime('%Y%m%d_%H%M%S')
    results = [r for r in (process(d, args, stamp) for d in dirs) if r]
    total = sum(r['dropped'] for r in results)
    bad = [r['dir'] for r in results if not r['ok']]
    print(f"\n{'-' * 60}")
    print(f"{len(results)} csv(s), {total} rows dropped"
          + ("  (nothing written)" if args.dry_run or args.verify else ""))
    if bad:
        print(f"mismatched: {', '.join(bad)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
