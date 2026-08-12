#!/usr/bin/env python
"""
dedupe_states.py -- collapse repeated (branch, Re, amp) rows.

A trunk whose `mu` crosses zero twice fires the branch jump twice, and both
spawns of a given sign are continued under the same branch_id. Where their Re
ranges overlap the store holds the same point twice.

Keeps the FIRST occurrence of each key and reports, per run, whether the copies
actually agree -- if they disagree on C_L_rom or converged, the duplication is
not benign and the script refuses to write.

    python dedupe_states.py --check                    # report only
    python dedupe_states.py --runs 'states/E7*' 'states/E13*'
    python dedupe_states.py --restore
"""

import argparse, csv, glob, json, os, shutil, time

CSV_NAME = 'sweep_results.csv'
COMPARE = ('C_L_rom', 'mu', 'converged', 'iterations')


def key(branch, Re, amp):
    return (str(branch), round(float(Re), 6), round(float(amp), 6))


def scan(rows, get):
    """-> (keep_flags, n_dup, disagreements)"""
    seen, flags, disagree = {}, [], {}
    for i, r in enumerate(rows):
        k = key(get(r, 'branch'), get(r, 'Re'), get(r, 'amp'))
        if k not in seen:
            seen[k] = i
            flags.append(True)
        else:
            flags.append(False)
            first = rows[seen[k]]
            for col in COMPARE:
                a, b = get(first, col), get(r, col)
                if a is None or b is None or a == '' or b == '':
                    continue
                try:
                    same = abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))
                except (TypeError, ValueError):
                    same = str(a) == str(b)
                if not same:
                    disagree[col] = disagree.get(col, 0) + 1
    return flags, sum(1 for f in flags if not f), disagree


def run(state_dir, args, stamp):
    ipath = os.path.join(state_dir, 'index.jsonl')
    if not os.path.isfile(ipath):
        return 0
    with open(ipath) as f:
        lines = [l for l in f if l.strip()]
    recs = [json.loads(l) for l in lines]
    flags, ndup, disagree = scan(recs, lambda r, c: r.get(c))
    if ndup == 0:
        return 0

    print(f"\n{state_dir}")
    print(f"  index {len(recs)} -> {len(recs) - ndup} rows  ({ndup} duplicate keys)")
    by_branch = {}
    for r, keep in zip(recs, flags):
        if not keep:
            by_branch[r['branch'].split('@')[0]] = by_branch.get(
                r['branch'].split('@')[0], 0) + 1
    print("  by branch: " + ", ".join(f"{k} {v}" for k, v in sorted(by_branch.items())))
    if disagree:
        print("  DISAGREE: " + ", ".join(f"{k} on {v} rows" for k, v in disagree.items()))
        if not args.force:
            print("  not written -- copies are not identical (--force to override)")
            return 0
    else:
        print(f"  copies identical on {', '.join(COMPARE)}")

    if args.check:
        return ndup

    shutil.copy2(ipath, f"{ipath}.prebak_{stamp}")
    tmp = ipath + '.tmp'
    with open(tmp, 'w') as f:
        f.writelines(l for l, k in zip(lines, flags) if k)
    os.replace(tmp, ipath)

    cpath = os.path.join(state_dir, CSV_NAME)
    if os.path.isfile(cpath):
        with open(cpath, newline='') as f:
            rd = csv.DictReader(f)
            fields, rows = rd.fieldnames, list(rd)
        cflags, cdup, _ = scan(rows, lambda r, c: r.get(c))
        if cdup:
            shutil.copy2(cpath, f"{cpath}.prebak_{stamp}")
            tmp = cpath + '.tmp'
            with open(tmp, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(r for r, k in zip(rows, cflags) if k)
            os.replace(tmp, cpath)
            print(f"  csv {len(rows)} -> {len(rows) - cdup} rows")
    print("  written")
    return ndup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=['states/*'])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()

    dirs = sorted({d for pat in args.runs for d in glob.glob(pat)
                   if os.path.isdir(d) and '.bak_' not in d})
    if args.restore:
        n = 0
        for d in dirs:
            for name in ('index.jsonl', CSV_NAME):
                p = os.path.join(d, name)
                baks = sorted(glob.glob(p + '.prebak_*'))
                if baks:
                    shutil.copy2(baks[-1], p)
                    print(f"{d}: restored {name} from {os.path.basename(baks[-1])}")
                    n += 1
        print(f"restored {n} file(s)")
        return 0

    stamp = time.strftime('%Y%m%d_%H%M%S')
    total = sum(run(d, args, stamp) for d in dirs)
    print(f"\n{'-' * 60}\n{total} duplicate rows"
          + ("  (nothing written)" if args.check else ""))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
