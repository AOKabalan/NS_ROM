"""
Speed-up reporting for the bifurcation-diagram sweeps.

This is the canonical ``nsrom.plotting.speedup`` implementation, used by
``nsrom.workflows.local_pipeline`` to print the speed-up table after a sweep.
The current ``build_full_diagram_bare`` workflow returns a flat list of records,
each carrying (when fom_compute=True):

    t_rom, t_fom, speedup, converged, fom_converged, branch, Re, amp

This module reduces that to the numbers you actually report:

    - median and IQR of the per-solve speed-up (the headline)
    - aggregate speed-up = sum(t_fom) / sum(t_rom) over the whole diagram
    - per-branch breakdown

Why median, not mean: the first FOM point on every leg is seeded from the ROM
field, so it is an unusually good initial guess and converges in very few
Newton steps -- it understates t_fom. Near-merge rows do the opposite. The
median over converged points is the honest per-solve figure; the aggregate
sum/sum is the honest "cost of drawing the whole diagram" figure. Report both.
"""

import numpy as np


def _clean(records, *, require_fom=True, drop_leg_firsts=False):
    """Rows with a finite t_rom and t_fom where both solves converged."""
    out, seen_branches = [], set()
    for r in records:
        if not r.get('converged', False):
            continue
        t_rom = r.get('t_rom')
        t_fom = r.get('t_fom')
        if t_rom is None or t_rom <= 0:
            continue
        if require_fom:
            if t_fom is None or not np.isfinite(t_fom):
                continue
            if not r.get('fom_converged', False):
                continue
        # optionally skip the first converged point of each branch leg, whose
        # FOM solve was ROM-seeded (unfairly cheap for the FOM side)
        if drop_leg_firsts:
            b = r.get('branch')
            if b not in seen_branches:
                seen_branches.add(b)
                continue
        out.append(r)
    return out


def speedup_summary(records, *, drop_leg_firsts=True, verbose=True):
    """Return a dict of speed-up statistics and (optionally) print a table.

    drop_leg_firsts=True excludes the ROM-seeded first FOM solve on each leg,
    which is the fair choice for the per-solve median. The aggregate sum/sum is
    computed over ALL converged rows regardless, since it is a total-cost figure.
    """
    rows_all   = _clean(records, drop_leg_firsts=False)
    rows_fair  = _clean(records, drop_leg_firsts=drop_leg_firsts)

    if not rows_all:
        if verbose:
            print("[speedup] no converged FOM/ROM pairs -- was fom_compute=True?")
        return {}

    t_rom_all = np.array([r['t_rom'] for r in rows_all])
    t_fom_all = np.array([r['t_fom'] for r in rows_all])
    su_fair   = np.array([r['t_fom'] / r['t_rom'] for r in rows_fair]) \
                if rows_fair else np.array([r['t_fom'] / r['t_rom'] for r in rows_all])

    agg = float(t_fom_all.sum() / t_rom_all.sum())

    stats = {
        'n_points':        len(rows_all),
        'n_points_fair':   len(su_fair),
        't_rom_median':    float(np.median(t_rom_all)),
        't_fom_median':    float(np.median(t_fom_all)),
        'speedup_median':  float(np.median(su_fair)),
        'speedup_p25':     float(np.percentile(su_fair, 25)),
        'speedup_p75':     float(np.percentile(su_fair, 75)),
        'speedup_min':     float(su_fair.min()),
        'speedup_max':     float(su_fair.max()),
        'speedup_aggregate': agg,
    }

    if verbose:
        print("=" * 60)
        print("SPEED-UP SUMMARY  (solve-only, FOM continuation vs ROM)")
        print("=" * 60)
        print(f"  points (converged pairs):     {stats['n_points']}")
        print(f"  points used for median:       {stats['n_points_fair']}"
              f"  (leg-first FOM solves dropped: {drop_leg_firsts})")
        print(f"  median t_rom:                 {stats['t_rom_median']*1e3:8.2f} ms")
        print(f"  median t_fom:                 {stats['t_fom_median']*1e3:8.2f} ms")
        print(f"  per-solve speed-up (median):  {stats['speedup_median']:8.1f} x")
        print(f"    IQR [p25, p75]:             "
              f"[{stats['speedup_p25']:.1f}, {stats['speedup_p75']:.1f}] x")
        print(f"    range [min, max]:           "
              f"[{stats['speedup_min']:.1f}, {stats['speedup_max']:.1f}] x")
        print(f"  aggregate speed-up (sum/sum): {stats['speedup_aggregate']:8.1f} x")
        print("=" * 60)
        _per_branch(rows_all)

    return stats


def _per_branch(rows):
    """Aggregate sum/sum speed-up per branch id."""
    from collections import defaultdict
    tr = defaultdict(float)
    tf = defaultdict(float)
    n  = defaultdict(int)
    for r in rows:
        b = r.get('branch', '?')
        tr[b] += r['t_rom']
        tf[b] += r['t_fom']
        n[b]  += 1
    print("  per-branch (aggregate sum t_fom / sum t_rom):")
    for b in sorted(tr):
        su = tf[b] / tr[b] if tr[b] > 0 else float('nan')
        print(f"    {b:24s}  n={n[b]:3d}  "
              f"t_rom={tr[b]*1e3:8.1f}ms  t_fom={tf[b]*1e3:8.1f}ms  {su:6.1f}x")


if __name__ == '__main__':
    # quick self-test on synthetic records
    import random
    recs = []
    for b in ('sym', 'asym+1', 'asym-1'):
        for i in range(20):
            tr = 0.01 + random.random() * 0.005
            tf = tr * (80 + random.random() * 40)
            recs.append({'branch': b, 'converged': True, 'fom_converged': True,
                         't_rom': tr, 't_fom': tf})
    speedup_summary(recs)
