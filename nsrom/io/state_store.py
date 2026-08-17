"""
Per-point state store for bifurcation-diagram runs.

Purpose
-------
A continuation run is path-dependent: each point's initial guess comes from
the previous converged point, so two configurations (say mode='tensor' and
mode='deim') visit different (Re, amp) sets and start each solve from
different guesses. Errors and timings computed across such runs are not
comparable.

This module records, at every point of a *master* run:

  * the reduced initial guess actually used  (u_in, p_in)  -- the replay seed
  * the converged reduced coefficients        (u_out, p_out)
  * which cluster, branch, and mode produced them
  * optionally the full-order FOM velocity/pressure DOFs at that point

A later replay script can then loop over these frozen points and re-solve
each one, in any mode, from the identical guess -- giving a controlled
per-point comparison with no path dependence.

Layout
------
    state_dir/
        index.jsonl          one JSON object per point (scalars only)
        points/000000.npz    per-point arrays
        points/000001.npz
        ...

Each .npz holds: u_in, p_in, u_out, p_out, and (when the FOM ran and the
point was selected by `fom_field_stride`) u_fom, p_fom.

FOM fields dominate the size: one point costs
``(n_u + n_p) * itemsize`` bytes. At ~200k velocity DOFs and float32 that is
about 0.8 MB per point, so a few thousand points runs to gigabytes. Use
`fom_field_stride` to keep every Nth field, or `save_fom_fields=False` to
keep only the scalars.

Usage
-----
    writer = StateWriter("states/K4_tensor", mode='tensor',
                         fom_field_stride=1)
    ...
    writer.write(...)          # once per diagram point
    writer.close()

    store = StateStore("states/K4_tensor")
    for pt in store:
        ...                    # pt.u_in, pt.Re, pt.cluster, ...
"""

import json
import os

import numpy as np

__all__ = ['StateWriter', 'StateStore', 'StatePoint']


_INDEX_NAME = "index.jsonl"
_POINTS_DIR = "points"


def _as_f64(a):
    return None if a is None else np.ascontiguousarray(a, dtype=np.float64)


class StateWriter:
    """
    Append-only writer. One .npz per point plus a JSONL index of scalars.

    Parameters
    ----------
    state_dir : str
        Directory to create/populate.
    mode : str
        The mode of the run producing these states ('exact'|'deim'|'tensor').
        Recorded per point so a replay can assert it is comparing like runs.
    save_fom_fields : bool, default True
        Store full-order velocity/pressure DOFs. Required if a replay is to
        recompute velocity errors against the FOM.
    fom_field_stride : int, default 1
        Keep the FOM field on every Nth point that has one. Scalars
        (C_L_fom, t_fom, ...) are always kept for every point.
    fom_field_dtype : {'float32', 'float64'}, default 'float32'
        Storage precision for FOM fields. float32 halves the size; it is
        well below the ROM error scale, but use float64 if you intend to
        quote errors below ~1e-6.
    meta : dict or None
        Free-form run metadata (K, inner product, tolerances, git hash...)
        written once to state_dir/run_meta.json.
    """

    def __init__(self, state_dir, *, mode, save_fom_fields=True,
                 fom_field_stride=1, fom_field_dtype='float32', meta=None):
        if fom_field_dtype not in ('float32', 'float64'):
            raise ValueError("fom_field_dtype must be 'float32' or 'float64'")
        if fom_field_stride < 1:
            raise ValueError("fom_field_stride must be >= 1")

        self.state_dir = state_dir
        self.points_dir = os.path.join(state_dir, _POINTS_DIR)
        os.makedirs(self.points_dir, exist_ok=True)

        self.mode = mode
        self.save_fom_fields = save_fom_fields
        self.fom_field_stride = int(fom_field_stride)
        self.fom_field_dtype = fom_field_dtype

        self._n = 0                 # points written
        self._n_fom_fields = 0      # points whose FOM field was stored
        self._bytes_fields = 0

        self._index = open(os.path.join(state_dir, _INDEX_NAME), "w")

        run_meta = {
            'mode': mode,
            'save_fom_fields': bool(save_fom_fields),
            'fom_field_stride': self.fom_field_stride,
            'fom_field_dtype': fom_field_dtype,
        }
        if meta:
            run_meta.update(meta)
        with open(os.path.join(state_dir, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2, default=str)

    # -- writing ----------------------------------------------------------

    def write(self, *, Re, amp, branch, cluster, converged, iterations,
              u_in, p_in, u_out=None, p_out=None,
              t_rom=None, mu=None, C_L_rom=None, C_D_rom=None,
              u_fom=None, p_fom=None, C_L_fom=None, err_u=None,
              t_fom=None, fom_converged=None, fom_iters=None):
        """
        Record one diagram point. Returns its integer index.

        `u_in`/`p_in` are the reduced coefficients handed to the solver as
        the initial guess -- these are what a replay must reuse. `u_out`/
        `p_out` are the converged coefficients (pass None if diverged).
        """
        idx = self._n
        arrays = {
            'u_in': _as_f64(u_in),
            'p_in': _as_f64(p_in),
        }
        if u_out is not None:
            arrays['u_out'] = _as_f64(u_out)
        if p_out is not None:
            arrays['p_out'] = _as_f64(p_out)

        stored_field = False
        if (self.save_fom_fields and u_fom is not None
                and (self._n_fom_fields * 0 + idx) % self.fom_field_stride == 0):
            dt = np.dtype(self.fom_field_dtype)
            arrays['u_fom'] = np.ascontiguousarray(u_fom, dtype=dt)
            if p_fom is not None:
                arrays['p_fom'] = np.ascontiguousarray(p_fom, dtype=dt)
            stored_field = True
            self._n_fom_fields += 1
            self._bytes_fields += sum(
                arrays[k].nbytes for k in ('u_fom', 'p_fom') if k in arrays
            )

        np.savez_compressed(
            os.path.join(self.points_dir, f"{idx:06d}.npz"), **arrays
        )

        rec = {
            'idx': idx,
            'Re': float(Re),
            'amp': float(amp),
            'branch': str(branch),
            'cluster': int(cluster),
            'mode': self.mode,
            'converged': bool(converged),
            'iterations': int(iterations),
            'has_fom_field': bool(stored_field),
        }
        for key, val in (('t_rom', t_rom), ('mu', mu),
                         ('C_L_rom', C_L_rom), ('C_D_rom', C_D_rom),
                         ('C_L_fom', C_L_fom), ('err_u', err_u),
                         ('t_fom', t_fom)):
            if val is not None:
                rec[key] = float(val)
        if fom_converged is not None:
            rec['fom_converged'] = bool(fom_converged)
        if fom_iters is not None:
            rec['fom_iters'] = int(fom_iters)

        self._index.write(json.dumps(rec) + "\n")
        self._index.flush()      # crash-safe: a killed run keeps its points

        self._n += 1
        return idx

    def close(self, verbose=True):
        self._index.close()
        if verbose:
            mb = self._bytes_fields / 1e6
            print(f"  [states] {self._n} points -> {self.state_dir}")
            print(f"  [states] {self._n_fom_fields} FOM fields stored "
                  f"(~{mb:.0f} MB raw, compressed on disk)")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# =============================================================================
# READING
# =============================================================================

class StatePoint:
    """One stored point. Arrays are loaded lazily on first access."""

    __slots__ = ('_rec', '_dir', '_arrays')

    def __init__(self, rec, points_dir):
        self._rec = rec
        self._dir = points_dir
        self._arrays = None

    def __getattr__(self, name):
        if name in self._rec:
            return self._rec[name]
        arrays = self._load()
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)

    def _load(self):
        if self._arrays is None:
            path = os.path.join(self._dir, f"{self._rec['idx']:06d}.npz")
            with np.load(path) as d:
                self._arrays = {k: d[k] for k in d.files}
        return self._arrays

    def get(self, name, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    @property
    def record(self):
        return dict(self._rec)

    def __repr__(self):
        return (f"StatePoint(idx={self._rec['idx']}, Re={self._rec['Re']:.2f}, "
                f"amp={self._rec['amp']:+.2f}, branch={self._rec['branch']!r}, "
                f"k={self._rec['cluster']}, conv={self._rec['converged']})")


class StateStore:
    """Read-only view over a directory written by StateWriter."""

    def __init__(self, state_dir):
        self.state_dir = state_dir
        self.points_dir = os.path.join(state_dir, _POINTS_DIR)

        meta_path = os.path.join(state_dir, "run_meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

        self.records = []
        with open(os.path.join(state_dir, _INDEX_NAME)) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return StatePoint(self.records[i], self.points_dir)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def filter(self, *, converged=None, branch=None, cluster=None,
               has_fom_field=None, predicate=None):
        """Return the subset of points matching all given criteria."""
        out = []
        for rec in self.records:
            if converged is not None and rec['converged'] != converged:
                continue
            if branch is not None and not rec['branch'].startswith(branch):
                continue
            if cluster is not None and rec['cluster'] != cluster:
                continue
            if (has_fom_field is not None
                    and rec.get('has_fom_field', False) != has_fom_field):
                continue
            pt = StatePoint(rec, self.points_dir)
            if predicate is not None and not predicate(pt):
                continue
            out.append(pt)
        return out

    def summary(self):
        n = len(self.records)
        n_conv = sum(r['converged'] for r in self.records)
        n_fld = sum(r.get('has_fom_field', False) for r in self.records)
        branches = sorted({r['branch'] for r in self.records})
        print(f"StateStore: {self.state_dir}")
        print(f"  mode:       {self.meta.get('mode', '?')}")
        print(f"  points:     {n}  ({n_conv} converged)")
        print(f"  FOM fields: {n_fld}")
        print(f"  branches:   {len(branches)}")
