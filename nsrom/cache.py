import hashlib
import inspect
import json
import os
import time
import functools
import numpy as np


# ── Hashing helpers ──────────────────────────────────────────────────────────

def _hash_update(h, obj):
    """Recursively feed an object into a hashlib hasher."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        h.update(repr(obj).encode())

    elif isinstance(obj, np.ndarray):
        # Fast fingerprint: shape + dtype + head/tail bytes
        h.update(f"ndarray:{obj.shape}:{obj.dtype}".encode())
        flat = obj.ravel()
        n = min(500, len(flat))
        h.update(flat[:n].tobytes())
        h.update(flat[-n:].tobytes())

    elif isinstance(obj, (list, tuple)):
        h.update(f"seq:{len(obj)}".encode())
        for item in obj:
            _hash_update(h, item)

    elif isinstance(obj, dict):
        for k in sorted(obj.keys()):
            h.update(str(k).encode())
            _hash_update(h, obj[k])

    else:
        # Non-hashable (Firedrake objects etc.) — skip silently
        h.update(f"__unhashable__:{type(obj).__name__}".encode())


def _hash_directory(path):
    """Cheap fingerprint of a directory: file names + sizes + mtimes."""
    h = hashlib.sha256()
    if not os.path.isdir(path):
        h.update(b"__not_a_dir__")
        return h.hexdigest()[:16]
    for root, dirs, files in sorted(os.walk(path)):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            stat = os.stat(fpath)
            h.update(f"{fpath}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


# ── Serialization helpers ────────────────────────────────────────────────────

def _save_cache(cache_path, result_dict, exclude_keys):
    """Save cacheable entries from a dict. numpy arrays → .npz, rest skipped."""
    to_save = {}
    skipped = []
    for k, v in result_dict.items():
        if k in exclude_keys:
            skipped.append(k)
            continue
        if isinstance(v, np.ndarray):
            to_save[k] = v
        else:
            skipped.append(k)
    np.savez_compressed(cache_path, **to_save)
    # Save metadata so we know what was excluded
    meta_path = cache_path.replace(".npz", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"cached_keys": list(to_save.keys()), "skipped_keys": skipped}, f)
    return to_save.keys()


def _load_cache(cache_path):
    """Load cached arrays back into a dict."""
    data = np.load(cache_path)
    return {k: data[k] for k in data.files}


# ── The decorator ────────────────────────────────────────────────────────────

def disk_cache(cache_dir=".cache", exclude_keys=None, dir_args=None, hash_source=True):
    """
    Cache decorator for functions that return a dict of numpy arrays.

    Parameters
    ----------
    cache_dir : str
        Where to store .npz cache files.
    exclude_keys : list of str
        Return-dict keys to never cache (non-serializable objects like
        Firedrake problem, PETSc matrices). These are recomputed every time
        by a `rebuild` dict you return alongside the main result.
    dir_args : list of str
        Names of arguments that are directory paths. Their *contents* will
        be hashed (via mtime/size) so the cache auto-invalidates when files
        in those directories change.
    hash_source : bool
        If True, the function's source code is included in the hash, so
        editing the function body invalidates the cache automatically.
    """
    exclude_keys = set(exclude_keys or [])
    dir_args = set(dir_args or [])

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # ── Build cache key ──────────────────────────────────────
            h = hashlib.sha256()
            h.update(func.__qualname__.encode())

            if hash_source:
                try:
                    h.update(inspect.getsource(func).encode())
                except OSError:
                    pass  # e.g. interactive session

            for name, value in bound.arguments.items():
                h.update(name.encode())
                if name in dir_args and isinstance(value, str):
                    h.update(_hash_directory(value).encode())
                else:
                    _hash_update(h, value)

            cache_key = h.hexdigest()[:24]
            cache_path = os.path.join(cache_dir, f"{func.__name__}_{cache_key}.npz")

            # ── Cache hit ────────────────────────────────────────────
            if os.path.exists(cache_path):
                print(f"[cache] HIT  {func.__name__} → {cache_path}")
                result = _load_cache(cache_path)

                # Rebuild non-cacheable keys
                if exclude_keys:
                    rebuild = func.__rebuild__(*args, **kwargs) if hasattr(func, '__rebuild__') else {}
                    result.update(rebuild)
                return result

            # ── Cache miss: full computation ─────────────────────────
            print(f"[cache] MISS {func.__name__} — computing...")
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"[cache] Computed in {elapsed:.1f}s")

            if not isinstance(result, dict):
                raise TypeError(f"disk_cache requires {func.__name__} to return a dict")

            os.makedirs(cache_dir, exist_ok=True)
            cached_keys = _save_cache(cache_path, result, exclude_keys)
            print(f"[cache] Saved {len(list(cached_keys))} arrays → {cache_path}")
            return result

        # Convenience: manual invalidation
        wrapper.clear_cache = lambda: [
            os.remove(os.path.join(cache_dir, f))
            for f in os.listdir(cache_dir)
            if f.startswith(func.__name__) and f.endswith(".npz")
        ] if os.path.isdir(cache_dir) else None

        return wrapper
    return decorator