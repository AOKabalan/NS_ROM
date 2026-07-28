"""
Configuration for local ROM construction.

All fields are required. The user must explicitly set every parameter so that
no run depends on a hidden default.
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List


# Convection-evaluation strategies. Mirrors nsrom.rom.rom_solver.VALID_MODES;
# duplicated here so config validation does not import Firedrake.
VALID_MODES = ('exact', 'deim', 'tensor')


@dataclass(kw_only=True)
class RunManifest:
    """
    Run-level inputs that aren't ROM construction knobs but define the run.

    Sits alongside LocalROMConfig in the persisted record.
    """
    # --- Inputs ---
    snapshot_dir       : str
    deim_snapshot_dir  : Optional[str]   # None unless cfg.mode == 'deim'
    lifting_dir        : str
    mesh_file          : str
    fom_checkpoint     : str
    reynolds_init      : float
    amplitude_init     : float

    # --- Single-point test ---
    test_re  : float
    test_amp : float

    # --- Sweep ---
    run_sweep  : bool
    fom_every  : int
    re_values  : List[float]
    amp_values : List[float]


@dataclass(kw_only=True)
class LocalROMConfig:
    # --- Clustering ---
    n_clusters: int
    inner_product_type: str

    # --- POD truncation ---
    pod_energy_tol: float
    n_velocity_max: int
    n_pressure_max: int
    n_supremizer_max: int
    boundary_markers: Tuple[int, ...]

    # --- Convection evaluation ---
    mode: str                          # 'exact' | 'deim' | 'tensor'
    compute_affine_convection: bool

    # --- DEIM ---
    deim_energy_tol_F: float
    deim_energy_tol_J: float
    m_F_max: int
    m_J_max: int
    n_modes_F: Optional[int]
    n_modes_J: Optional[int]

    # --- Cache control ---
    recompute_clustering: bool
    recompute_pod: bool
    recompute_deim: bool
    recompute_tensor: bool

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown mode {self.mode!r}; expected one of {VALID_MODES}."
            )
        if self.mode == 'tensor' and not self.compute_affine_convection:
            raise ValueError(
                "mode='tensor' requires compute_affine_convection=True "
                "(the tensor represents the pure quadratic term only)."
            )

    # Convenience predicates, so call sites read naturally without
    # reintroducing the old boolean fields.
    @property
    def use_deim(self) -> bool:
        return self.mode == 'deim'

    @property
    def use_tensor(self) -> bool:
        return self.mode == 'tensor'


# =============================================================================
# PERSISTENCE
# =============================================================================

def _migrate_config_dict(d: dict) -> dict:
    """
    Upgrade a config dict written before `mode` replaced use_deim/use_tensor.

    Legacy records carry the two booleans; new ones carry `mode`. Both are
    accepted so old run.json files stay loadable.
    """
    d = dict(d)

    if 'mode' not in d:
        use_deim = d.pop('use_deim', False)
        use_tensor = d.pop('use_tensor', False)
        if use_deim and use_tensor:
            raise ValueError(
                "Legacy config has both use_deim and use_tensor set; "
                "cannot infer mode. Edit the file and set mode= explicitly."
            )
        d['mode'] = 'tensor' if use_tensor else ('deim' if use_deim else 'exact')
    else:
        d.pop('use_deim', None)
        d.pop('use_tensor', None)

    d.setdefault('recompute_tensor', False)
    d['boundary_markers'] = tuple(d['boundary_markers'])
    return d


def save_config(cfg, path):
    """Dump a LocalROMConfig to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = asdict(cfg)
    # Tuples round-trip as lists through JSON; make that explicit.
    d["boundary_markers"] = list(d["boundary_markers"])
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_config(path):
    """Load a LocalROMConfig from JSON, migrating legacy records."""
    with open(path) as f:
        d = json.load(f)
    return LocalROMConfig(**_migrate_config_dict(d))


def save_run(cfg, manifest, path):
    """Dump a (config, manifest) pair to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_d = asdict(cfg)
    cfg_d["boundary_markers"] = list(cfg_d["boundary_markers"])
    man_d = asdict(manifest)
    with open(path, "w") as f:
        json.dump({"config": cfg_d, "manifest": man_d}, f, indent=2)


def load_run(path):
    """Load a (config, manifest) pair from JSON, migrating legacy records."""
    with open(path) as f:
        d = json.load(f)
    cfg = LocalROMConfig(**_migrate_config_dict(d["config"]))
    return cfg, RunManifest(**d["manifest"])