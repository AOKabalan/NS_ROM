"""
Configuration for local ROM construction.

All fields are required. The user must explicitly set every parameter so that
no run depends on a hidden default.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import json
import os
from dataclasses import asdict

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass(kw_only=True)
class RunManifest:
    """
    Run-level inputs that aren't ROM construction knobs but define the run.

    Sits alongside LocalROMConfig in the persisted record.
    """
    # --- Inputs ---
    snapshot_dir       : str
    deim_snapshot_dir  : Optional[str]   # None when use_deim is False
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


def save_run(cfg, manifest, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_d = asdict(cfg)
    cfg_d["boundary_markers"] = list(cfg_d["boundary_markers"])
    man_d = asdict(manifest)
    with open(path, "w") as f:
        json.dump({"config": cfg_d, "manifest": man_d}, f, indent=2)


def load_run(path):
    with open(path) as f:
        d = json.load(f)
    cfg_d = d["config"]
    cfg_d["boundary_markers"] = tuple(cfg_d["boundary_markers"])
    return LocalROMConfig(**cfg_d), RunManifest(**d["manifest"])

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

    # --- Mode flags ---
    use_deim: bool
    use_tensor: bool
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

def save_config(cfg, path):
    """Dump a LocalROMConfig to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = asdict(cfg)
    # Tuples round-trip as lists through JSON; make that explicit.
    d["boundary_markers"] = list(d["boundary_markers"])
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_config(path):
    """Load a LocalROMConfig from JSON."""
    with open(path) as f:
        d = json.load(f)
    d["boundary_markers"] = tuple(d["boundary_markers"])
    return LocalROMConfig(**d)