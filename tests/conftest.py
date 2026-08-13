"""
Session-scoped pytest fixtures for the characterization suite.

Everything expensive — the Firedrake import, the saved-data loads, the mesh /
problem construction, and the prebuilt-ROM load — is paid ONCE per pytest
process and shared across tests.
"""
import json
import os

import pytest

from tests import characterization as ch

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


# --- shared data (Tier A) ----------------------------------------------------
@pytest.fixture(scope="session")
def snapshots():
    return ch.load_stratified_snapshots()


@pytest.fixture(scope="session")
def S(snapshots):
    return snapshots["S"]


@pytest.fixture(scope="session")
def M_u():
    return ch.load_Mu()


# --- Firedrake objects (Tier B) ---------------------------------------------
@pytest.fixture(scope="session")
def problem():
    return ch.build_problem()


@pytest.fixture(scope="session")
def rom_bundle(problem):
    """Prebuilt K4_H1 tensor ROM, loaded read-only. Shared by B2/B3."""
    return ch._build_rom_bundle_readonly(problem)


# --- golden accessor ---------------------------------------------------------
@pytest.fixture(scope="session")
def golden():
    def _load(case):
        path = os.path.join(GOLDEN_DIR, f"{case}.json")
        if not os.path.exists(path):
            pytest.skip(f"golden/{case}.json not captured yet "
                        "(run tests/generate_golden.py --write)")
        with open(path) as f:
            return json.load(f)
    return _load
