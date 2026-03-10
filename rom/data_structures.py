"""
ROM Data Structures and I/O.

Provides dataclasses for:
- PODBasis: POD modes and eigenvalues
- ReducedOperators: Reduced-order operators with K-indexed lifting
- OnlineSystem: Assembled reduced system ready for solver
- ROMSolution: ROM solution coefficients

All data stored as numpy arrays for portability (no Firedrake dependency to load).
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
import numpy as np
import os
import json

__all__ = [
    'PODBasis',
    'ReducedOperators',
    'OnlineSystem',
    'ROMSolution',
    'save_pod_basis',
    'load_pod_basis',
    'save_reduced_operators',
    'load_reduced_operators',
    'save_rom_solution',
    'load_rom_solution',
]


# =============================================================================
# POD BASIS
# =============================================================================

@dataclass
class PODBasis:
    """
    POD basis stored as numpy arrays.

    Modes are stored as (n_dofs, n_modes) arrays where columns are modes.
    """
    velocity_modes: np.ndarray
    pressure_modes: np.ndarray
    velocity_eigenvalues: np.ndarray
    pressure_eigenvalues: np.ndarray
    supremizer_modes: Optional[np.ndarray] = None
    supremizer_eigenvalues: Optional[np.ndarray] = None
    inner_product_type: Literal['L2', 'H1', 'H1_semi'] = 'L2'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.velocity_modes.ndim == 2
        assert self.pressure_modes.ndim == 2
        assert len(self.velocity_eigenvalues) == self.velocity_modes.shape[1]
        assert len(self.pressure_eigenvalues) == self.pressure_modes.shape[1]
        if self.supremizer_modes is not None:
            assert self.supremizer_modes.ndim == 2
            assert self.supremizer_modes.shape[0] == self.velocity_modes.shape[0]
            if self.supremizer_eigenvalues is not None:
                assert len(self.supremizer_eigenvalues) == self.supremizer_modes.shape[1]

    @property
    def n_velocity_modes(self) -> int:
        return self.velocity_modes.shape[1]

    @property
    def n_pressure_modes(self) -> int:
        return self.pressure_modes.shape[1]

    @property
    def n_supremizer_modes(self) -> int:
        return 0 if self.supremizer_modes is None else self.supremizer_modes.shape[1]

    @property
    def n_velocity_dofs(self) -> int:
        return self.velocity_modes.shape[0]

    @property
    def n_pressure_dofs(self) -> int:
        return self.pressure_modes.shape[0]

    @property
    def enriched_velocity_modes(self) -> np.ndarray:
        if self.supremizer_modes is not None:
            return np.hstack([self.velocity_modes, self.supremizer_modes])
        return self.velocity_modes

    @property
    def n_enriched_velocity_modes(self) -> int:
        return self.n_velocity_modes + self.n_supremizer_modes

    def velocity_energy_fraction(self, n_modes: int = None) -> float:
        if n_modes is None:
            n_modes = self.n_velocity_modes
        total = self.metadata.get('total_velocity_energy', np.sum(self.velocity_eigenvalues))
        return np.sum(self.velocity_eigenvalues[:n_modes]) / total if total else 0.0

    def pressure_energy_fraction(self, n_modes: int = None) -> float:
        if n_modes is None:
            n_modes = self.n_pressure_modes
        total = self.metadata.get('total_pressure_energy', np.sum(self.pressure_eigenvalues))
        return np.sum(self.pressure_eigenvalues[:n_modes]) / total if total else 0.0

    def truncate(self, n_velocity=None, n_pressure=None, n_supremizer=None) -> 'PODBasis':
        if n_velocity is None:
            n_velocity = self.n_velocity_modes
        if n_pressure is None:
            n_pressure = self.n_pressure_modes
        if n_supremizer is None:
            n_supremizer = self.n_supremizer_modes

        sup_modes = None
        sup_eigs = None
        if self.supremizer_modes is not None and n_supremizer > 0:
            sup_modes = self.supremizer_modes[:, :n_supremizer]
            if self.supremizer_eigenvalues is not None:
                sup_eigs = self.supremizer_eigenvalues[:n_supremizer]

        return PODBasis(
            velocity_modes=self.velocity_modes[:, :n_velocity],
            pressure_modes=self.pressure_modes[:, :n_pressure],
            velocity_eigenvalues=self.velocity_eigenvalues[:n_velocity],
            pressure_eigenvalues=self.pressure_eigenvalues[:n_pressure],
            supremizer_modes=sup_modes,
            supremizer_eigenvalues=sup_eigs,
            inner_product_type=self.inner_product_type,
            metadata={
                **self.metadata,
                'truncated_from': {
                    'n_velocity': self.n_velocity_modes,
                    'n_pressure': self.n_pressure_modes,
                    'n_supremizer': self.n_supremizer_modes,
                }
            }
        )


# =============================================================================
# ONLINE SYSTEM (output of assemble_online)
# =============================================================================

@dataclass
class OnlineSystem:
    """
    Assembled reduced system ready for the solver.

    The solver receives this and only needs to add the DEIM quadratic
    contribution on top. Everything else is exact and precomputed.

    Momentum residual (without quadratic DEIM part):
        R_u = A_eff @ alpha + B_N.T @ beta - f_eff

    Continuity residual:
        R_p = B_N @ alpha - g_eff

    Momentum Jacobian (without quadratic DEIM part):
        J_uu = A_eff
        J_up = B_N.T
        J_pu = B_N
    """
    A_eff: np.ndarray     # (N_u, N_u) nu*A_N + C_lin(thetas)
    B_N: np.ndarray       # (N_p, N_u) divergence (unchanged)
    f_eff: np.ndarray     # (N_u,) adjusted forcing + convection constant
    g_eff: np.ndarray     # (N_p,) adjusted divergence RHS
    C_lin: np.ndarray     # (N_u, N_u) linear convection matrix
    c_0: np.ndarray       # (N_u,) convection constant vector
    nu: float
    thetas: list


# =============================================================================
# REDUCED OPERATORS
# =============================================================================

@dataclass
class ReducedOperators:
    """
    Reduced-order operators with K-indexed lifting components.

    Lifting decomposition:
        u_lift(params) = sum_k  theta_k(params) * lifting_dofs[k]

    For your pinball:
        K=2: lifting_dofs = [u_base, u_control]
             thetas(amp) = [1.0, amp]

    Affine convection terms (optional, for split DEIM):
        c_conv[k,l] = Z_u^T @ assemble( (u_k . grad) u_l )   -- (K, K, N_u)
        N_conv[k]   = Z_u^T @ N(u_k) @ Z_u                   -- (K, N_u, N_u)
        M_conv[k]   = Z_u^T @ M(u_k) @ Z_u                   -- (K, N_u, N_u)
    """
    # Core reduced operators
    A_N: np.ndarray          # (N_u, N_u)
    B_N: np.ndarray          # (N_p, N_u)
    f_N: np.ndarray          # (N_u,)
    g_N: np.ndarray          # (N_p,)

    # Basis matrices
    Z_u: np.ndarray          # (n_vel_dofs, N_u)
    Z_p: np.ndarray          # (n_prs_dofs, N_p)

    # K-indexed lifting
    n_lifting: int                     # K
    lifting_dofs: np.ndarray           # (K, n_vel_dofs)
    A_lift: np.ndarray                 # (K, N_u)
    B_lift: np.ndarray                 # (K, N_p)

    # Affine convection terms (optional)
    c_conv: Optional[np.ndarray] = None   # (K, K, N_u)
    N_conv: Optional[np.ndarray] = None   # (K, N_u, N_u)
    M_conv: Optional[np.ndarray] = None   # (K, N_u, N_u)

    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        N_u = self.A_N.shape[0]
        N_p = self.B_N.shape[0]
        K = self.n_lifting

        assert self.A_N.shape == (N_u, N_u)
        assert self.B_N.shape == (N_p, N_u)
        assert len(self.f_N) == N_u
        assert len(self.g_N) == N_p
        assert self.Z_u.shape[1] == N_u
        assert self.Z_p.shape[1] == N_p

        assert self.lifting_dofs.shape == (K, self.Z_u.shape[0]), \
            f"lifting_dofs shape {self.lifting_dofs.shape} != ({K}, {self.Z_u.shape[0]})"
        assert self.A_lift.shape == (K, N_u), \
            f"A_lift shape {self.A_lift.shape} != ({K}, {N_u})"
        assert self.B_lift.shape == (K, N_p), \
            f"B_lift shape {self.B_lift.shape} != ({K}, {N_p})"

        if self.has_affine_convection:
            assert self.c_conv.shape == (K, K, N_u)
            assert self.N_conv.shape == (K, N_u, N_u)
            assert self.M_conv.shape == (K, N_u, N_u)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def has_affine_convection(self) -> bool:
        return self.c_conv is not None

    @property
    def n_velocity_modes(self) -> int:
        return self.A_N.shape[0]

    @property
    def n_pressure_modes(self) -> int:
        return self.B_N.shape[0]

    @property
    def n_velocity_dofs(self) -> int:
        return self.Z_u.shape[0]

    @property
    def n_pressure_dofs(self) -> int:
        return self.Z_p.shape[0]

    # -----------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------

    @property
    def u_base(self) -> np.ndarray:
        """First lifting function (base flow)."""
        return self.lifting_dofs[0]

    @property
    def u_control(self) -> np.ndarray:
        """Second lifting function (control), if K >= 2."""
        assert self.n_lifting >= 2, "No control lifting function (K < 2)"
        return self.lifting_dofs[1]

    # -----------------------------------------------------------------
    # Online assembly
    # -----------------------------------------------------------------

    def assemble_online(self, nu: float, thetas: list) -> OnlineSystem:
        """
        Assemble the full reduced system for given parameters.

        Everything returned is exact (no approximation). The solver
        only needs to add the DEIM quadratic-in-alpha part on top.

        Args:
            nu: Viscosity (1/Re)
            thetas: [theta_0, theta_1, ...] parameter weights
                    For pinball: [1.0, amp]

        Returns:
            OnlineSystem ready for the solver
        """
        assert len(thetas) == self.n_lifting, \
            f"Expected {self.n_lifting} thetas, got {len(thetas)}"

        K = self.n_lifting
        N_u = self.n_velocity_modes

        # Forcing: f_eff = f_N - nu * sum_k theta_k * A_lift[k]
        f_eff = self.f_N - nu * np.einsum('i,ij->j', thetas, self.A_lift)

        # Divergence: g_eff = g_N - sum_k theta_k * B_lift[k]
        g_eff = self.g_N - np.einsum('i,ij->j', thetas, self.B_lift)

        # Convection (if precomputed)
        if self.has_affine_convection:
            # c_0 = sum_{k,l} theta_k * theta_l * c_conv[k,l]
            c_0 = np.einsum('k,l,klj->j', thetas, thetas, self.c_conv)

            # C_lin = sum_k theta_k * (N_conv[k] + M_conv[k])
            C_lin = np.einsum('k,kij->ij', thetas, self.N_conv + self.M_conv)

            # Move constant to RHS
            f_eff = f_eff - c_0
        else:
            c_0 = np.zeros(N_u)
            C_lin = np.zeros((N_u, N_u))

        # Effective stiffness
        A_eff = nu * self.A_N + C_lin

        return OnlineSystem(
            A_eff=A_eff,
            B_N=self.B_N,
            f_eff=f_eff,
            g_eff=g_eff,
            C_lin=C_lin,
            c_0=c_0,
            nu=nu,
            thetas=list(thetas),
        )

    # -----------------------------------------------------------------
    # Legacy methods (backward compatibility with old solver code)
    # -----------------------------------------------------------------

    def get_adjusted_forcing(self, nu: float, amp: float) -> np.ndarray:
        """Legacy wrapper."""
        thetas = self._default_thetas(amp)
        return self.f_N - nu * np.einsum('i,ij->j', thetas, self.A_lift)

    def get_adjusted_divergence_rhs(self, amp: float) -> np.ndarray:
        """Legacy wrapper."""
        thetas = self._default_thetas(amp)
        return self.g_N - np.einsum('i,ij->j', thetas, self.B_lift)

    def _default_thetas(self, amp: float) -> np.ndarray:
        """Build default thetas for backward compatibility: [1.0, amp]."""
        if self.n_lifting == 1:
            return np.array([1.0])
        elif self.n_lifting == 2:
            return np.array([1.0, amp])
        else:
            raise ValueError(
                f"Cannot infer thetas for n_lifting={self.n_lifting}. "
                f"Use assemble_online(nu, thetas) instead."
            )


# =============================================================================
# ROM SOLUTION
# =============================================================================

@dataclass
class ROMSolution:
    """ROM solution coefficients and reconstruction data."""
    velocity_coeffs: np.ndarray
    pressure_coeffs: np.ndarray
    reynolds: float
    amplitude: float
    velocity_dofs: Optional[np.ndarray] = None
    pressure_dofs: Optional[np.ndarray] = None
    iterations: int = 0
    converged: bool = True
    residual_norm: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.velocity_coeffs.ndim == 1
        assert self.pressure_coeffs.ndim == 1

    @property
    def n_velocity_modes(self) -> int:
        return len(self.velocity_coeffs)

    @property
    def n_pressure_modes(self) -> int:
        return len(self.pressure_coeffs)


# =============================================================================
# I/O - POD BASIS
# =============================================================================

def save_pod_basis(basis: PODBasis, output_dir: str = "pod_basis"):
    os.makedirs(output_dir, exist_ok=True)

    arrays = {
        'velocity_modes': basis.velocity_modes,
        'pressure_modes': basis.pressure_modes,
        'velocity_eigenvalues': basis.velocity_eigenvalues,
        'pressure_eigenvalues': basis.pressure_eigenvalues,
    }
    if basis.supremizer_modes is not None:
        arrays['supremizer_modes'] = basis.supremizer_modes
    if basis.supremizer_eigenvalues is not None:
        arrays['supremizer_eigenvalues'] = basis.supremizer_eigenvalues

    np.savez_compressed(os.path.join(output_dir, "pod_basis.npz"), **arrays)

    json_data = {
        'n_velocity_modes': basis.n_velocity_modes,
        'n_pressure_modes': basis.n_pressure_modes,
        'n_supremizer_modes': basis.n_supremizer_modes,
        'n_velocity_dofs': basis.n_velocity_dofs,
        'n_pressure_dofs': basis.n_pressure_dofs,
        'inner_product_type': basis.inner_product_type,
        'velocity_energy_fraction': float(basis.velocity_energy_fraction()),
        'pressure_energy_fraction': float(basis.pressure_energy_fraction()),
    }
    json_data.update(basis.metadata)

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"POD basis saved to {output_dir}/")
    print(f"  Velocity: {basis.n_velocity_modes} modes ({basis.velocity_energy_fraction()*100:.10f}% energy)")
    print(f"  Pressure: {basis.n_pressure_modes} modes ({basis.pressure_energy_fraction()*100:.10f}% energy)")
    if basis.supremizer_modes is not None:
        print(f"  Supremizer: {basis.n_supremizer_modes} modes")


def load_pod_basis(input_dir: str) -> PODBasis:
    data = np.load(os.path.join(input_dir, "pod_basis.npz"))

    with open(os.path.join(input_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    inner_product_type = metadata.pop('inner_product_type', 'L2')
    for key in ['n_velocity_modes', 'n_pressure_modes', 'n_supremizer_modes',
                'n_velocity_dofs', 'n_pressure_dofs',
                'velocity_energy_fraction', 'pressure_energy_fraction']:
        metadata.pop(key, None)

    sup_modes = data['supremizer_modes'] if 'supremizer_modes' in data else None
    sup_eigs = data['supremizer_eigenvalues'] if 'supremizer_eigenvalues' in data else None

    basis = PODBasis(
        velocity_modes=data['velocity_modes'],
        pressure_modes=data['pressure_modes'],
        velocity_eigenvalues=data['velocity_eigenvalues'],
        pressure_eigenvalues=data['pressure_eigenvalues'],
        supremizer_modes=sup_modes,
        supremizer_eigenvalues=sup_eigs,
        inner_product_type=inner_product_type,
        metadata=metadata,
    )

    print(f"POD basis loaded from {input_dir}/")
    print(f"  Velocity: {basis.n_velocity_modes} modes")
    print(f"  Pressure: {basis.n_pressure_modes} modes")
    if basis.supremizer_modes is not None:
        print(f"  Supremizer: {basis.n_supremizer_modes} modes")

    return basis


# =============================================================================
# I/O - REDUCED OPERATORS
# =============================================================================

def save_reduced_operators(operators: ReducedOperators, output_dir: str = "rom_operators"):
    """
    Save reduced operators to disk.

    Creates:
        output_dir/
        ├── operators.npz     # Core + lifting
        ├── convection.npz    # Affine convection (if present)
        └── metadata.json
    """
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "operators.npz"),
        A_N=operators.A_N,
        B_N=operators.B_N,
        f_N=operators.f_N,
        g_N=operators.g_N,
        Z_u=operators.Z_u,
        Z_p=operators.Z_p,
        lifting_dofs=operators.lifting_dofs,
        A_lift=operators.A_lift,
        B_lift=operators.B_lift,
    )

    if operators.has_affine_convection:
        np.savez_compressed(
            os.path.join(output_dir, "convection.npz"),
            c_conv=operators.c_conv,
            N_conv=operators.N_conv,
            M_conv=operators.M_conv,
        )

    json_data = {
        'n_velocity_modes': operators.n_velocity_modes,
        'n_pressure_modes': operators.n_pressure_modes,
        'n_velocity_dofs': operators.n_velocity_dofs,
        'n_pressure_dofs': operators.n_pressure_dofs,
        'n_lifting': operators.n_lifting,
        'has_affine_convection': operators.has_affine_convection,
    }
    json_data.update(operators.metadata)

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Reduced operators saved to {output_dir}/")
    print(f"  N_velocity: {operators.n_velocity_modes}")
    print(f"  N_pressure: {operators.n_pressure_modes}")
    print(f"  N_lifting:  {operators.n_lifting}")
    if operators.has_affine_convection:
        print(f"  Affine convection: saved")


def load_reduced_operators(input_dir: str) -> ReducedOperators:
    data = np.load(os.path.join(input_dir, "operators.npz"))

    with open(os.path.join(input_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    n_lifting = metadata.pop('n_lifting')
    has_affine = metadata.pop('has_affine_convection', False)

    for key in ['n_velocity_modes', 'n_pressure_modes',
                'n_velocity_dofs', 'n_pressure_dofs']:
        metadata.pop(key, None)

    conv_kwargs = {}
    convection_file = os.path.join(input_dir, "convection.npz")
    if has_affine and os.path.exists(convection_file):
        conv_data = np.load(convection_file)
        conv_kwargs = {
            'c_conv': conv_data['c_conv'],
            'N_conv': conv_data['N_conv'],
            'M_conv': conv_data['M_conv'],
        }

    operators = ReducedOperators(
        A_N=data['A_N'],
        B_N=data['B_N'],
        f_N=data['f_N'],
        g_N=data['g_N'],
        Z_u=data['Z_u'],
        Z_p=data['Z_p'],
        n_lifting=n_lifting,
        lifting_dofs=data['lifting_dofs'],
        A_lift=data['A_lift'],
        B_lift=data['B_lift'],
        metadata=metadata,
        **conv_kwargs,
    )

    print(f"Reduced operators loaded from {input_dir}/")
    print(f"  N_velocity: {operators.n_velocity_modes}")
    print(f"  N_pressure: {operators.n_pressure_modes}")
    print(f"  N_lifting:  {operators.n_lifting}")
    if operators.has_affine_convection:
        print(f"  Affine convection: loaded")

    return operators


# =============================================================================
# I/O - ROM SOLUTION
# =============================================================================

def save_rom_solution(solution: ROMSolution, output_dir: str = "rom_solution",
                      save_full_dofs: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    arrays = {
        'velocity_coeffs': solution.velocity_coeffs,
        'pressure_coeffs': solution.pressure_coeffs,
    }
    if save_full_dofs:
        if solution.velocity_dofs is not None:
            arrays['velocity_dofs'] = solution.velocity_dofs
        if solution.pressure_dofs is not None:
            arrays['pressure_dofs'] = solution.pressure_dofs

    np.savez_compressed(os.path.join(output_dir, "solution.npz"), **arrays)

    json_data = {
        'reynolds': solution.reynolds,
        'amplitude': solution.amplitude,
        'n_velocity_modes': solution.n_velocity_modes,
        'n_pressure_modes': solution.n_pressure_modes,
        'iterations': solution.iterations,
        'converged': solution.converged,
        'residual_norm': solution.residual_norm,
        'has_full_dofs': solution.velocity_dofs is not None,
    }
    json_data.update(solution.metadata)

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"ROM solution saved to {output_dir}/")
    print(f"  Re={solution.reynolds}, amp={solution.amplitude}")
    print(f"  Converged: {solution.converged} ({solution.iterations} iterations)")


def load_rom_solution(input_dir: str) -> ROMSolution:
    data = np.load(os.path.join(input_dir, "solution.npz"))

    with open(os.path.join(input_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    reynolds = metadata.pop('reynolds')
    amplitude = metadata.pop('amplitude')
    iterations = metadata.pop('iterations', 0)
    converged = metadata.pop('converged', True)
    residual_norm = metadata.pop('residual_norm', 0.0)

    for key in ['n_velocity_modes', 'n_pressure_modes', 'has_full_dofs']:
        metadata.pop(key, None)

    return ROMSolution(
        velocity_coeffs=data['velocity_coeffs'],
        pressure_coeffs=data['pressure_coeffs'],
        reynolds=reynolds,
        amplitude=amplitude,
        velocity_dofs=data.get('velocity_dofs'),
        pressure_dofs=data.get('pressure_dofs'),
        iterations=iterations,
        converged=converged,
        residual_norm=residual_norm,
        metadata=metadata,
    )


# =============================================================================
# BATCH I/O
# =============================================================================

def save_rom_solutions_batch(solutions: List[ROMSolution],
                             output_dir: str = "rom_solutions",
                             save_full_dofs: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    n_solutions = len(solutions)
    arrays = {
        'velocity_coeffs': np.vstack([s.velocity_coeffs for s in solutions]),
        'pressure_coeffs': np.vstack([s.pressure_coeffs for s in solutions]),
        'reynolds': np.array([s.reynolds for s in solutions]),
        'amplitude': np.array([s.amplitude for s in solutions]),
        'iterations': np.array([s.iterations for s in solutions]),
        'converged': np.array([s.converged for s in solutions]),
        'residual_norms': np.array([s.residual_norm for s in solutions]),
    }

    if save_full_dofs and solutions[0].velocity_dofs is not None:
        arrays['velocity_dofs'] = np.vstack([s.velocity_dofs for s in solutions])
        arrays['pressure_dofs'] = np.vstack([s.pressure_dofs for s in solutions])

    np.savez_compressed(os.path.join(output_dir, "solutions_batch.npz"), **arrays)

    json_data = {
        'n_solutions': n_solutions,
        'n_velocity_modes': solutions[0].n_velocity_modes,
        'n_pressure_modes': solutions[0].n_pressure_modes,
        'reynolds_range': [float(arrays['reynolds'].min()), float(arrays['reynolds'].max())],
        'amplitude_range': [float(arrays['amplitude'].min()), float(arrays['amplitude'].max())],
        'all_converged': bool(np.all(arrays['converged'])),
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved {n_solutions} ROM solutions to {output_dir}/")


def load_rom_solutions_batch(input_dir: str) -> List[ROMSolution]:
    data = np.load(os.path.join(input_dir, "solutions_batch.npz"))

    with open(os.path.join(input_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    n_solutions = metadata['n_solutions']
    has_full_dofs = 'velocity_dofs' in data

    solutions = []
    for i in range(n_solutions):
        solutions.append(ROMSolution(
            velocity_coeffs=data['velocity_coeffs'][i],
            pressure_coeffs=data['pressure_coeffs'][i],
            reynolds=float(data['reynolds'][i]),
            amplitude=float(data['amplitude'][i]),
            velocity_dofs=data['velocity_dofs'][i] if has_full_dofs else None,
            pressure_dofs=data['pressure_dofs'][i] if has_full_dofs else None,
            iterations=int(data['iterations'][i]),
            converged=bool(data['converged'][i]),
            residual_norm=float(data['residual_norms'][i]),
        ))

    print(f"Loaded {n_solutions} ROM solutions from {input_dir}/")
    return solutions