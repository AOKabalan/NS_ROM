# online_operators.py
"""
DEIM Online Operators for ROM acceleration.

Provides precomputed operators for efficient online ROM evaluation:
- Reduced convective term via DEIM
- Reduced Jacobian via MDEIM
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .deim import deim_algorithm, load_basis


def csr_flat_to_coords(flat_indices, indptr, col_indices):
    """Convert flat CSR data indices to (row, col) coordinates."""
    rows = []
    cols = []
    for idx in flat_indices:
        # Binary search to find which row this data index belongs to
        row = np.searchsorted(indptr[1:], idx, side='right')
        col = col_indices[idx]
        rows.append(row)
        cols.append(col)
    return np.array(rows), np.array(cols)


def dofs_to_nnz(dofs, indptr, col_indices, col_filter=None):
    """
    Map residual (row) DOF indices to the flat CSR data indices of their
    Jacobian rows. This is the inverse-direction translation of
    csr_flat_to_coords: one DOF -> many nnz entries (its whole row stencil).

    Args:
        dofs: iterable of row DOF indices
        indptr: CSR indptr of the Jacobian sparsity pattern
        col_indices: CSR column indices of the Jacobian sparsity pattern
        col_filter: optional array of DOF indices; if given, keep only nnz
            entries (i, j) whose column j is in this set. Filtering to the
            residual sample set keeps the count modest and samples exactly
            the sub-block coupling sampled equations to sampled unknowns.

    Returns:
        (n_entries,) int64 array of flat CSR data indices (unsorted, may
        contain duplicates across rows only if dofs has duplicates).
    """
    dofs = np.asarray(dofs, dtype=np.int64)
    if col_filter is not None:
        col_filter = np.unique(np.asarray(col_filter, dtype=np.int64))
    chunks = []
    for i in dofs:
        row_nnz = np.arange(indptr[i], indptr[i + 1], dtype=np.int64)
        if col_filter is not None:
            keep = np.isin(col_indices[row_nnz], col_filter, assume_unique=False)
            row_nnz = row_nnz[keep]
        chunks.append(row_nnz)
    if not chunks:
        return np.array([], dtype=np.int64)
    return np.concatenate(chunks)


@dataclass
class DEIMOnlineOperators:
    """
    All precomputed data needed for online ROM solver.

    Attributes:
        indices_F: (m_F,) DEIM interpolation indices for convective term (flat CSR indices)
        indices_F_rows: (m_F,) row coordinates of DEIM interpolation points
        indices_F_cols: (m_F,) column coordinates of DEIM interpolation points
        V_T_B_F: (r, m_F) precomputed matrix for reduced convective
        indices_J: (m_J,) MDEIM interpolation indices for Jacobian (flat CSR indices)
        indices_J_rows: (m_J,) row coordinates of MDEIM interpolation points
        indices_J_cols: (m_J,) column coordinates of MDEIM interpolation points
        U_J_P_inv: (m_J, m_J) inverse of sampled Jacobian basis
        J_r_modes: (m_J, r, r) precomputed reduced Jacobian modes
        metadata: dict with basis_path, m_F, m_J, n_rom_modes
    """
    # Convective
    indices_F: np.ndarray           # (m_F,) flat CSR indices
    indices_F_rows: np.ndarray      # (m_F,) row coordinates
    indices_F_cols: np.ndarray      # (m_F,) column coordinates
    V_T_B_F: np.ndarray             # (r, m_F)

    # Jacobian
    indices_J: np.ndarray           # (m_J,) flat CSR indices
    indices_J_rows: np.ndarray      # (m_J,) row coordinates
    indices_J_cols: np.ndarray      # (m_J,) column coordinates
    U_J_P_inv: np.ndarray           # (m_J, m_J)
    J_r_modes: np.ndarray           # (m_J, r, r)

    # Metadata for cache validation
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def m_F(self) -> int:
        """Number of DEIM modes for convective term."""
        return len(self.indices_F)

    @property
    def m_J(self) -> int:
        """Number of MDEIM modes for Jacobian."""
        return len(self.indices_J)

    @property
    def r(self) -> int:
        """Number of ROM velocity modes."""
        return self.V_T_B_F.shape[0]

    def reduced_convective(self, F_full: np.ndarray) -> np.ndarray:
        """
        Compute reduced convective term via DEIM.

        Args:
            F_full: (n_dofs,) full convective vector

        Returns:
            F_r: (r,) reduced convective vector
        """
        return self.V_T_B_F @ F_full[self.indices_F]

    def reduced_jacobian(self, J_csr_data: np.ndarray) -> np.ndarray:
        """
        Compute reduced Jacobian via MDEIM.

        Args:
            J_csr_data: (nnz,) CSR data array of full Jacobian

        Returns:
            J_r: (r, r) reduced Jacobian matrix
        """
        c = self.U_J_P_inv @ J_csr_data[self.indices_J]
        return np.tensordot(c, self.J_r_modes, axes=([0], [0]))

    def save(
        self,
        path: str | Path,
        basis_path: Optional[str] = None,
    ):
        """
        Save operators to file.

        Args:
            path: Output path (.npz)
            basis_path: Path to DEIM basis file (stored as metadata)
        """
        # Update metadata
        save_metadata = {
            'basis_path': basis_path if basis_path else self.metadata.get('basis_path', ''),
            'm_F': self.m_F,
            'm_J': self.m_J,
            'n_rom_modes': self.r,
        }
        # Optional extended metadata (mode counts vs sample counts, flags)
        extra_meta = {
            f'meta_{k}': self.metadata[k]
            for k in ('n_modes_F', 'n_modes_J', 'n_deim_points_F',
                      'n_mdeim_points_J', 'oversample_residual',
                      'oversample_jacobian')
            if k in self.metadata
        }

        np.savez_compressed(
            path,
            **extra_meta,
            # Convective operator data
            indices_F=self.indices_F,
            indices_F_rows=self.indices_F_rows,
            indices_F_cols=self.indices_F_cols,
            V_T_B_F=self.V_T_B_F,
            # Jacobian operator data
            indices_J=self.indices_J,
            indices_J_rows=self.indices_J_rows,
            indices_J_cols=self.indices_J_cols,
            U_J_P_inv=self.U_J_P_inv,
            J_r_modes=self.J_r_modes,
            # Metadata
            meta_basis_path=save_metadata['basis_path'],
            meta_m_F=save_metadata['m_F'],
            meta_m_J=save_metadata['m_J'],
            meta_n_rom_modes=save_metadata['n_rom_modes'],
        )

        print(f"Saved DEIMOnlineOperators to {path}")
        print(f"  r={self.r}, m_F={self.m_F}, m_J={self.m_J}")

    @classmethod
    def load(cls, path: str | Path) -> "DEIMOnlineOperators":
        """
        Load operators from file.

        Args:
            path: Path to saved operators (.npz)

        Returns:
            DEIMOnlineOperators instance
        """
        data = np.load(path)

        metadata = {
            'basis_path': str(data['meta_basis_path']),
            'm_F': int(data['meta_m_F']),
            'm_J': int(data['meta_m_J']),
            'n_rom_modes': int(data['meta_n_rom_modes']),
        }
        for k in ('n_modes_F', 'n_modes_J', 'n_deim_points_F',
                  'n_mdeim_points_J', 'oversample_residual',
                  'oversample_jacobian'):
            key = f'meta_{k}'
            if key in data.files:
                metadata[k] = data[key].item()

        return cls(
            indices_F=data['indices_F'],
            indices_F_rows=data['indices_F_rows'],
            indices_F_cols=data['indices_F_cols'],
            V_T_B_F=data['V_T_B_F'],
            indices_J=data['indices_J'],
            indices_J_rows=data['indices_J_rows'],
            indices_J_cols=data['indices_J_cols'],
            U_J_P_inv=data['U_J_P_inv'],
            J_r_modes=data['J_r_modes'],
            metadata=metadata,
        )


def load_ops_metadata(path: str | Path) -> dict:
    """
    Load only metadata from saved operators (for quick cache checks).

    Args:
        path: Path to saved operators (.npz)

    Returns:
        Dictionary with basis_path, m_F, m_J, n_rom_modes
    """
    data = np.load(path)

    return {
        'basis_path': str(data['meta_basis_path']),
        'm_F': int(data['meta_m_F']),
        'm_J': int(data['meta_m_J']),
        'n_rom_modes': int(data['meta_n_rom_modes']),
    }


def build_deim_online_operators_old(
    basis_path: str | Path,
    V: np.ndarray,
    m_F: int,
    m_J: int,
) -> DEIMOnlineOperators:
    """
    Build online operators from saved DEIM basis.

    Args:
        basis_path: Path to saved DEIM basis (.npz)
        V: (n_dofs, r) velocity POD basis (enriched with supremizers)
        m_F: Number of DEIM modes for convective term
        m_J: Number of MDEIM modes for Jacobian

    Returns:
        DEIMOnlineOperators ready for online evaluation
    """
    basis = load_basis(basis_path)
    n_dofs = basis['n_dofs']
    r = V.shape[1]

    print(f"Building DEIM online operators:")
    print(f"  ROM modes (r): {r}")

    # Get available modes
    max_m_F = basis['modes_F'].shape[1]
    max_m_J = basis['modes_J'].shape[1]

    print(f"  Available F modes: {max_m_F}")
    print(f"  Available J modes: {max_m_J}")
    print(f"  Requested: m_F={m_F}, m_J={m_J}")

    # Check effective rank of F modes (detect near-zero modes)
    U_F_candidate = basis['modes_F'][:, :min(m_F, max_m_F)]
    mode_norms_F = np.linalg.norm(U_F_candidate, axis=0)
    effective_m_F = np.sum(mode_norms_F > 1e-10)

    if effective_m_F < m_F:
        print(f"  ⚠ F modes: only {effective_m_F} have non-zero norm (requested {m_F})")
        m_F = effective_m_F

    # Check effective rank of J modes
    U_J_candidate = basis['modes_J'][:, :min(m_J, max_m_J)]
    mode_norms_J = np.linalg.norm(U_J_candidate, axis=0)
    effective_m_J = np.sum(mode_norms_J > 1e-10)

    if effective_m_J < m_J:
        print(f"  ⚠ J modes: only {effective_m_J} have non-zero norm (requested {m_J})")
        m_J = effective_m_J

    # Cap at available modes
    if m_F > max_m_F:
        print(f"  ⚠ Capping m_F: {m_F} → {max_m_F}")
        m_F = max_m_F
    if m_J > max_m_J:
        print(f"  ⚠ Capping m_J: {m_J} → {max_m_J}")
        m_J = max_m_J

    # Ensure at least 1 mode
    m_F = max(1, m_F)
    m_J = max(1, m_J)

    print(f"  Final: m_F={m_F}, m_J={m_J}")

    # Get CSR structure for coordinate conversion
    indptr = basis['indptr']
    col_indices = basis['col_indices']

    # --- Convective: truncate, DEIM, build V_T_B_F ---
    print(f"\n  Building convective operators...")
    U_F = basis['modes_F'][:, :m_F]
    indices_F = deim_algorithm(U_F)

    # Convert flat indices to (row, col) coordinates
    indices_F_rows, indices_F_cols = csr_flat_to_coords(indices_F, indptr, col_indices)
    print(f"    DEIM selected {len(indices_F)} interpolation points for F")
    print(f"    Sample coordinates (row, col):")
    for i in range(min(3, len(indices_F))):
        print(f"      Point {i}: ({indices_F_rows[i]}, {indices_F_cols[i]})")

    U_F_P = U_F[indices_F, :]
    cond_F = np.linalg.cond(U_F_P)
    print(f"    Condition number: {cond_F:.2e}")

    if cond_F > 1e12:
        print(f"    ⚠ Warning: poorly conditioned, using pseudoinverse")
        U_F_P_inv = np.linalg.pinv(U_F_P)
    else:
        U_F_P_inv = np.linalg.inv(U_F_P)

    V_T_B_F = V.T @ (U_F @ U_F_P_inv)
    print(f"    V_T_B_F shape: {V_T_B_F.shape}")

    # --- Jacobian: truncate, DEIM, build J_r_modes ---
    print(f"\n  Building Jacobian operators...")
    U_J = basis['modes_J'][:, :m_J]
    indices_J = deim_algorithm(U_J)

    # Convert flat indices to (row, col) coordinates
    indices_J_rows, indices_J_cols = csr_flat_to_coords(indices_J, indptr, col_indices)
    print(f"    DEIM selected {len(indices_J)} interpolation points for J")
    print(f"    Sample coordinates (row, col):")
    for i in range(min(3, len(indices_J))):
        print(f"      Point {i}: ({indices_J_rows[i]}, {indices_J_cols[i]})")

    U_J_P = U_J[indices_J, :]
    cond_J = np.linalg.cond(U_J_P)
    print(f"    Condition number: {cond_J:.2e}")

    if cond_J > 1e12:
        print(f"    ⚠ Warning: poorly conditioned, using pseudoinverse")
        U_J_P_inv = np.linalg.pinv(U_J_P)
    else:
        U_J_P_inv = np.linalg.inv(U_J_P)

    J_r_modes = np.zeros((m_J, r, r))
    for k in range(m_J):
        if (k + 1) % 20 == 0 or k == 0:
            print(f"    Mode {k+1}/{m_J}")
        J_k = sp.csr_matrix(
            (U_J[:, k], col_indices, indptr),
            shape=(n_dofs, n_dofs)
        )
        J_r_modes[k] = V.T @ (J_k @ V)

    print(f"    J_r_modes shape: {J_r_modes.shape}")

    metadata = {
        'basis_path': str(basis_path),
        'm_F': m_F,
        'm_J': m_J,
        'n_rom_modes': r,
    }

    return DEIMOnlineOperators(
        indices_F=indices_F,
        indices_F_rows=indices_F_rows,
        indices_F_cols=indices_F_cols,
        V_T_B_F=V_T_B_F,
        indices_J=indices_J,
        indices_J_rows=indices_J_rows,
        indices_J_cols=indices_J_cols,
        U_J_P_inv=U_J_P_inv,
        J_r_modes=J_r_modes,
        metadata=metadata,
    )


# def build_deim_online_operators(
#     basis_path: str | Path,
#     V: np.ndarray,
#     m_F: int,
#     m_J: int,
# ) -> DEIMOnlineOperators:
#     """
#     Build online operators from saved DEIM basis.

#     Args:
#         basis_path: Path to saved DEIM basis (.npz)
#         V: (n_dofs, r) velocity POD basis (enriched with supremizers)
#         m_F: Number of DEIM modes for convective term
#         m_J: Number of MDEIM modes for Jacobian

#     Returns:
#         DEIMOnlineOperators ready for online evaluation
#     """
#     basis = load_basis(basis_path)
#     n_dofs = basis['n_dofs']
#     r = V.shape[1]

#     print(f"Building DEIM online operators:")
#     print(f"  ROM modes (r): {r}")
#     print(f"  Convective DEIM modes (m_F): {m_F}")
#     print(f"  Jacobian MDEIM modes (m_J): {m_J}")

#     # Check truncation bounds
#     max_m_F = basis['modes_F'].shape[1]
#     max_m_J = basis['modes_J'].shape[1]

#     if m_F > max_m_F:
#         raise ValueError(f"m_F={m_F} exceeds available modes ({max_m_F})")
#     if m_J > max_m_J:
#         raise ValueError(f"m_J={m_J} exceeds available modes ({max_m_J})")

#     # --- Convective: truncate, DEIM, build V_T_B_F ---
#     print(f"\n  Building convective operators...")
#     U_F = basis['modes_F'][:, :m_F]
#     indices_F = deim_algorithm(U_F)
#     U_F_P_inv = np.linalg.inv(U_F[indices_F, :])
#     V_T_B_F = V.T @ (U_F @ U_F_P_inv)  # (r, m_F)
#     print(f"    V_T_B_F shape: {V_T_B_F.shape}")

#     # --- Jacobian: truncate, DEIM, build J_r_modes ---
#     print(f"\n  Building Jacobian operators...")
#     U_J = basis['modes_J'][:, :m_J]
#     indices_J = deim_algorithm(U_J)
#     U_J_P_inv = np.linalg.inv(U_J[indices_J, :])

#     indptr = basis['indptr']
#     col_indices = basis['col_indices']

#     J_r_modes = np.zeros((m_J, r, r))
#     for k in range(m_J):
#         if (k + 1) % 20 == 0 or k == 0:
#             print(f"    Mode {k+1}/{m_J}")
#         J_k = sp.csr_matrix(
#             (U_J[:, k], col_indices, indptr),
#             shape=(n_dofs, n_dofs)
#         )
#         J_r_modes[k] = V.T @ (J_k @ V)

#     print(f"    J_r_modes shape: {J_r_modes.shape}")

#     # Build metadata
#     metadata = {
#         'basis_path': str(basis_path),
#         'm_F': m_F,
#         'm_J': m_J,
#         'n_rom_modes': r,
#     }

#     return DEIMOnlineOperators(
#         indices_F=indices_F,
#         V_T_B_F=V_T_B_F,
#         indices_J=indices_J,
#         U_J_P_inv=U_J_P_inv,
#         J_r_modes=J_r_modes,
#         metadata=metadata,
#     )


def build_deim_online_operators(
    basis_path,
    V: np.ndarray,
    m_F: int,
    m_J: int,
    oversample_residual: bool = False,
    include_jacobian_cols: bool = False,
    oversample_jacobian: bool = False,
    jacobian_filter_cols: bool = False,
) -> DEIMOnlineOperators:
    """
    Build online operators from saved DEIM basis, with optional oversampled
    (least-squares) reconstruction of the residual and/or the Jacobian, so
    that both operators are anchored on a shared point set.

    Args:
        basis_path: Path to saved DEIM basis (.npz)
        V: (n_dofs, r) velocity POD basis (enriched with supremizers)
        m_F: Number of DEIM modes for convective term
        m_J: Number of MDEIM modes for Jacobian
        oversample_residual: if True, sample the residual on the union of the
            DEIM points and the MDEIM row/col DOFs, reconstruct by least squares.
            if False, original square DEIM (exact reproduction of old behavior).
        include_jacobian_cols: if True, add MDEIM column DOFs as well as rows to
            the residual oversample set. Rows are the residual (test-function)
            DOFs; cols are the trial DOFs the convective Jacobian reads. Both are
            velocity DOFs in the convective region. Set False for rows-only.
        oversample_jacobian: if True, extend the MDEIM sample set with the nnz
            entries of the Jacobian rows belonging to the residual DEIM points
            (the symmetric direction: J is anchored where F is sampled), and
            reconstruct the Jacobian coefficients by least squares. The mode
            count m_J and J_r_modes are unchanged; only the sample set grows.
        jacobian_filter_cols: if True (and oversample_jacobian), keep only nnz
            entries (i, j) whose column j lies in the residual sample set, i.e.
            sample the sub-block coupling sampled equations to sampled unknowns.
            If False, take the full row stencils (more entries, denser fit).
    """
    basis = load_basis(basis_path)
    n_dofs = basis['n_dofs']
    r = V.shape[1]

    print(f"Building DEIM online operators:")
    print(f"  ROM modes (r): {r}")

    max_m_F = basis['modes_F'].shape[1]
    max_m_J = basis['modes_J'].shape[1]
    print(f"  Available F modes: {max_m_F}, J modes: {max_m_J}")
    print(f"  Requested: m_F={m_F}, m_J={m_J}")

    # --- effective-rank / cap guards (unchanged from original) ---
    U_F_candidate = basis['modes_F'][:, :min(m_F, max_m_F)]
    effective_m_F = int(np.sum(np.linalg.norm(U_F_candidate, axis=0) > 1e-10))
    if effective_m_F < m_F:
        print(f"  ⚠ F modes: only {effective_m_F} non-zero (requested {m_F})")
        m_F = effective_m_F
    U_J_candidate = basis['modes_J'][:, :min(m_J, max_m_J)]
    effective_m_J = int(np.sum(np.linalg.norm(U_J_candidate, axis=0) > 1e-10))
    if effective_m_J < m_J:
        print(f"  ⚠ J modes: only {effective_m_J} non-zero (requested {m_J})")
        m_J = effective_m_J
    m_F = max(1, min(m_F, max_m_F))
    m_J = max(1, min(m_J, max_m_J))
    print(f"  Final: m_F={m_F}, m_J={m_J}")

    indptr = basis['indptr']
    col_indices = basis['col_indices']

    # =====================================================================
    # Greedy selections FIRST (both), so the residual oversample set can use
    # the MDEIM row/col DOFs.
    # =====================================================================
    U_F = basis['modes_F'][:, :m_F]
    deim_pts_F = deim_algorithm(U_F)                 # (m_F,) residual VECTOR DOFs

    U_J = basis['modes_J'][:, :m_J]
    mdeim_pts_J = deim_algorithm(U_J)                # (m_J,) Jacobian CSR-data idx
    indices_J_rows, indices_J_cols = csr_flat_to_coords(
        mdeim_pts_J, indptr, col_indices)           # DOF indices in [0, n_dofs)

    # =====================================================================
    # Convective (residual) operator
    # =====================================================================
    print(f"\n  Building convective operators...")

    if oversample_residual:
        extra = indices_J_rows
        if include_jacobian_cols:
            extra = np.concatenate([indices_J_rows, indices_J_cols])
        # union of residual DEIM points and MDEIM row/(col) DOFs, all in
        # residual-vector index space; sorted+unique for a clean sample set.
        oversample_pts = np.unique(
            np.concatenate([deim_pts_F, extra]).astype(np.int64)
        )
        indices_F = oversample_pts
        print(f"    Oversampled residual: {len(deim_pts_F)} DEIM points "
              f"+ MDEIM {'rows+cols' if include_jacobian_cols else 'rows'} "
              f"-> {len(indices_F)} union points")
    else:
        indices_F = deim_pts_F
        print(f"    Square DEIM: {len(indices_F)} points")

    # Vestigial for the residual path (SubMeshDEIM uses only indices_F). Kept for
    # struct/shape compatibility with save/load. For the residual, the "row" IS
    # the DOF; cols are not meaningful for a vector sample.
    indices_F_rows = indices_F.copy()
    indices_F_cols = np.zeros_like(indices_F)

    U_F_P = U_F[indices_F, :]                         # (n_over, m_F), tall if oversampled
    cond_F = np.linalg.cond(U_F_P)
    print(f"    Sample matrix U_F_P shape: {U_F_P.shape}, cond: {cond_F:.2e}")

    if oversample_residual or U_F_P.shape[0] != U_F_P.shape[1] or cond_F > 1e12:
        # least-squares reconstruction (required when tall; safe when square)
        U_F_P_inv = np.linalg.pinv(U_F_P)            # (m_F, n_over)
        if oversample_residual:
            # report the least-squares fit quality of the basis onto the points
            recon = U_F_P @ (U_F_P_inv @ U_F_P)
            print(f"    LSQ self-consistency ||U_F_P - U_F_P (P^+ P)||/||U_F_P|| "
                  f"= {np.linalg.norm(U_F_P - recon)/np.linalg.norm(U_F_P):.2e}")
    else:
        U_F_P_inv = np.linalg.inv(U_F_P)

    # associativity makes this identical to (V.T @ U_F) @ U_F_P_inv;
    # shape is (r, n_over)
    V_T_B_F = V.T @ (U_F @ U_F_P_inv)
    print(f"    V_T_B_F shape: {V_T_B_F.shape}")

    # =====================================================================
    # Jacobian (MDEIM) operator -- optionally oversampled on the DEIM rows
    # =====================================================================
    print(f"\n  Building Jacobian operators...")

    if oversample_jacobian:
        # For each residual DEIM point i, add (a subset of) the nnz entries of
        # Jacobian row i. Filtering columns to the residual sample set keeps
        # the count modest and makes the two sample sets mutually consistent
        # in one pass (rows are deim_pts_F ⊂ indices_F; cols ⊂ indices_F).
        col_filter = indices_F if jacobian_filter_cols else None
        extra_nnz = dofs_to_nnz(deim_pts_F, indptr, col_indices, col_filter)
        indices_J = np.unique(
            np.concatenate([mdeim_pts_J, extra_nnz]).astype(np.int64)
        )
        # Refresh coordinates for the expanded set: these feed the submesh
        # construction, which must now cover the added rows' cell patches
        # (already covered if the residual submesh includes deim_pts_F).
        indices_J_rows, indices_J_cols = csr_flat_to_coords(
            indices_J, indptr, col_indices)
        print(f"    Oversampled Jacobian: {len(mdeim_pts_J)} MDEIM points "
              f"+ {len(extra_nnz)} row-stencil entries "
              f"({'col-filtered' if jacobian_filter_cols else 'full rows'}) "
              f"-> {len(indices_J)} union points")
    else:
        indices_J = mdeim_pts_J
        print(f"    Square MDEIM: {len(indices_J)} points")

    U_J_P = U_J[indices_J, :]                        # (n_sample_J, m_J)
    cond_J = np.linalg.cond(U_J_P)
    print(f"    Sample matrix U_J_P shape: {U_J_P.shape}, cond: {cond_J:.2e}")
    if oversample_jacobian or U_J_P.shape[0] != U_J_P.shape[1] or cond_J > 1e12:
        # least-squares reconstruction (required when tall; safe when square)
        U_J_P_inv = np.linalg.pinv(U_J_P)            # (m_J, n_sample_J)
    else:
        U_J_P_inv = np.linalg.inv(U_J_P)

    J_r_modes = np.zeros((m_J, r, r))
    J_k = sp.csr_matrix((np.empty(len(col_indices)), col_indices, indptr),
                        shape=(n_dofs, n_dofs))          # pattern built ONCE
    for k in range(m_J):
        if (k + 1) % 20 == 0 or k == 0:
            print(f"    Mode {k+1}/{m_J}")
        J_k.data = np.ascontiguousarray(U_J[:, k])       # swap values only
        J_r_modes[k] = V.T @ (J_k @ V)
    # J_r_modes = np.zeros((m_J, r, r))
    # for k in range(m_J):
    #     if (k + 1) % 20 == 0 or k == 0:
    #         print(f"    Mode {k+1}/{m_J}")
    #     J_k = sp.csr_matrix(
    #         (U_J[:, k], col_indices, indptr), shape=(n_dofs, n_dofs)
    #     )
    #     J_r_modes[k] = V.T @ (J_k @ V)
    print(f"    J_r_modes shape: {J_r_modes.shape}")

    metadata = {
        'basis_path': str(basis_path),
        # sample counts (what the online solver reads/assembles)
        'm_F': int(len(indices_F)),
        'm_J': int(len(indices_J)),
        # mode counts (what convergence studies should sweep)
        'n_modes_F': int(m_F),
        'n_modes_J': int(U_J.shape[1]),
        'n_deim_points_F': int(len(deim_pts_F)),
        'n_mdeim_points_J': int(len(mdeim_pts_J)),
        'n_rom_modes': int(r),
        'oversample_residual': bool(oversample_residual),
        'oversample_jacobian': bool(oversample_jacobian),
    }

    return DEIMOnlineOperators(
        indices_F=indices_F,
        indices_F_rows=indices_F_rows,
        indices_F_cols=indices_F_cols,
        V_T_B_F=V_T_B_F,
        indices_J=indices_J,
        indices_J_rows=indices_J_rows,
        indices_J_cols=indices_J_cols,
        U_J_P_inv=U_J_P_inv,
        J_r_modes=J_r_modes,
        metadata=metadata,
    )
