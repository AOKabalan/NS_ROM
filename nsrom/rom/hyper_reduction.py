"""
Submesh-based hyper-reduction for DEIM ROM.

Instead of assembling into the full DOF vector and discarding 99.9%,
we build a small submesh containing only DEIM-relevant cells and
assemble into that.

Usage:
    from hyper_reduction import SubMeshDEIM

    # Setup (once before Newton iterations)
    sm = SubMeshDEIM(problem.velocity_space, deim_ops)

    # In residual callback:
    sm.transfer_to_submesh(u_work)
    F_sub = assemble(inner(grad(sm.u_sub) * sm.u_sub, sm.v_test) * dx)
    conv_r = sm.extract_residual(F_sub)

    # In Jacobian callback:
    J_sub = assemble(..., mat_type='aij')
    _, _, J_data = J_sub.M.handle.getValuesCSR()
    conv_J = sm.extract_jacobian(J_data)
"""

import numpy as np
from firedrake import (
    Function, TrialFunction, TestFunction,
    FunctionSpace, VectorFunctionSpace,
    inner, grad, dx, assemble,
    SpatialCoordinate, Submesh,
)


def _nodes_to_cells(cell_node_map, nodes):
    """Find all cells that contain any of the given nodes."""
    cells = set()
    node_set = set(int(n) for n in nodes)
    for cell_idx in range(cell_node_map.shape[0]):
        if node_set & set(int(n) for n in cell_node_map[cell_idx]):
            cells.add(cell_idx)
    return cells


def _node_pairs_to_cells(cell_node_map, row_nodes, col_nodes):
    """Find all cells that contain both a row node and a col node."""
    cells = set()
    for rn, cn in zip(row_nodes, col_nodes):
        rn, cn = int(rn), int(cn)
        for cell_idx in range(cell_node_map.shape[0]):
            cell_nodes = set(int(n) for n in cell_node_map[cell_idx])
            if rn in cell_nodes and cn in cell_nodes:
                cells.add(cell_idx)
    return cells


class SubMeshDEIM:
    """
    Submesh infrastructure for DEIM/MDEIM hyper-reduction.

    Encapsulates:
        - Submesh creation from DEIM cell indices
        - DOF remapping (full mesh ↔ submesh)
        - DEIM/MDEIM index translation
        - Velocity transfer and extraction

    Attributes:
        V_sub:      VectorFunctionSpace on submesh
        u_sub:      Function(V_sub) — workspace for submesh velocity
        v_test:     TestFunction(V_sub)
        u_trial:    TrialFunction(V_sub)
        deim_ops:   Reference to the DEIMOnlineOperators
    """

    def __init__(self, V_full, deim_ops):
        """
        Build submesh and all mappings.

        Args:
            V_full: Full-mesh VectorFunctionSpace
            deim_ops: DEIMOnlineOperators with indices_F, indices_J,
                      indices_J_rows, indices_J_cols, V_T_B_F,
                      U_J_P_inv, J_r_modes
        """
        self.V_full = V_full
        self.deim_ops = deim_ops
        mesh = V_full.mesh()
        value_size = V_full.value_size

        cell_node = V_full.cell_node_map().values

        # ---- Identify reduced cells ----
        res_nodes = np.unique(deim_ops.indices_F // value_size)
        cells_res = _nodes_to_cells(cell_node, res_nodes)

        jac_row_nodes = deim_ops.indices_J_rows // value_size
        jac_col_nodes = deim_ops.indices_J_cols // value_size
        cells_jac = _node_pairs_to_cells(cell_node, jac_row_nodes, jac_col_nodes)

        reduced_cells = np.array(
            sorted(cells_res | cells_jac), dtype=np.int32
        )

        n_cells = cell_node.shape[0]
        print(f"\n  SubMeshDEIM: building submesh...")
        print(f"    Full mesh: {n_cells} cells, {V_full.dim()} DOFs")
        print(f"    Reduced cells: {len(reduced_cells)} "
              f"({100*len(reduced_cells)/n_cells:.1f}%)")

        # ---- Create submesh via DG0 indicator ----
        DG0 = FunctionSpace(mesh, "DG", 0)
        indicator = Function(DG0)
        cnm_dg0 = DG0.cell_node_map().values.flatten()
        for cell_idx in reduced_cells:
            indicator.dat.data[cnm_dg0[cell_idx]] = 1.0

        label_value = 999
        mesh.mark_entities(indicator, label_value)
        # dim = mesh.topological_dimension()
        dim = mesh.topological_dimension      # property now, not a method
        subm = Submesh(mesh, dim, label_value)

        # ---- Function space on submesh (same element) ----
        elem = V_full.ufl_element()
        self.V_sub = VectorFunctionSpace(subm, elem.family(), elem.degree())

        print(f"    Submesh: {subm.num_cells()} cells, {self.V_sub.dim()} DOFs")

        # ---- DOF remapping ----
        self._build_dof_remapping(V_full, self.V_sub, subm)

        # ---- Remap DEIM indices ----
        self.sub_indices_F = self._remap_residual_indices()
        self.sub_indices_J = self._remap_jacobian_indices()

        # ---- Pre-allocate workspace ----
        self.u_sub = Function(self.V_sub, name="u_sub")
        self.v_test = TestFunction(self.V_sub)
        self.u_trial = TrialFunction(self.V_sub)

        print(f"    SubMeshDEIM ready ✓")

    # =================================================================
    # DOF remapping (via submesh_child_cell_parent_cell_map)
    # =================================================================

    def _build_dof_remapping(self, V_full, V_sub, subm):
        """Build node/DOF mapping using Firedrake's built-in cell correspondence."""
        value_size = V_full.value_size

        # submesh_child_cell_parent_cell_map[sub_cell] = parent_cell
        child_parent_map = subm.topology.submesh_child_cell_parent_cell_map.values.ravel()
        cnm_full = V_full.cell_node_map().values
        cnm_sub = V_sub.cell_node_map().values

        full_to_sub_node = {}
        for sub_cell in range(cnm_sub.shape[0]):
            parent_cell = child_parent_map[sub_cell]
            if parent_cell < 0:
                continue
            for local in range(cnm_sub.shape[1]):
                sub_n = int(cnm_sub[sub_cell, local])
                full_n = int(cnm_full[parent_cell, local])
                if full_n in full_to_sub_node:
                    assert full_to_sub_node[full_n] == sub_n, \
                        f"Inconsistent: full node {full_n} → " \
                        f"{full_to_sub_node[full_n]} and {sub_n}"
                else:
                    full_to_sub_node[full_n] = sub_n

        # Coordinate verification
        coords_full = Function(V_full).interpolate(SpatialCoordinate(V_full.mesh()))
        coords_sub = Function(V_sub).interpolate(SpatialCoordinate(subm))
        max_err = max(
            np.linalg.norm(
                coords_full.dat.data_ro[fn] - coords_sub.dat.data_ro[sn]
            )
            for fn, sn in full_to_sub_node.items()
        )
        assert max_err < 1e-10, f"Coordinate check failed: max error {max_err}"
        print(f"    Node mapping: {len(full_to_sub_node)} pairs, "
              f"coord error = {max_err:.1e}")

        # DOF-level mapping
        self.full_to_sub_dof = {}
        self.sub_to_full_dof = np.zeros(V_sub.dim(), dtype=np.int64)

        for full_n, sub_n in full_to_sub_node.items():
            for comp in range(value_size):
                full_dof = full_n * value_size + comp
                sub_dof = sub_n * value_size + comp
                self.full_to_sub_dof[full_dof] = sub_dof
                self.sub_to_full_dof[sub_dof] = full_dof

        # Node arrays for fast vectorized transfer
        self.full_nodes = np.array(
            list(full_to_sub_node.keys()), dtype=np.int64
        )
        self.sub_nodes = np.array(
            list(full_to_sub_node.values()), dtype=np.int64
        )

    # =================================================================
    # DEIM index remapping
    # =================================================================

    def _remap_residual_indices(self):
        """Translate DEIM residual indices from full mesh to submesh."""
        sub_idx = np.zeros_like(self.deim_ops.indices_F)
        for k, full_dof in enumerate(self.deim_ops.indices_F):
            sub_idx[k] = self.full_to_sub_dof[int(full_dof)]
        print(f"    DEIM residual: {len(sub_idx)} indices remapped ✓")
        return sub_idx

    def _remap_jacobian_indices(self):
        """Translate MDEIM Jacobian indices from full CSR to submesh CSR."""
        # Assemble a dummy Jacobian on submesh to get sparsity pattern
        u_dum = Function(self.V_sub)
        u_dum.dat.data[:] = 1.0
        v_t = TestFunction(self.V_sub)
        u_tr = TrialFunction(self.V_sub)
        J_dum = assemble(
            (inner(grad(u_tr) * u_dum, v_t)
             + inner(grad(u_dum) * u_tr, v_t)) * dx,
            mat_type='aij'
        )
        indptr, cols, _ = J_dum.M.handle.getValuesCSR()

        m_J = len(self.deim_ops.indices_J)
        sub_idx = np.zeros(m_J, dtype=np.int64)

        for k in range(m_J):
            row_full = int(self.deim_ops.indices_J_rows[k])
            col_full = int(self.deim_ops.indices_J_cols[k])
            row_sub = self.full_to_sub_dof[row_full]
            col_sub = self.full_to_sub_dof[col_full]

            start = indptr[row_sub]
            end = indptr[row_sub + 1]
            row_cols = cols[start:end]
            local_idx = np.searchsorted(row_cols, col_sub)
            assert local_idx < len(row_cols) and row_cols[local_idx] == col_sub, \
                f"MDEIM entry ({row_sub},{col_sub}) not in submesh sparsity"
            sub_idx[k] = start + local_idx

        print(f"    MDEIM Jacobian: {m_J} indices remapped ✓")
        return sub_idx

    # =================================================================
    # Online operations (called every Newton iteration)
    # =================================================================

    def transfer_to_submesh(self, u_full):
        """
        Copy velocity DOFs from full-mesh Function to submesh Function.

        Args:
            u_full: Function(V_full) with current velocity
        """
        self.u_sub.dat.data[self.sub_nodes] = u_full.dat.data_ro[self.full_nodes]

    def extract_residual(self, F_sub_assembled):
        """
        Extract DEIM residual from assembled submesh vector.

        Args:
            F_sub_assembled: assembled form on submesh (Firedrake Function/vector)

        Returns:
            conv_r: (r,) reduced convective vector
        """
        F_data = F_sub_assembled.dat.data_ro.flatten()
        F_at_deim = F_data[self.sub_indices_F]
        return self.deim_ops.V_T_B_F @ F_at_deim

    def extract_jacobian(self, J_csr_data_sub):
        """
        Extract MDEIM Jacobian from assembled submesh matrix CSR data.

        Args:
            J_csr_data_sub: CSR .data array from submesh Jacobian

        Returns:
            conv_J: (r, r) reduced convective Jacobian
        """
        J_at_mdeim = J_csr_data_sub[self.sub_indices_J]
        coeffs = self.deim_ops.U_J_P_inv @ J_at_mdeim
        return np.tensordot(coeffs, self.deim_ops.J_r_modes, axes=([0], [0]))
