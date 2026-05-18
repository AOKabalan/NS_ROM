from firedrake import *
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
from .navier_stokes import save_to_paraview, load_solution, setup_navier_stokes_problem, save_solution,NavierStokesSolution

def load_save_paraview(
    load_dir: str,
    output_dir: str,
    problem
    ):
    os.makedirs(output_dir, exist_ok=True)

    with np.load("snapshots/metadata.npz") as meta:
        params = meta["parameters"]
        num_params = len(params)
    for i in range(num_params):

        O_filename = f"snapshot_{i:04d}"
        I_filename = f"snapshot_{i:04d}.pvd"
        filepath = os.path.join(load_dir,O_filename)
        print(f'the file path :{filepath}')
        solution = load_solution(filepath,problem)
        save_to_paraview(solution,f'{output_dir}/{I_filename}')

def dofs_to_function(dofs: np.ndarray, space, name: str = "func") -> Function:
    """
    Convert DOF array back to Firedrake Function.

    Args:
        dofs: 1D array of DOFs
        space: FunctionSpace
        name: Function name

    Returns:
        Firedrake Function
    """
    f = Function(space, name=name)
    # Handle shape mismatch (in case dofs is flattened)
    if dofs.shape != f.dat.data.shape:
        f.dat.data[:] = dofs.reshape(f.dat.data.shape)
    else:
        f.dat.data[:] = dofs

    return f


def dofs_to_functions(
    dof_matrix: np.ndarray,
    space,
    name_prefix: str = "snapshot"
) -> List[Function]:

    n_snapshots = dof_matrix.shape[0]
    functions = []

    for i in range(n_snapshots):
        f = dofs_to_function(dof_matrix[i], space, name=f"{name_prefix}_{i}")
        functions.append(f)

    return functions


def functions_to_dofs(
    functions: List[Function],
    ) -> np.ndarray:
    if not functions:
        raise ValueError("Empty function list")

    n_snapshots = len(functions)
    n_dofs = functions[0].dat.data.flatten().shape[0]

    # Pre-allocate matrix [n_dofs, n_snapshots]
    dof_matrix = np.zeros((n_dofs, n_snapshots))

    for i, func in enumerate(functions):
        dof_matrix[:, i] = func.dat.data_ro.flatten().copy()


    return dof_matrix



def save_numpy_h5_paraview_mixed(
    ndof_data_u :np.ndarray,
    ndof_data_p :np.ndarray,
    output_dir: str,
    problem
    ):
    os.makedirs(output_dir,exist_ok= True)
    print(f'The shape of the ndof u is {ndof_data_u.shape} and p is :{ndof_data_p.shape}')

    num_params = ndof_data_u.shape[1]
    mesh = problem.mesh
    functions_u = dofs_to_functions(ndof_data_u.T,problem.velocity_space)
    functions_p = dofs_to_functions(ndof_data_p.T,problem.pressure_space)

    for i in range(num_params):

        O_filename = f"mode_{i}"
        filepath = os.path.join(output_dir,O_filename)
        u = functions_u[i]
        p = functions_p[i]

        with CheckpointFile(f'{filepath}'+ '.h5', 'w') as afile:
            afile.save_mesh(mesh)

            afile.save_function(u, name="velocity")
            afile.save_function(p, name="pressure")
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        pvd = VTKFile(f'{filepath}'+'.pvd')
        pvd.write(u, p)

        print(f"Solution saved as h5 and ParaView: {O_filename}")

def save_numpy_h5_paraview(
    ndof_data :np.ndarray,
    space: FunctionSpace,
    output_dir: str,
    problem
    ):
    os.makedirs(output_dir,exist_ok= True)
    print(f'The shape of the data is {ndof_data.shape} ')

    num_params = ndof_data.shape[1]
    mesh = problem.mesh

    functions = dofs_to_functions(ndof_data.T,space)

    for i in range(num_params):

        O_filename = f"mode_{i}"
        filepath = os.path.join(output_dir,O_filename)
        x = functions[i]


        with CheckpointFile(f'{filepath}'+ '.h5', 'w') as afile:
            afile.save_mesh(mesh)

            afile.save_function(x, name="func")

        pvd = VTKFile(f'{filepath}'+'.pvd')
        pvd.write(x)

        print(f"Solution saved as h5 and ParaView: {O_filename}")


def modes_for_tolerance(eigenvalues, tol, max_modes):
    total = np.sum(eigenvalues)
    cumulative = np.cumsum(eigenvalues) / total
    n = int(np.searchsorted(cumulative, 1.0 - tol) + 1)
    return min(n, max_modes)


if __name__ == "__main__":
    problem = setup_navier_stokes_problem(
        mesh_file='mesh/mid_pinball.msh',
        reynolds_init=80.0,
        amplitude_init=0.0
    )


    load_save_paraview("snapshots","paraview_snaps2",problem)