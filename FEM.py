import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax.experimental.sparse import BCOO
from jax.scipy.optimize import minimize
import threading
from functools import partial
from itertools import combinations
jax.config.update("jax_enable_x64", True)
from scipy.interpolate import RBFInterpolator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import scipy
# from scipy.sparse import csc_matrix
# import scipy.sparse
# from scipy.interpolate import griddata
import time, sys

import gauss, material# , interpolator
 

# FEM functions

def create_3d_uniform_grid(lengths, num_elements):
    """
    Create a 3D uniform grid mesh for finite element analysis.
    
    Parameters:
    lengths (tuple): Lengths of the x, y, z dimensions (Lx, Ly, Lz).
    num_elements (tuple): Number of elements in each direction (Nx, Ny, Nz).
    
    Returns:
    tuple: Tuple containing:
        - nodes (numpy.ndarray): Array of nodal coordinates.
        - elements (numpy.ndarray): Array of element connectivity.
    """
    
    Lx, Ly, Lz = lengths
    Nx, Ny, Nz = num_elements
    
    # Calculate number of nodes in each direction
    nx = Nx + 1
    ny = Ny + 1
    nz = Nz + 1
    
    # Create arrays for node coordinates
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    
    # Create meshgrid of node coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten the meshgrid arrays and stack them to get coordinates
    nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Create array for element connectivity
    elements = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                n0 = i * ny * nz + j * nz + k
                n1 = n0 + 1
                n2 = n0 + nz
                n3 = n2 + 1
                n4 = n0 + ny * nz
                n5 = n4 + 1
                n6 = n4 + nz
                n7 = n6 + 1
                elements.append([n0, n1, n3, n2, n4, n5, n7, n6])
    
    elements = np.array(elements)
    
    return nodes, jnp.array(nodes), elements, jnp.array(elements)

def get_shape_val_functions(elem_type, dim):
    """Hard-coded first order shape functions in the parent domain.
    Important: f1-f8 order must match "self.cells" by gmsh file!
    """
    
    ## Brick elements
    if (elem_type == 'CPE4' or 'C3D8' in elem_type) and dim == 2:
        f1 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1)) # (-1, -1)
        f2 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1)) # (-1, x` 1)
        shape_fun = [f1, f2, f3, f4]
        
    elif elem_type == 'C3D8I' and dim == 3: # inhouse node numbering
        f1 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # (-1,-1,-1)
        f2 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # (-1,-1, 1)
        f3 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # (-1, 1, 1)
        f4 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # (-1, 1,-1)
        
        f5 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # ( 1,-1,-1)
        f6 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # ( 1,-1, 1)
        f7 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # ( 1, 1, 1)
        f8 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # ( 1, 1,-1)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8]
    
    elif elem_type == 'C3D8' and dim == 3: # abaqus node numbering
        f1 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # (-1, 1, 1)
        f2 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # (-1,-1, 1)
        f3 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # (-1,-1,-1)
        f4 = lambda x: 1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # (-1, 1,-1)
        f5 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # ( 1, 1, 1)
        f6 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # ( 1,-1, 1)
        f7 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # ( 1,-1,-1)
        f8 = lambda x: 1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # ( 1, 1,-1)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8]
        
    elif (elem_type == 'CPE8' or elem_type == 'C3D20') and dim == 2:
        f1 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 - x[0]*(-1) - x[1]*(-1)) # (-1, -1)
        f2 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 - x[0]*( 1) - x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 - x[0]*( 1) - x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 - x[0]*(-1) - x[1]*( 1)) # (-1,  1)
        f5 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*(-1)) # ( 0, -1)
        f6 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*( 1)) # ( 1,  0)
        f7 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*( 1)) # ( 0,  1)
        f8 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*(-1)) # (-1,  0)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8]
    
    elif elem_type == 'C3D20' and dim == 3:
        f0 = lambda x: (1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*(-1))
                                 *(x[0]*( 1)     + x[1]*(-1)     + x[2]*(-1) - 2)) # ( 1,-1,-1)
        f1 = lambda x: (1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*(-1))
                                 *(x[0]*( 1)     + x[1]*( 1)     + x[2]*(-1) - 2)) # ( 1, 1,-1)
        f2 = lambda x: (1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*(-1))
                                 *(x[0]*(-1)     + x[1]*( 1)     + x[2]*(-1) - 2)) # (-1, 1,-1)
        f3 = lambda x: (1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*(-1))
                                 *(x[0]*(-1)     + x[1]*(-1)     + x[2]*(-1) - 2)) # (-1,-1,-1)
        f4 = lambda x: (1./8.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 + x[2]*( 1))
                                 *(x[0]*( 1)     + x[1]*(-1)     + x[2]*( 1) - 2)) # ( 1,-1, 1)
        f5 = lambda x: (1./8.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 + x[2]*( 1))
                                 *(x[0]*( 1)     + x[1]*( 1)     + x[2]*( 1) - 2)) # ( 1, 1, 1)
        f6 = lambda x: (1./8.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 + x[2]*( 1))
                                 *(x[0]*(-1)     + x[1]*( 1)     + x[2]*( 1) - 2)) # (-1, 1, 1)
        f7 = lambda x: (1./8.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 + x[2]*( 1))
                                 *(x[0]*(-1)     + x[1]*(-1)     + x[2]*( 1) - 2)) # (-1,-1, 1)
        
        f16 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 - x[2]**2  ) # ( 1,-1, 0)
        f17 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 - x[2]**2  ) # ( 1, 1, 0)
        f18 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 - x[2]**2  ) # (-1, 1, 0)
        f19 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 - x[2]**2  ) # (-1,-1, 0)
        
        f11 = lambda x: 1./4.*(1 - x[0]**2  )*(1 + x[1]*(-1))*(1 + x[2]*(-1)) # ( 0,-1,-1)
        f9  = lambda x: 1./4.*(1 - x[0]**2  )*(1 + x[1]*( 1))*(1 + x[2]*(-1)) # ( 0, 1,-1)
        f13 = lambda x: 1./4.*(1 - x[0]**2  )*(1 + x[1]*( 1))*(1 + x[2]*( 1)) # ( 0, 1, 1)
        f15 = lambda x: 1./4.*(1 - x[0]**2  )*(1 + x[1]*(-1))*(1 + x[2]*( 1)) # ( 0,-1, 1)
        
        f8  = lambda x: 1./4.*(1 + x[0]*( 1))*(1 - x[1]**2  )*(1 + x[2]*(-1)) # ( 1, 0,-1)
        f10 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 - x[1]**2  )*(1 + x[2]*(-1)) # (-1, 0,-1)
        f14 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 - x[1]**2  )*(1 + x[2]*( 1)) # (-1, 0, 1)
        f12 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 - x[1]**2  )*(1 + x[2]*( 1)) # ( 1, 0, 1)
        
        shape_fun = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]
    
    ## Tet elements
    elif (elem_type == 'CPE3' or elem_type == 'C3D4') and dim == 2:
        f1 = lambda x: x[0]
        f2 = lambda x: x[1]
        f3 = lambda x: 1 - x[0] - x[1]
        shape_fun = [f1, f2, f3]
    
    elif elem_type == 'C3D4' and dim == 3:
        f2 = lambda x: x[0]
        f3 = lambda x: x[1]
        f4 = lambda x: x[2]
        f1 = lambda x: 1 - x[0] - x[1] - x[2]
        shape_fun = [f1, f2, f3, f4]
        
    elif (elem_type == 'CPE6' or elem_type == 'C3D10') and dim == 2:
        f1 = lambda x: x[0]*(2*x[0]-1)
        f2 = lambda x: x[1]*(2*x[1]-1)
        f3 = lambda x: (1-x[0]-x[1])*(1-2*x[0]-2*x[1])
        f4 = lambda x: 4*x[0]*x[1]
        f5 = lambda x: 4*x[1]*(1-x[0]-x[1])
        f6 = lambda x: 4*x[0]*(1-x[0]-x[1])
        shape_fun = [f1, f2, f3, f4, f5, f6]
    

    elif elem_type == 'C3D10' and dim == 3:
        # u = (1 - x[0] - x[1] - x[2])
        # r = x[0]
        # s = x[1]
        # t = x[2]
        f1 = lambda x: (1 - x[0] - x[1] - x[2]) * (2*(1 - x[0] - x[1] - x[2])-1)
        f2 = lambda x: x[0]                     * (2*x[0]                    -1)
        f3 = lambda x: x[1]                     * (2*x[1]                    -1)
        f4 = lambda x: x[2]                     * (2*x[2]                    -1)
        
        f5 = lambda x: 4 * x[0] * (1 - x[0] - x[1] - x[2])
        f6 = lambda x: 4 * x[0] * x[1]
        f7 = lambda x: 4 * x[1] * (1 - x[0] - x[1] - x[2])
        
        f8 = lambda x: 4 * x[2] * (1 - x[0] - x[1] - x[2])
        f9 = lambda x: 4 * x[0] * x[2]
        f10= lambda x: 4 * x[1] * x[2]
        
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    
    return shape_fun

def get_shape_grad_functions(elem_type, dim):
    shape_fns = get_shape_val_functions(elem_type, dim)
    # print(elem_type, dim)
    return [jax.grad(f) for f in shape_fns]

# @partial(jax.jit, static_argnames=['Gauss_num', 'dim', 'elem_type', 'device_main']) Warning!! We cannot assign correct device_main if we jit this functin.. 
def get_shape_vals(Gauss_num, dim, elem_type, device_main):
    """Pre-compute shape function values

    Returns
    -------
    shape_vals: ndarray
        (8, 8) = (num_quads, num_nodes)  
    """
    shape_val_fns = get_shape_val_functions(elem_type, dim)
    quad_points, quad_weights = gauss.get_quad_points(Gauss_num, dim, elem_type, device_main)
    shape_vals = []
    for quad_point in quad_points:
        physical_shape_vals = []
        for shape_val_fn in shape_val_fns:
            physical_shape_val = shape_val_fn(quad_point) 
            physical_shape_vals.append(physical_shape_val)
 
        shape_vals.append(physical_shape_vals)

    shape_vals = jnp.array(shape_vals, device=device_main) # (num_quads, num_nodes)
    # assert shape_vals.shape == (global_args['num_quads'], global_args['num_nodes'])
    return shape_vals

# @partial(jax.jit, static_argnames=['Gauss_num', 'dim', 'elem_type']) Warning!! We cannot assign correct device_main if we jit this functin.. 
def get_shape_grads(Gauss_num, dim, elem_type, XY, Elem_nodes, device_main):
    """Pre-compute shape function gradients

    Returns
    -------
    shape_grads_physical: ndarray
        (cell, num_quads, num_nodes, dim)  
    JxW: ndarray
        (cell, num_quads)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type, dim)
    quad_points, quad_weights = gauss.get_quad_points(Gauss_num, dim, elem_type, device_main)
    # print('quad points weights')
    # print(quad_points, quad_weights)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            # See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
            # Page 147, Eq. (3.9.3)
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
 
        shape_grads.append(physical_shape_grads)
    
    
    shape_grads = jnp.array(shape_grads, device=device_main) # (num_quads, num_nodes, dim) 
    physical_coos = jnp.take(XY, Elem_nodes, axis=0) # (num_cells, num_nodes, dim)

    # physical_coos: (num_cells, none,      num_nodes, dim, none)
    # shape_grads:   (none,      num_quads, num_nodes, none, dim)
    # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
    jacobian_dx_deta = jnp.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True)
    
    jacobian_det = jnp.linalg.det(jacobian_dx_deta).reshape(len(Elem_nodes), len(quad_weights))# (num_cells, num_quads)

    jacobian_deta_dx = jnp.linalg.inv(jacobian_dx_deta)
    # print(jacobian_deta_dx[0, :, 0, :, :])
    # print(shape_grads)
    shape_grads_physical = (shape_grads[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    # print(shape_grads_physical[0])

    # For first order FEM with 8 quad points, those quad weights are all equal to one
    # quad_weights = 1.
    c = 1
    if (elem_type == 'CPE3' or elem_type == 'CPE6' or elem_type.startswith('C3D4') or elem_type.startswith('C3D10')) and dim == 2:
        c = 1/2
    elif (elem_type.startswith('C3D4') or elem_type.startswith('C3D10')) and dim == 3:
        c = 1/6
        
    JxW = jacobian_det * quad_weights[None, :] * c
    return shape_grads_physical, JxW # (num_cells, num_quads, num_nodes, dim), (num_cells, num_quads)

# @partial(jax.jit, static_argnames=['Gauss_num', 'dim', 'elem_type'])
def get_JxW_tr(Gauss_num, dim, elem_type, physical_coos, device_main):
    """Pre-compute shape function gradients

    Returns
    -------
    shape_grads_physical: ndarray
        (cell, num_quads, num_nodes, dim)  
    JxW: ndarray
        (cell, num_quads)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type, dim)
    quad_points, quad_weights = gauss.get_quad_points(Gauss_num, dim, elem_type, device_main)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
        shape_grads.append(physical_shape_grads)
    
    shape_grads = jnp.array(shape_grads, device=device_main) # (num_quads, num_nodes, dim) 
    jacobian_dx_deta = jnp.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True)
    jacobian_det = jnp.linalg.det(jacobian_dx_deta).reshape(len(physical_coos), len(quad_weights))# (num_cells, num_quads)
    # jacobian_deta_dx = jnp.linalg.inv(jacobian_dx_deta)
    # shape_grads_physical = (shape_grads[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    c = 1
    if (elem_type == 'CPE3' or elem_type == 'CPE6' or elem_type.startswith('C3D4') or elem_type.startswith('C3D10')) and dim == 2:
        c = 1/2
    # elif (elem_type == 'C3D4' or elem_type == 'C3D10') and dim == 3:
    #     c = 1/6
    JxW = jacobian_det * quad_weights[None, :] * c
    return JxW # (num_cells, num_quads, num_nodes, dim), (num_cells, num_quads)


def get_connectivity(Elemental_patch_nodes, var):
    """ [numpy array function] Given Elem_nodes in FEM or Elemental_patch_nodes in C-FEM, computes global DoFs, i.e., connectivity matrix
    [numpy variables]
    --- inputs ---
    Elemental_patch_nodes (or Elem_nodes): (nelem, edex_max) or (nelem, nodes_per_elem) int.
    dim
    --- outputs ---
    connectivity: (nelem, edex_max * var) or (nelem, nodes_per_elem * var)
    """
    # get connectivity vector
    (nelem, edex_max) = Elemental_patch_nodes.shape
    connectivity = np.zeros((nelem, edex_max*var), dtype = np.int64)
    connectivity[:, np.arange(0,edex_max*var, var)] = Elemental_patch_nodes*var
    connectivity[:, np.arange(1,edex_max*var, var)] = Elemental_patch_nodes*var+1
    if var == 3:
        connectivity[:, np.arange(2,edex_max*var, var)] = Elemental_patch_nodes*var+2
    return connectivity


# Ensure the nodes are on the same plane
def check_coplanar(nodes):
    return jnp.isclose(jnp.linalg.det(jnp.vstack([nodes.T, jnp.ones(4)])), 0)
# assert jnp.all(jax.vmap(check_coplanar)(elements)), "One or more elements are not coplanar."

def sort_elem_idx(elem_nodes, sorted_index):
    # elem_nodes: (nnode,), int
    # sorted_index: (nnode,), int
    return elem_nodes[sorted_index]

def project_to_2d_plane(nodes, device_main):
    """
    nodes: (3, dim=3), 3 nodes in 3D space 
    """
    # Compute two vectors in the plane
    vec1 = nodes[1] - nodes[0] # (dim=3,)
    vec2 = nodes[2] - nodes[0] # (dim=3,)

    # Compute the normal vector of the plane
    normal = jnp.cross(vec1, vec2)
    normal = normal / jnp.linalg.norm(normal)

    # Create a local coordinate system in the plane
    local_x = vec1 / jnp.linalg.norm(vec1)
    local_y = jnp.cross(normal, local_x)
    local_y = local_y / jnp.linalg.norm(local_y)

    # Project nodes onto the local coordinate system
    def project_point(point, device_main):
        relative_point = point - nodes[0]
        x_coord = jnp.dot(relative_point, local_x)
        y_coord = jnp.dot(relative_point, local_y)
        return jnp.array([x_coord, y_coord], device=device_main)

    projected_coords = jax.vmap(project_point, in_axes=[0,None])(nodes, device_main)

    return projected_coords




# compute FEM shape function and store in RAM memory
def get_FEM_shape_fun(XY, Elem_nodes, elem_type, Gauss_num, quad_num, max_array_size_block, device_main):
    """ Computes FEM shape functions and store in RAM memory
    --- inputs ---
    XY: (nnode, dim): jnp array
    Elem_nodes: (nelem, nodes_per_elem): jnp array
    --- outputs ---
    shape_vals: (nelem, quad_num, nodes_per_elem), np.array
    shape_grads_physical: (nelem, quad_num, nodes_per_elem, dim), np.array
    JxW: (nelem, quad_num), np.array
    """
    ## define basic constants
    nelem, nodes_per_elem = Elem_nodes.shape
    dim = XY.shape[1]

    ## assign RAM space in CPU
    shape_grads_physical_host = np.zeros((nelem, quad_num, nodes_per_elem, dim), dtype=np.float64)
    JxW_host = np.zeros((nelem, quad_num), dtype=np.float64)

    ## compute shape function and directly store at GPU
    shape_vals = get_shape_vals(Gauss_num, dim, elem_type, device_main) # (quad_num, nodes_per_elem)

    ## decide how many blocks are we gonnna use
    size_largest_array = nelem * quad_num * nodes_per_elem * dim # size of shape_grads_physical
    nblock = int(size_largest_array // max_array_size_block + 1)
    Elem_nodes_list = jnp.array_split(Elem_nodes, nblock, axis=0)
    Elem_idx_host_list = np.array_split(np.arange(nelem, dtype=np.int64), nblock, axis=0) # global element index divided into nblocks
    print(f"\tFEM shape fun -> {nblock} elemental blocks with {len(Elem_idx_host_list[0])} elements per block")

    for Elem_nodes_block, Elem_idx_block_host in zip(Elem_nodes_list, Elem_idx_host_list):
        shape_grads_physical_block, JxW_block = get_shape_grads(Gauss_num, dim, elem_type, XY, Elem_nodes_block, device_main)
        shape_grads_physical_host[Elem_idx_block_host] = np.array(shape_grads_physical_block)
        JxW_host[Elem_idx_block_host] = np.array(JxW_block)
        
    return shape_vals, shape_grads_physical_host, JxW_host

@jax.jit
def get_Bmat_block(shape_grads_physical_block, Elem_dofs_block):
    
    elem_dof = Elem_dofs_block.shape[1]
    nelem_per_block, quad_num, dim = shape_grads_physical_block.shape[0], shape_grads_physical_block.shape[1], shape_grads_physical_block.shape[3]
    Bmat_block = jnp.zeros((nelem_per_block, quad_num, 6, elem_dof), dtype=jnp.float64) # (nelem_block, quad_num, 6, elem_dof)

    Bmat_block = Bmat_block.at[:,:,0,jnp.arange(0, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,0]) # N,x
    Bmat_block = Bmat_block.at[:,:,1,jnp.arange(1, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,1]) # N,y
    Bmat_block = Bmat_block.at[:,:,2,jnp.arange(2, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,2]) # N,z
    
    Bmat_block = Bmat_block.at[:,:,3,jnp.arange(0, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,1]) # N,y
    Bmat_block = Bmat_block.at[:,:,3,jnp.arange(1, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,0]) # N,x
    
    Bmat_block = Bmat_block.at[:,:,4,jnp.arange(1, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,2]) # N,z
    Bmat_block = Bmat_block.at[:,:,4,jnp.arange(2, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,1]) # N,y
    
    Bmat_block = Bmat_block.at[:,:,5,jnp.arange(0, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,2]) # N,z
    Bmat_block = Bmat_block.at[:,:,5,jnp.arange(2, elem_dof, dim)].set(shape_grads_physical_block[:,:,:,0]) # N,x
    
    return Bmat_block

   
@partial(jax.jit, static_argnames=['nelem_per_block', 'quad_num', 'elem_dof', 'dim', 'vv_b_fun'])
def get_b_block(b, XY, Elem_nodes_block, connectivity_block, shape_vals, JxW_block, 
                nelem_per_block, quad_num, elem_dof, dim, vv_b_fun):
    
    XY_elem = jnp.take(XY, Elem_nodes_block, axis=0) # (nelem_per_block, nodes_per_elem, dim)
    XY_GPs = jnp.sum(shape_vals[None,:,:,None] * XY_elem[:,None,:,:], axis=2) # (nelem_per_block, quad_num, dim)
    b_block = vv_b_fun(XY_GPs[:,:,0], XY_GPs[:,:,1], XY_GPs[:,:,2]) # (nelem_per_block, quad_num, var)
    
    
    N_block = jnp.zeros((nelem_per_block, quad_num, elem_dof, dim), dtype=jnp.float64) # (nelem_per_block, quad_num, elem_dof, var)
    N_block = N_block.at[:,:,jnp.arange(0, elem_dof, dim),0].set(shape_vals[None,:,:]) # column 0
    N_block = N_block.at[:,:,jnp.arange(1, elem_dof, dim),1].set(shape_vals[None,:,:]) # column 1
    N_block = N_block.at[:,:,jnp.arange(2, elem_dof, dim),2].set(shape_vals[None,:,:]) # column 2
    
    Nb_block = jnp.sum(N_block[:,:,:,:] * b_block[:,:,None,:], axis=3) # (nelem_per_block, quad_num, elem_dof)
    Nb_block = jnp.sum(Nb_block[:,:,:] * JxW_block[:,:,None], axis=1) # (nelem_per_block, elem_dof)
    b = b.at[connectivity_block.reshape(-1)].add(Nb_block.reshape(-1))
    return b


## Dirichlet boundary conditions
def get_BCs_Abaqus(XY, Elem_nodes, config, device_main):
    ''' 
    --- inputs ---
    XY: (nnode, dim) nodal coordinates
    Elem_nodes: (nelem, nodes_per_elem)
    var: scalar, output variables
    --- outputs ---
    Dirichlet: dictionary, Dirichlet BC info
    Neumann: dictionary, Neurmann BC info
    '''
    var = config["INPUT_PARAM"]["var"]
    elem_type = config["elem_type"]
    Dirichlet_dof, Dirichlet_val = [], [] # for multiple Dirichlet BCs
    Neumann_Elem_nodes_tr, Neumann_XY_Elems_tr, Neumann_XY_projected, Neumann_Elem_dofs_tr, Neumann_force = [],[],[],[],[]
    # Dirichlet_dof_, Dirichlet_val_ = [], [] # for multiple Dirichlet BCs
    for key, value in config["BC"].items():
        # key: BC_disp or BC_track
        # value: list of dictionaries.
        
        ## Displacement BC
        if 'disp' in key:
            for BC_disp in value:
                nodal_idx = jnp.array(config["nsets"][BC_disp["nset"]], device=device_main)            
                dirichlet_dof = nodal_idx.reshape(-1,1) * var + jnp.arange(var, device=device_main) # (nnode, var) global dofs: u,v,w
                if BC_disp["Type_detail"] == "ENCASTRE":
                    dirichlet_val = jnp.zeros_like(dirichlet_dof, device=device_main)
                # dirichlet_val = jnp.tile(value['val'], (len(dirichlet_dof), 1))
                Dirichlet_dof.append(dirichlet_dof.reshape(-1))
                Dirichlet_val.append(dirichlet_val.reshape(-1))

        ## Traction BC
        elif 'trac' in key:

            if not value: # if there is no traction BC, skip
                continue
            
            ## Assume there is only one traction BC for now. Multiple traction BCs are not supported yet.
            value = value[0] # traction BC is a list of one dictionary
            surface_name = value["elset"] # for one traction BC

            # jnp.array(config["nsets"][value["Elset"]])
            for elset_name, elset in config["elsets"].items():
                # connectivity, XY_projected,  = []
                if surface_name in elset_name: # if the elset is for surface traction
                    
                    if "C3D4" in elem_type:
                        ## for C3D4 elements
                        if "S1" in elset_name:
                            tr_idx = jnp.array([0,1,2], device=device_main)
                        elif "S2" in elset_name:
                            tr_idx = jnp.array([3,0,1], device=device_main)
                        elif "S3" in elset_name:
                            tr_idx = jnp.array([1,2,3], device=device_main)
                        elif "S4" in elset_name:
                            tr_idx = jnp.array([2,3,0], device=device_main)
                        else:
                            print("Error: check traction elset names")
                            sys.exit()
                    
                    elif "C3D8" in elem_type:
                        ## for C3D8 elements
                        if "S1" in elset_name:
                            tr_idx = jnp.array([0,1,2,3], device=device_main)
                        elif "S2" in elset_name:
                            tr_idx = jnp.array([4,5,6,7], device=device_main)
                        elif "S3" in elset_name:
                            tr_idx = jnp.array([0,1,5,4], device=device_main)
                        elif "S4" in elset_name:
                            tr_idx = jnp.array([1,5,6,2], device=device_main)
                        elif "S5" in elset_name:
                            tr_idx = jnp.array([2,6,7,3], device=device_main)
                        elif "S6" in elset_name:
                            tr_idx = jnp.array([0,4,7,3], device=device_main)
                        else:
                            print("Error: check traction elset names")
                            sys.exit()
                    else:
                        print("Error: check traction elset names")
                        sys.exit()

                    Elem_nodes_tr = Elem_nodes[elset,:][:,tr_idx].reshape(-1,len(tr_idx)) # (nelem_tr,npe=3) for C3D4 element
                    Elem_nodes_tr_host = np.array(Elem_nodes_tr)
                    XY_Elems_tr = jnp.take(XY, Elem_nodes_tr, axis=0) # (nelem_tr, npe=3, dim=3)
                    XY_projected = jax.vmap(project_to_2d_plane, in_axes=[0,None])(XY_Elems_tr, device_main) # *(nelem_tr, npe=3, dim=2)
                    Elem_dofs_tr = jax.device_put(get_connectivity(Elem_nodes_tr_host, var), device_main) # (nelem_tr, var * npe=3)

                    if "neumann_XY_projected" not in locals():
                        neumann_Elem_nodes_tr = Elem_nodes_tr
                        neumann_XY_Elems_tr = XY_Elems_tr
                        neumann_XY_projected = XY_projected
                        neumann_Elem_dofs_tr = Elem_dofs_tr
                    else:
                        neumann_Elem_nodes_tr = jnp.concatenate((neumann_Elem_nodes_tr, Elem_nodes_tr), axis=0)
                        neumann_XY_Elems_tr = jnp.concatenate((neumann_XY_Elems_tr, XY_Elems_tr), axis=0)
                        neumann_XY_projected = jnp.concatenate((neumann_XY_projected, XY_projected), axis=0)
                        neumann_Elem_dofs_tr = jnp.concatenate((neumann_Elem_dofs_tr, Elem_dofs_tr), axis=0)

            Neumann_Elem_nodes_tr.append(neumann_Elem_nodes_tr)
            Neumann_XY_Elems_tr.append(neumann_XY_Elems_tr)
            Neumann_XY_projected.append(neumann_XY_projected)
            Neumann_Elem_dofs_tr.append(neumann_Elem_dofs_tr)
            Neumann_force.append(float(value["Magnitude"])*jnp.array(value["Vector"], device=device_main))

    Dirichlet = {'dof': Dirichlet_dof, 'val': Dirichlet_val}

    Neumann = {'Elem_nodes_tr':Neumann_Elem_nodes_tr, # (nelem_tr, npe_tr)
               'XY_Elems_tr': Neumann_XY_Elems_tr,  # (nelem_tr, npe_tr, dim)
               'XY_projected': Neumann_XY_projected, # (nelem_tr, npe_tr, dim-1)
               'Elem_dofs_tr': Neumann_Elem_dofs_tr, # (nelem_tr, npe_tr*var)
               'force': Neumann_force} # (var, )
    
    return Dirichlet, Neumann



############################ Assembly-free solution scheme ############################

def get_x0_Dirichlet(dof_global, Dirichlet, device_main):
    """ initialize solution vector x0 with Dirichlet BC
    """
    x0 = jnp.zeros(dof_global, dtype=jnp.float64, device=device_main) # apply dirichlet BC
    Dirichlet_dof = Dirichlet['dof']
    Dirichlet_val = Dirichlet['val']
    for dirichlet_dof, dirichlet_val in zip(Dirichlet_dof, Dirichlet_val):
        x0 = x0.at[dirichlet_dof].set(dirichlet_val)
    return x0

@jax.jit # turn this off when debugging
def get_residual(Ax, sol, shape_grads_physical_block, JxW_block, Elem_dofs_block, D):
    """ Given the shape function gradients for elemental block, compute residual vector
    --- inputs ---
    Ax: (dof_global,) current residual
    sol: (dof_global,) current solution
    shape_grads_physical_block (or Elem_dofs_block for C-HiDeNN): (nelem_block, quad_num, npe or edex_max, dim)
    JxW_block: (nelem_block, quad_num)
    Elem_dofs_block (or Elem_dofs_block for C-HiDeNN): (nelem_block, elem_dof)
    D: elastic modulus
    """

    # start_residual = time.time()
    Bmat_block = get_Bmat_block(shape_grads_physical_block, Elem_dofs_block) # (nelem_block, quad_num, 6, elem_dof)
    u_block = jnp.take(sol, Elem_dofs_block, axis=0) # (nelem_block, elem_dof)
    Bu = jnp.sum(Bmat_block * u_block[:,None,None,:], axis=3) # (nelem_block, quad_num, 6), strain
    DBu = jnp.sum(D[None,None,:,:] * Bu[:,:,:,None], axis=2) # (nelem_block, quad_num, 6), stress
    BTDBu = jnp.sum(Bmat_block[:,:,:,:] * DBu[:,:,:,None], axis=2) # (nelem_block, quad_num, elem_dof)
    BTDBuJxW = jnp.sum(BTDBu[:,:,:] * JxW_block[:, :, None], axis=1) # (nelem_block, elem_dof), volumetric integration
    Ax = Ax.at[Elem_dofs_block.reshape(-1)].add(BTDBuJxW.reshape(-1))
    # print(f"\t\t{iblock+1}/{nblock} residual took -> {time.time() - start_residual}")

    return Ax


def get_device_list_and_b(shape_fun_dicts, config, 
            Dirichlet, Neumann, max_array_size_block, devices):
    """ This function allocates the shape function arrays on each device.
    --- outputs ---
    shape_grads_physical_dlist: list of list. The first list is the device list, and the second is the element-wise block list
    JxW_dlist: list of list
    Elem_dofs_dlist: list of list
    D_dlist: list of (6,6) jnp array for the elasticity matrix
    b: External forcing vector. jnp arrray stored at the device_main
    """
    
    ## bring basic constants
    dof_global      = config["dof_global"]
    dim             = config["INPUT_PARAM"]["dim"]
    var             = config["INPUT_PARAM"]["var"]
    n_devices       = len(devices)
    device_main     = devices[0]
    
    shape_grads_physical_host   = shape_fun_dicts[0]["shape_grads_physical_host"]
    JxW_host                    = shape_fun_dicts[0]["JxW_host"]
    Elem_dofs_host              = shape_fun_dicts[0]['Elem_dofs_host'] 

    D = material.get_material_model(config, device_main)
    b = jnp.zeros(dof_global, dtype=jnp.float64, device=device_main)
    Dirichlet_dof = Dirichlet['dof']
    Dirichlet_val = Dirichlet['val']
    large_value = 1e16


    ## Step 1: split with devices
    shape_grads_physical_host_dlist = np.array_split(shape_grads_physical_host, n_devices, axis=0)
    JxW_host_dlist = np.array_split(JxW_host, n_devices, axis=0)
    Elem_dofs_host_dlist = np.array_split(Elem_dofs_host, n_devices, axis=0)
    shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist = [],[],[],[]
    for device, shape_grads_physical_host_device, JxW_host_device, Elem_dofs_host_device in zip(
        devices, shape_grads_physical_host_dlist, JxW_host_dlist, Elem_dofs_host_dlist): # loop over devices

        ### decide how many elemental blocks are we gonnna use
        size_largest_array = (shape_grads_physical_host_device.shape[0]*shape_grads_physical_host_device.shape[1]*
                              shape_grads_physical_host_device.shape[2]*shape_grads_physical_host_device.shape[3])
        nblock = int(size_largest_array // (max_array_size_block) + 1)
        shape_grads_physical_host_device_list = np.array_split(shape_grads_physical_host_device, nblock, axis=0)
        JxW_host_device_list = np.array_split(JxW_host_device, nblock, axis=0)
        Elem_dofs_host_device_list = np.array_split(Elem_dofs_host_device, nblock, axis=0)
        
        ### transfer from CPU RAM to GPU VRAM
        shape_grads_physical_device_list, JxW_device_list, Elem_dofs_device_list = [],[],[]
        for shape_grads_physical, JxW, Elem_dofs in zip(shape_grads_physical_host_device_list, 
                                                        JxW_host_device_list, Elem_dofs_host_device_list):
            shape_grads_physical_device_list.append(jnp.array(shape_grads_physical, device=device))
            JxW_device_list.append(jnp.array(JxW, device=device))
            Elem_dofs_device_list.append(jnp.array(Elem_dofs, device=device))

        ### save at dlist variables
        shape_grads_physical_dlist.append(shape_grads_physical_device_list)
        JxW_dlist.append(JxW_device_list)
        Elem_dofs_dlist.append(Elem_dofs_device_list)
        D_dlist.append(jax.device_put(D, device))

 
    ## Neumann boundary conditions
    if len(Neumann) != 0:
        
        for Elem_nodes_tr, XY_Elems_tr, XY_projected, Elem_dofs_tr, force in zip(
                Neumann['Elem_nodes_tr'], Neumann['XY_Elems_tr'], Neumann['XY_projected'], Neumann['Elem_dofs_tr'], Neumann['force']):
            nelem_tr, npe_tr = Elem_nodes_tr.shape
            Gauss_num_tr = config["Gauss"]["Gauss_num_tr"]
            quad_num_tr = config["Gauss"]["quad_num_tr"]
            elem_type = config["elem_type"]

            shape_vals_tr = get_shape_vals(Gauss_num_tr, dim-1, elem_type, device_main) # (quad_num, npe=4) for 2D element
            JxW_tr = get_JxW_tr(Gauss_num_tr, dim-1, elem_type, XY_projected, device_main) # (nelem, quad_num_tr)
            
            N_trac = jnp.zeros((nelem_tr, quad_num_tr, npe_tr*var, var), dtype=jnp.float64, device=device_main) # (nelem, quad_num, elem_dof=npe*var, var)
            N_trac = N_trac.at[:,:,jnp.arange(0,npe_tr*var,var, device=device_main),0].set(shape_vals_tr[None,:,:]) # column 0
            N_trac = N_trac.at[:,:,jnp.arange(1,npe_tr*var,var, device=device_main),1].set(shape_vals_tr[None,:,:]) # column 1
            N_trac = N_trac.at[:,:,jnp.arange(2,npe_tr*var,var, device=device_main),2].set(shape_vals_tr[None,:,:]) # column 2
            
            
            NP = jnp.sum( jnp.matmul(N_trac, force.reshape(-1,1)[None,None,:,:]) * JxW_tr[:,:,None,None], axis=1) # (nelem, elem_dof, 1)
            b = b.at[Elem_dofs_tr.reshape(-1)].add(NP.reshape(-1))  # assemble 
    
    ## Dirichlet BC applided to b vector
    for dirichlet_dof, dirichlet_val in zip(Dirichlet_dof, Dirichlet_val):
        b = b.at[dirichlet_dof].set(large_value * dirichlet_val)

    return shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist, b

################# Analytical solution #################
def u_exact(x, device_main):
    # x: 3D vector
    # u: 3D vector
    alpha, beta, gamma = 1.0, 1.0, 1.0
    # alpha, beta, gamma = 5.0, 5.0, 10.0
    u_x = jnp.sin(jnp.pi * alpha * x[0]) * jnp.sin(jnp.pi * beta * x[1]) * jnp.sin(jnp.pi * gamma * x[2])
    u_y = jnp.sin(jnp.pi * alpha * x[0]) * jnp.sin(jnp.pi * beta * x[1]) * jnp.sin(jnp.pi * gamma * x[2])
    u_z = jnp.sin(jnp.pi * alpha * x[0]) * jnp.sin(jnp.pi * beta * x[1]) * jnp.sin(jnp.pi * gamma * x[2])
    return jnp.array([u_x, u_y, u_z], device=device_main)
v_u_exact = jax.vmap(u_exact, in_axes=(0,None))
vv_u_exact = jax.vmap(v_u_exact, in_axes=(0,None))

# Gradient of u(x): ∇u is a 3x3 Jacobian matrix
# Compute deformation gradient
def compute_grad_u(x, device_main): 
    grad_u_fn = jax.jacobian(u_exact)  # Returns a function ∇u(x)
    grad_u = grad_u_fn(x, device_main)  # Shape (3,3): du_i/dx_j
    return grad_u

# Strain tensor ε
def compute_strain(grad_u):
    eps = 0.5 * (grad_u + grad_u.T)
    return eps

# Cauchy stress σ
def compute_cauchy_stress(x, E, nu, device_main):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    grad_u = compute_grad_u(x, device_main)  # (3,3)
    eps = compute_strain(grad_u)             # (3,3)
    trace_eps = jnp.trace(eps)               # scalar
    sigma = lam * trace_eps * jnp.eye(3) + 2 * mu * eps
    return sigma
v_compute_cauchy_stress = jax.vmap(compute_cauchy_stress, in_axes=(0,None,None,None))
vv_compute_cauchy_stress = jax.vmap(v_compute_cauchy_stress, in_axes=(0,None,None,None))

def compute_body_force(x, E, nu, device_main):
    """
    Compute the body force vector at a given point x for linear elasticity,
    where body force b = -div(sigma), sigma = cauchy stress tensor.
    Returns a vector of shape (3,).
    """
    def sigma_fn(x_):
        return compute_cauchy_stress(x_, E, nu, device_main)
    # Compute divergence of sigma: b_i = -∑_j ∂σ_ij/∂x_j
    div_sigma = jnp.zeros(3, dtype=jnp.float64, device=device_main)
    for i in range(3):
        # For each component i, sum over j the derivative wrt x_j of sigma_ij
        def sigma_ij(x_, j):
            return sigma_fn(x_)[i, j]
        div_sigma_i = 0.0
        for j in range(3):
            grad_sigma_ij = jax.grad(lambda x_: sigma_ij(x_, j))(x)
            div_sigma_i += grad_sigma_ij[j]
        div_sigma = div_sigma.at[i].set(div_sigma_i)
    body_force = -div_sigma
    return body_force
v_compute_body_force = jax.vmap(compute_body_force, in_axes=(0, None, None, None))
vv_compute_body_force = jax.vmap(v_compute_body_force, in_axes=(0, None, None, None))

################# Analytical solution #################

def get_list_and_b(XY, Elem_nodes, shape_fun_dicts, config, 
            Dirichlet, Neumann, max_array_size_block, device_main):
    """ This function allocates the shape function arrays on each device.
    --- outputs ---
    shape_grads_physical_dlist: list of list. The first list is the device list, and the second is the element-wise block list
    JxW_dlist: list of list
    Elem_dofs_dlist: list of list
    D_dlist: list of (6,6) jnp array for the elasticity matrix
    b: External forcing vector. jnp arrray stored at the device_main
    """
    
    ## bring basic constants
    dof_global      = config["dof_global"]
    dim             = config["INPUT_PARAM"]["dim"]
    var             = config["INPUT_PARAM"]["var"]
    
    shape_vals                  = shape_fun_dicts[0]["shape_vals"]
    if shape_fun_dicts[0]["type"] == "FEM":
        shape_vals_FEM = shape_vals
    elif shape_fun_dicts[0]["type"] == "C-HiDeNN":
        shape_vals_FEM = shape_fun_dicts[0]["shape_vals_FEM"]

    shape_grads_physical_host   = shape_fun_dicts[0]["shape_grads_physical_host"]
    JxW_host                    = shape_fun_dicts[0]["JxW_host"]
    Elem_dofs_host              = shape_fun_dicts[0]['Elem_dofs_host'] 

    shape_grads_physical = jnp.array(shape_grads_physical_host, device=device_main) # (nelem, qauss_num_FEM, nodes_per_elem, dim)
    JxW = jnp.array(JxW_host, device=device_main) # (nelem, quad_num_FEM)
    Elem_dofs = jnp.array(Elem_dofs_host, device=device_main) # (nelem, nodes_per_elem*var)

    ## divide the arrays into blocks
    size_largest_array = (shape_grads_physical_host.shape[0]*shape_grads_physical_host.shape[1]*
                              shape_grads_physical_host.shape[2]*shape_grads_physical_host.shape[3])
    nblock = int(size_largest_array // (max_array_size_block) + 1)
    shape_grads_physical_list = jnp.array_split(shape_grads_physical, nblock, axis=0)
    JxW_list = jnp.array_split(JxW, nblock, axis=0)
    Elem_dofs_list = jnp.array_split(Elem_dofs, nblock, axis=0)

    D = material.get_material_model(config, device_main)
    b = jnp.zeros(dof_global, dtype=jnp.float64, device=device_main)
    Dirichlet_dof = Dirichlet['dof']
    Dirichlet_val = Dirichlet['val']
    large_value = 1e16

    ## When there is a body force
    if "convergence" in config["INPUT_PARAM"]["input_name"]:

        ## divide into blocks
        size_largest_array = (shape_grads_physical_host.shape[0]*shape_grads_physical_host.shape[1]*
                                shape_grads_physical_host.shape[2]*shape_grads_physical_host.shape[3])
        nblock = int(size_largest_array // (max_array_size_block) + 1)
        # shape_grads_physical_host_list = np.array_split(shape_grads_physical_host, nblock, axis=0)
        JxW_host_list = np.array_split(JxW_host, nblock, axis=0)
        Elem_dofs_host_list = np.array_split(Elem_dofs_host, nblock, axis=0)
        Elem_nodes_list = jnp.array_split(Elem_nodes, nblock, axis=0)
        Elem_idx_list = jnp.array_split(jnp.arange(shape_grads_physical_host.shape[0], dtype=jnp.int64), nblock, axis=0) # global element index divided into nblocks

        for JxW_block_host, Elem_dofs_block_host, Elem_nodes_block, Elem_idx_block in zip(
        JxW_host_list, Elem_dofs_host_list, Elem_nodes_list, Elem_idx_list):
            
            ### convert to jnp array
            JxW_block = jnp.array(JxW_block_host, device=device_main) # (nelem_block, quad_num)
            Elem_dofs_block = jnp.array(Elem_dofs_block_host, device=device_main) # (nelem_block, elem_dof)
            nelem_block, quad_num = JxW_block.shape
            npe = shape_grads_physical_host.shape[2] # nodes per element
            
            ## compute gauss points coordinates
            XY_elem = jnp.take(XY, Elem_nodes_block, axis=0) # (nelem_block, nodes_per_elem, dim)
            XY_GPs = jnp.sum(shape_vals_FEM[None,:,:,None] * XY_elem[:,None,:,:], axis=2) # (nelem_per_block, quad_num, dim)
            
            # Compute body force at gauss points
            body_force_block = vv_compute_body_force(XY_GPs, config["MATERIAL"]["E"], config["MATERIAL"]["nu"], device_main)  # (nelem_block, quad_num, var=3)
            
            # Build N matrix: (nelem_block, quad_num, nodes_per_elem*var, var)
            N_block = jnp.zeros((nelem_block, quad_num, npe*var, var), dtype=jnp.float64, device=device_main)
            if shape_fun_dicts[0]["type"] == "FEM":
                N_block = N_block.at[:,:,jnp.arange(0, npe*var, var), 0].set(shape_vals[None,:,:])
                N_block = N_block.at[:,:,jnp.arange(1, npe*var, var), 1].set(shape_vals[None,:,:])
                N_block = N_block.at[:,:,jnp.arange(2, npe*var, var), 2].set(shape_vals[None,:,:])
            elif shape_fun_dicts[0]["type"] == "C-HiDeNN":
                N_block = N_block.at[:,:,jnp.arange(0, npe*var, var), 0].set(shape_vals[Elem_idx_block,:,:])
                N_block = N_block.at[:,:,jnp.arange(1, npe*var, var), 1].set(shape_vals[Elem_idx_block,:,:])
                N_block = N_block.at[:,:,jnp.arange(2, npe*var, var), 2].set(shape_vals[Elem_idx_block,:,:])

            # Multiply N by body force at each GP: (nelem_block, quad_num, nodes_per_elem*var)
            NB_block = jnp.sum(N_block * body_force_block[:,:,None,:], axis=3)
            # Integrate over GPs with JxW: (nelem_block, nodes_per_elem*var)
            NBJxW_block = jnp.sum(NB_block * JxW_block[:,:,None], axis=1)
            # Assemble into global b vector
            b = b.at[Elem_dofs_block.reshape(-1)].add(NBJxW_block.reshape(-1))

    ## Neumann boundary conditions
    if len(Neumann) != 0:
        
        for Elem_nodes_tr, XY_Elems_tr, XY_projected, Elem_dofs_tr, force in zip(
                Neumann['Elem_nodes_tr'], Neumann['XY_Elems_tr'], Neumann['XY_projected'], Neumann['Elem_dofs_tr'], Neumann['force']):
            nelem_tr, npe_tr = Elem_nodes_tr.shape
            Gauss_num_tr = config["Gauss"]["Gauss_num_tr"]
            quad_num_tr = config["Gauss"]["quad_num_tr"]
            elem_type = config["elem_type"]

            shape_vals_tr = get_shape_vals(Gauss_num_tr, dim-1, elem_type, device_main) # (quad_num, npe_tr=3) for 2D element
            JxW_tr = get_JxW_tr(Gauss_num_tr, dim-1, elem_type, XY_projected, device_main) # (nelem, quad_num_tr)
            
            N_trac = jnp.zeros((nelem_tr, quad_num_tr, npe_tr*var, var), dtype=jnp.float64, device=device_main) # (nelem, quad_num, elem_dof=npe*var, var)
            N_trac = N_trac.at[:,:,jnp.arange(0,npe_tr*var,var, device=device_main),0].set(shape_vals_tr[None,:,:]) # column 0
            N_trac = N_trac.at[:,:,jnp.arange(1,npe_tr*var,var, device=device_main),1].set(shape_vals_tr[None,:,:]) # column 1
            N_trac = N_trac.at[:,:,jnp.arange(2,npe_tr*var,var, device=device_main),2].set(shape_vals_tr[None,:,:]) # column 2
            
            if "convergence" in config["INPUT_PARAM"]["input_name"]: # for convergence study, we apply exact traction force
                # XY_Elems_tr # (nelem_tr, npe_tr=3, dim)
                # shape_vals_tr # (quad_num, npe_tr=3)
                XY_GPs = jnp.sum(shape_vals_tr[None,:,:,None] * XY_Elems_tr[:,None,:,:], axis=2) # (nelem_tr, quad_num_tr, dim)
                R = 0.2 # radius of the cylindrical surface
                normal = -(XY_GPs - 0.5)/R
                normal = normal.at[:,:,2].set(0.0) # set z-component to 0 / (nelem_tr, npe_tr=3, var=3)
                cauchy_stress = vv_compute_cauchy_stress(XY_GPs, config["MATERIAL"]["E"], config["MATERIAL"]["nu"], device_main) # (nelem_tr, quad_num_tr, dim, var)
                force = jnp.sum(cauchy_stress * normal[:,:,None,:], axis=3, keepdims=True) # (nelem, quad_num, var, 1)
                NP = jnp.sum( jnp.matmul(N_trac, force) * JxW_tr[:,:,None,None], axis=1) # (nelem, elem_dof, 1)
            else: # for Abaqus and Ansys, simple constant
                NP = jnp.sum( jnp.matmul(N_trac, force.reshape(-1,1)[None,None,:,:]) * JxW_tr[:,:,None,None], axis=1) # (nelem, elem_dof, 1)
            b = b.at[Elem_dofs_tr.reshape(-1)].add(NP.reshape(-1))  # assemble 
    
    ## Dirichlet BC applided to b vector
    for dirichlet_dof, dirichlet_val in zip(Dirichlet_dof, Dirichlet_val):
        b = b.at[dirichlet_dof].set(large_value * dirichlet_val)

    return shape_grads_physical_list, JxW_list, Elem_dofs_list, D, b


def A_fun_device(sol, shape_grads_physical_list, JxW_list, Elem_dofs_list, D, device):
    """ computation of Ax for each device. All variables should be stored at the same device as the one in the input.
    """
    ## loop over blocks
    Ax = jnp.zeros_like(sol, dtype=jnp.float64, device=device)
    for shape_grads_physical_block, JxW_block, Elem_dofs_block in zip(shape_grads_physical_list, JxW_list, Elem_dofs_list):
        # start_residual = time.time()
        Ax = get_residual(Ax, sol, shape_grads_physical_block, JxW_block, Elem_dofs_block, D)
        # print(f"\t\t{iblock+1}/{nblock} residual took -> {time.time() - start_residual}")
    
    return Ax


def A_fun_singleGPU(sol, shape_grads_physical_list, JxW_list, Elem_dofs_list, D, device_main, Dirichlet):
    """ Single GPU threading to compute residual
    """

    # Compute basic stuffs
    large_value = 1e16
    
    # Function to run on multi GPUs
    Ax = A_fun_device(sol, shape_grads_physical_list, JxW_list, Elem_dofs_list, D, device_main)  # Compute on that device
    
        
    ## Dirichlet boundary condition applied to Ax
    # start_difichlet = time.time()
    for dirichlet_dof in Dirichlet['dof']:
        Ax = Ax.at[dirichlet_dof].set(large_value * sol[dirichlet_dof])
    # print(f"\n\t\tDirichlet BC took -> {time.time() - start_difichlet}")


    return Ax



####################### Post-processing #######################


def get_error_norm(sol, XY, XY_host, Elem_nodes, Elem_nodes_host, Elem_dofs_host, D,
                    shape_vals_FEM, shape_vals, shape_grads_physical_host, JxW_host, max_array_size_block, config, device_main):
    """ Compute the von Mises stress for each element
    --- inputs ---
    sol: (dof_global,) solution vector
    XY: (nnode, dim) nodal coordinates
    Elem_nodes: (nelem, nodes_per_elem)
    elem_type: string, element type
    Gauss_num_norm: int, number of gauss points for the element
    quad_num_norm: int, number of quadrature points for the element
    max_array_size_block: int, maximum size of the array for block
    """

    
    ## divide into blocks
    size_largest_array = (shape_grads_physical_host.shape[0]*shape_grads_physical_host.shape[1]*
                              shape_grads_physical_host.shape[2]*shape_grads_physical_host.shape[3])
    nblock = int(size_largest_array // (max_array_size_block) + 1)
    shape_grads_physical_host_list = np.array_split(shape_grads_physical_host, nblock, axis=0)
    JxW_host_list = np.array_split(JxW_host, nblock, axis=0)
    Elem_dofs_host_list = np.array_split(Elem_dofs_host, nblock, axis=0)
    Elem_nodes_list = jnp.array_split(Elem_nodes, nblock, axis=0)
    Elem_idx_list = jnp.array_split(jnp.arange(shape_grads_physical_host.shape[0], dtype=jnp.int64), nblock, axis=0) # global element index divided into nblocks
    
    ## Compute mises stress per quadrature point (with coordinates)
    L2_norm_num, L2_norm_denom = 0.0, 0.0 # initialize L2 norm numerator and denominator
    energy_norm_num, energy_norm_denom = 0.0, 0.0 # initialize energy norm numerator and denominator
    for shape_grads_physical_block_host, JxW_block_host, Elem_dofs_block_host, Elem_nodes_block, Elem_idx_block in zip(
        shape_grads_physical_host_list, JxW_host_list, Elem_dofs_host_list, Elem_nodes_list, Elem_idx_list):
        
        ## Bring useful constants
        nelem_block, quad_num, npe, dim = shape_grads_physical_block_host.shape
        
        ## Convert to jnp array
        shape_grads_physical_block = jnp.array(shape_grads_physical_block_host, device=device_main)
        JxW_block = jnp.array(JxW_block_host, device=device_main) # (nelem_block, quad_num)
        Elem_dofs_block = jnp.array(Elem_dofs_block_host, device=device_main)
        
        ## Compute FEM values
        Bmat_block = get_Bmat_block(shape_grads_physical_block, Elem_dofs_block) # (nelem_block, quad_num, 6, elem_dof)
        u_elem_block = jnp.take(sol, Elem_dofs_block, axis=0) # (nelem_block, elem_dof)
        strain = jnp.sum(Bmat_block * u_elem_block[:,None,None,:], axis=3) # (nelem_block, quad_num, 6), strain
        stress = jnp.sum(D[None,None,:,:] * strain[:,:,:,None], axis=2) # (nelem_block, quad_num, 6), stress

        ## Compute exact values
        XY_elem = jnp.take(XY, Elem_nodes_block, axis=0) # (nelem_block, nodes_per_elem, dim)
        XY_GPs = jnp.sum(shape_vals_FEM[None,:,:,None] * XY_elem[:,None,:,:], axis=2) # (nelem_per_block, quad_num, dim)
        u_exact = vv_u_exact(XY_GPs, device_main) # (nelem_block, quad_num, dim), exact solution at Gauss points

        u_elem_block = u_elem_block.reshape(nelem_block, npe, -1) # (nelem_block, npe, var)
        if shape_vals_FEM.shape == shape_vals.shape: # for FEM
            u_block = jnp.sum(shape_vals_FEM[None,:,:,None] * u_elem_block[:,None,:,:], axis=2) # (nelem_block, quad_num, dim)    
        else: # for C-HiDeNN
            u_block = jnp.sum(shape_vals[Elem_idx_block,:,:,None] * u_elem_block[:,None,:,:], axis=2) # (nelem_block, quad_num, dim)
        cauchy_stress_exact = vv_compute_cauchy_stress(XY_GPs, config["MATERIAL"]["E"], config["MATERIAL"]["nu"], device_main)  # (nelem_block, quad_num, 3, 3)
        
        ### Convert cauchy_stress_exact (tensor) to Voigt notation (6,)
        def tensor_to_voigt(stress_tensor):
            # stress_tensor: (..., 3, 3)
            s = stress_tensor
            return jnp.stack([
            s[..., 0, 0],  # σ_xx
            s[..., 1, 1],  # σ_yy
            s[..., 2, 2],  # σ_zz
            s[..., 0, 1],  # σ_xy
            s[..., 1, 2],  # σ_yz
            s[..., 0, 2],  # σ_xz
            ], axis=-1)
        
        ## Compute error norms
        D_inv = jnp.linalg.inv(D) # (6,6)
        stress_exact = tensor_to_voigt(cauchy_stress_exact) # (nelem_block, quad_num, 6)
        strain_exact = jnp.sum(D_inv[None,None,:,:] * stress_exact[:,:,:,None], axis=2) # (nelem_block, quad_num, 6), strain exact
        stress_error = stress - stress_exact # (nelem_block, quad_num, 6)
        strain_error = jnp.sum(D_inv[None,None,:,:] * stress_error[:,:,:,None], axis=2) # (nelem_block, quad_num, 6), strain error
        
        energy_norm_num += jnp.sum((0.5 * jnp.sum(strain_error * stress_error, axis=2)) * JxW_block[:,:])
        energy_norm_denom += jnp.sum((0.5 * jnp.sum(strain_exact * stress_exact, axis=2)) * JxW_block[:,:]) # (nelem_block, quad_num)

        L2_norm_num += jnp.sum(jnp.sum((u_block - u_exact)**2, axis=2) * JxW_block[:,:]) # (nelem_block, quad_num)
        L2_norm_denom += jnp.sum(jnp.sum(u_exact**2, axis=2) * JxW_block[:,:]) # (nelem_block, quad_num)    

    # Compute L2 norm
    L2_norm = jnp.sqrt(L2_norm_num / L2_norm_denom)
    print(f"L2 norm of the error: {L2_norm:.4e}")
    # Compute energy norm
    energy_norm = jnp.sqrt(energy_norm_num / energy_norm_denom)
    print(f"Energy norm of the error: {energy_norm:.4e}")
    return energy_norm