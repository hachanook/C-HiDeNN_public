import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax.experimental.sparse import BCOO
from jax.scipy.optimize import minimize
# import jaxopt
import scipy
from scipy.sparse import csc_matrix, csr_matrix
import scipy.sparse
# import td_solver
import sys, time


########################################################################


def solve_iter(A_sp_scipy, b, max_array_size_block=0):
    """ FEM iterative solver
    """
    
    # start_solve = time.time() 
    size_A = A_sp_scipy.nnz
    print(f"\tA_sp matrix has {size_A:.2e} components")
    dof_global = len(b)
    try:
        # Try to run on GPU
        sol = jnp.zeros(dof_global, dtype=jnp.float64)              # (dof,)
        A_sp = BCOO.from_scipy_sparse(A_sp_scipy)
        print("\tIterative CG solver on GPU")
        sol, info = jax.scipy.sparse.linalg.cg(A_sp, b, x0=sol, tol=1e-10) # , maxiter=maxiter, tol=tol
        # def A_fun(x):
        #     return A_sp@x
        # sol, info = jax.scipy.sparse.linalg.cg(A_fun, b, x0=sol, tol=1e-10) # , maxiter=maxiter, tol=tol
        sol_host = np.array(sol)

        ## debug
        norm_before = jnp.linalg.norm(b)
        res_after = A_sp @ sol - b
        norm_after = jnp.linalg.norm(res_after)
        print(f"\tIterative solver res change: from {norm_before:.2e} to {norm_after:.2e}")
        
        
    except Exception:
        # If an error occurs, print the error message and execute operation B
        print("\tIterative CG solver on CPU")
        sol_host = np.zeros(dof_global, dtype=np.float64)              # (dof,)
        sol_host, info = scipy.sparse.linalg.cg(A_sp_scipy, np.array(b), x0=sol_host, tol=1e-10)
        sol = jnp.array(sol_host)

    # print(f"Iterative solver took: {time.time() - start_solve:.2f} seconds")
    return sol, sol_host


def solve_iter_assemble_free(A_fun, b, x0, 
                             shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist, devices, Dirichlet, tol=1e-5):
    
    sol, step = CG_solver(A_fun, b, x0, 
                          shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist, devices, Dirichlet, tol=tol)
    sol_host = np.array(sol)
    ## debug
    norm_before = jnp.linalg.norm(b)
    res_after = b - A_fun(sol, shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist, devices, Dirichlet)
    norm_after = jnp.linalg.norm(res_after)
    print(f"\tConverged at step {step}. Residual changed: from {norm_before:.2e} to {norm_after:.2e}")
    return sol, sol_host


## inhouse CG solver
def CG_solver(A_fun, b, sol, 
              shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist, devices, Dirichlet, tol=1e-5):
    # start_step = time.time()
    # tol = jnp.array(tol**2)
    # device_main = devices[0]
    dof_global = len(b)
    tol *= jnp.linalg.norm(b) # normalize the tolerance. Now, the solver stops when the residual becomes 1e-10 times smaller than the initial residual
    r = b - A_fun(sol, shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist, devices, Dirichlet)
    p = r
    rsold = jnp.dot(r,r)
    period = 2000
    start_step = time.time()
    for step in range(dof_global):
        # start_step_ = time.time()
        # start_A_fun = time.time()
        Ap = A_fun(p, shape_grads_physical_dlist, JxW_dlist, Elem_dofs_dlist, D_dlist, devices, Dirichlet)
        # print(f"\tA_fun took -> {time.time() - start_A_fun:.6f} sec")

        # start_before = time.time()
        alpha = rsold / jnp.dot(p, Ap)
        sol += alpha * p
        r -= alpha * Ap
        rsnew = jnp.dot(r,r)
        # print(f"\tstep = {step}, before took {time.time()-start_before:.6f} sec/setp") 
        
        # start_convergence1 = time.time()
        if step%period == 0:
            print(f"\tstep = {step}, res l_2 = {rsnew**0.5:.2e}, took {(time.time()-start_step)/period:.4f} sec/setp") 
            start_step = time.time()
        # print(f"\tstep = {step}, convergence1 took {time.time()-start_convergence1:.6f} sec/setp") 
        # start_convergence2 = time.time()
        if rsnew**0.5 < tol:
            break
        # print(f"\tstep = {step}, convergence2 took {time.time()-start_convergence2:.6f} sec/setp") 

        # start_after = time.time()
        p = r + (rsnew/rsold) * p
        rsold = rsnew
        # print(f"\tstep = {step}, after took {time.time()-start_after:.6f} sec/setp")
        
        # print(f"step = {step} took {time.time()-start_step_:.6f} sec/setp\n")

    # print(f"CG solver took {time.time() - start_step:.4f} seconds\n") 
    return sol, step