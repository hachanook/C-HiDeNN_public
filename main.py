import os, sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import scipy
import scipy.sparse
import yaml
import time 
import importlib.util
import pyvista as pv

import FEM, solver, gauss, read_mesh

if importlib.util.find_spec("GPUtil") is not None: # for linux & GPU
    ''' If you are funning on GPU, please install the following libraries on your anaconda environment via 
    $ conda install -c conda-forge humanize
    $ conda install -c conda-forge psutil
    $ conda install -c conda-forge gputil
    ''' 
    import humanize, psutil, GPUtil
    
    # memory report
    def mem_report(num, gpu_idx):
        ''' This function reports memory usage for both CPU and GPU'''
        print(f"\t-{num}-CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
         
        GPUs = GPUtil.getGPUs()
        gpu = GPUs[gpu_idx]
        # for i, gpu in enumerate(GPUs):
        print('\t---GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%\n'.format(gpu_idx, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))


#####################  Input files ###################

### Ansys input files

### Abaqus input files

## regular mesh convergence study
iterate = ["convergence_reg_C3D4_seed01"]

########################### User defined parameters ###############################

software = 'Abaqus' # 'Ansys' or 'Abaqus'
# software = 'Ansys' # 'Ansys' or 'Abaqus'
s_patch = 1 
p_order = 1
# bool_s_geq_p2 = True # s >= p/2
bool_s_geq_p2 = False # s >= p
activation = 'polynomial'

######################## Set computing resources ##################################

## Set GPU index    
gpu_idx = 0

## Set running types: matrix-free by default
run_FEM = True
# run_FEM = False

run_C_HiDeNN = True
# run_C_HiDeNN = False 

## Set plot
bool_plot = True
# bool_plot = False

###########################################################

# Create required directories if they do not exist
for subdir in ["Abaqus", "Ansys"]:
    input_dir = os.path.join("input_files", subdir)
    output_dir = os.path.join("output_files", subdir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

## memory assignment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing
devices = jax.devices()
device_main = None
try:
    gpus = jax.devices("gpu")
    if gpus:
        device_main = gpus[0]
    else:
        tpus = jax.devices("tpu")
        if tpus:
            device_main = tpus[0]
except RuntimeError:
    # Only CPU is available
    device_main = None

if importlib.util.find_spec("GPUtil") is not None:    
    GPUs = GPUtil.getGPUs()
    if len(GPUs) > 1: # Running on Athena
        max_array_size_block = 5e7  # 5e8 for Athena
    else: # Running on laptop GPU
        max_array_size_block = 5e6  #5e7 for laptop
else: # Running on laptop CPU or laptop GPU
    max_array_size_block = 1e6
    


###########################################################

for input_name in iterate:
    
    ## Make a configuration dictionary
    config = {}
    config["INPUT_PARAM"] = {"input_name": input_name,
                            "software": software, 
                             "dim": 3, # 2D or 3D
                             "var": 3, # number of variables per node (e.g., 3 for displacement u,v,w)
                            }
    config["C_HiDeNN"] = {  "bool_s_geq_p2": bool_s_geq_p2,
                            "s_patch": s_patch, 
                            "alpha_dil": 40.0, # dilation parameter for C-HiDeNN
                            "p_order": p_order, 
                             "radial_basis": "cubicSpline", 
                             "activation": activation}
    
    ## Read mesh
    parent_dir = os.path.abspath(os.getcwd())
    if config["INPUT_PARAM"]["software"] == "Abaqus":
        iXY, iElem_nodes_list, elem_type_list, nsets, elsets, MATERIAL, BC = read_mesh.read_mesh_Abaqus_3D(parent_dir, input_name)
        part_idx = 0 # index of part in the Abaqus assembly
        XY_host = iXY[:,1:]
        Elem_nodes_host = iElem_nodes_list[part_idx][:,1:]
        XY = jnp.array(XY_host, device=device_main)
        Elem_nodes = jnp.array(Elem_nodes_host, device=device_main)
        elem_type = elem_type_list[part_idx]
        config["MATERIAL"] = MATERIAL
        config["BC"] = BC # boundary conditions in Abaqus
        config["nsets"] = nsets # nodal sets in Abaqus; used for boundary conditions
        config["elsets"] = elsets # elemental sets in Abaqus; used for boundary conditions
        config["elem_type"] = elem_type
        ## read boundary conditions
        Dirichlet, Neumann = FEM.get_BCs_Abaqus(XY, Elem_nodes, config, device_main)
        

    nodes_per_elem = Elem_nodes_host.shape[1]
    config["nodes_per_elem"] = nodes_per_elem

    dim = config["INPUT_PARAM"]["dim"]
    var = config["INPUT_PARAM"]["var"]
    elem_dof = nodes_per_elem * var

    nnode, nelem, dof_global = len(XY_host), len(Elem_nodes_host), len(XY_host) * config["INPUT_PARAM"]["var"]
    config["nnode"] = nnode
    config["nelem"] = nelem
    config["dof_global"] = dof_global


    ## define quardature rule
    Gauss_num_FEM, Gauss_num_CFEM, Gauss_num_norm, Gauss_num_tr, quad_num_FEM, quad_num_CFEM, quad_num_norm, quad_num_tr = gauss.get_Gauss_num(elem_type, config["INPUT_PARAM"]["dim"])
    config["Gauss"] = {"Gauss_num_FEM":Gauss_num_FEM, "Gauss_num_CFEM": Gauss_num_CFEM, 
                        "Gauss_num_norm":Gauss_num_norm, "Gauss_num_tr": Gauss_num_tr,
                        "quad_num_FEM": quad_num_FEM, "quad_num_CFEM": quad_num_CFEM,
                        "quad_num_norm": quad_num_norm, "quad_num_tr": quad_num_tr}



    if run_FEM:
        print(f"\n---------------- FEM {input_name}, nelem: {nelem}, dof_global: {dof_global} --------------")

        ## get shape functions and store them in CPU memory
        start_shape = time.time()
        shape_vals, shape_grads_physical_host, JxW_host = FEM.get_FEM_shape_fun(XY, Elem_nodes, elem_type, 
                                                            Gauss_num_FEM, quad_num_FEM, max_array_size_block, device_main)
        Elem_dofs_host = FEM.get_connectivity(Elem_nodes_host, config["INPUT_PARAM"]["var"]) 
        
        shape_fun_dicts_FEM = {} # define shape function dictionary. 
        # The indexing [0] refers to the first part.
        # If the FEM mesh consists of multiple parts, it will have multiple indexing.
        shape_fun_dicts_FEM[0] = {'type': 'FEM',
                                  'shape_vals': shape_vals,
                                'shape_grads_physical_host': shape_grads_physical_host,
                                'JxW_host': JxW_host,
                                'Elem_dofs_host': Elem_dofs_host,
                                }
        print(f"FEM shape fun took: {time.time() - start_shape:.2f} seconds")


        ## Solve with assembly-free solver
        start_solve = time.time()

        shape_grads_physical_list, JxW_list, Elem_dofs_list, D, b = FEM.get_list_and_b(XY, Elem_nodes, shape_fun_dicts_FEM, config,
                                                                    Dirichlet, Neumann, max_array_size_block, device_main) # singleGPU

        x0 = FEM.get_x0_Dirichlet(dof_global, Dirichlet, device_main)
        sol_FEM, sol_FEM_host = solver.solve_iter_assemble_free(FEM.A_fun_singleGPU, b, x0,
                            shape_grads_physical_list, JxW_list, Elem_dofs_list, D, device_main, Dirichlet, tol=1e-10) # singleGPU
        print(f"FEM assembly-free iterative solver took: {time.time() - start_solve:.2f} seconds")


        if bool_plot:
            
            ### Post-processing

            if "convergence" in input_name:
                
                ## compute shape functions for Gauss_num_norm
                shape_vals, shape_grads_physical_host, JxW_host = FEM.get_FEM_shape_fun(XY, Elem_nodes, elem_type, 
                                                            Gauss_num_norm, quad_num_norm, max_array_size_block, device_main)
                ## Measure mises stress at Gauss points norm
                energy_norm = FEM.get_error_norm(sol_FEM, XY, XY_host, Elem_nodes, Elem_nodes_host, 
                                                Elem_dofs_host, D, shape_vals, shape_vals, shape_grads_physical_host, JxW_host, 
                                                max_array_size_block, config, device_main)
                
                
            ## Save results as .vtk format
            uvw_host = sol_FEM_host.reshape(nnode,var) # (nnode, var)
            cell_array = np.hstack([np.full((nelem, 1), nodes_per_elem), Elem_nodes_host])  # Add cell size prefix
            cell_array = cell_array.flatten()
            
            if "C3D4" in elem_type:
                cell_types = np.full(nelem, pv.CellType.TETRA, dtype=np.uint8)
            elif "C3D8" in elem_type:
                cell_types = np.full(nelem, pv.CellType.HEXAHEDRON, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cell_array, cell_types, XY_host)
            
            ### Add scalar values (displacement) to nodes
            grid.point_data["u"] = uvw_host[:,0]
            grid.point_data["v"] = uvw_host[:,1]
            grid.point_data["w"] = uvw_host[:,2]
            
            if "convergence" in input_name:
                # Compute exact displacement using v_u_exact from FEM.py
                uvw_exact = FEM.v_u_exact(XY_host, device_main)
                grid.point_data["u_exact"] = uvw_exact[:, 0]
                grid.point_data["v_exact"] = uvw_exact[:, 1]
                grid.point_data["w_exact"] = uvw_exact[:, 2]


            ### Visualize the mesh
            if importlib.util.find_spec("GPUtil") is None:    
                plotter = pv.Plotter()
                plotter.add_mesh(grid, show_edges=True, opacity=0.5, scalars="u", cmap="coolwarm", scalar_bar_args={"title": "Disp u"})
                plotter.show_axes()  # Add XYZ axis
                plotter.show()
            
            ### Export the mesh
            grid.save(os.path.join(parent_dir, f'output_files/{config["INPUT_PARAM"]["software"]}/{input_name}_FEM.vtk'))


    if run_C_HiDeNN:

        ## Get C-HiDeNN parameters
        if config["C_HiDeNN"]["bool_s_geq_p2"]: # s >= p/2 for efficiency
            s_patch = config["C_HiDeNN"]["p_order"] + 1
            config["C_HiDeNN"]["s_patch"] = s_patch # new rule!
            
        alpha_dil = config["C_HiDeNN"]["alpha_dil"]
        # d_c = Convolution.get_characteristic_length(XY, Elem_nodes)
        # a_dil = float(d_c * alpha_dil)
        p_order = config["C_HiDeNN"]["p_order"]
        p_dict={0:0, 1:4, 2:10, 3:20, 4:35}
        # mbasis = p_dict[p_order]
        activation = config["C_HiDeNN"]["activation"]
        print(f"\n---------------- C-HiDeNN {input_name}, nelem: {nelem}, dof_global: {dof_global}, s_patch: {s_patch}, p_order: {p_order}, activation: {activation} --------------")

        
        ## Read saved shape functions (optional)
        if software == "Abaqus":
            save_dir = os.path.join(parent_dir, "input_files", "Abaqus")
        elif software == "Ansys":
            save_dir = os.path.join(parent_dir, "input_files", "Ansys")
        else:
            save_dir = parent_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{input_name}_C-HiDeNN_s{s_patch}_p{p_order}.npz")
        data = np.load(save_path, allow_pickle=True)
        shape_fun_dicts_CFEM = data['shape_fun_dicts_CFEM'].tolist()
        
        
        ## Solve C-HiDeNN
        start_solve = time.time()
        Grad_N_til_list, JxW_list, Elem_patch_dofs_list, D, b = FEM.get_list_and_b(XY, Elem_nodes, shape_fun_dicts_CFEM, config,
                                                                    Dirichlet, Neumann, max_array_size_block, device_main) # singleGPU
            
        x0 = FEM.get_x0_Dirichlet(dof_global, Dirichlet, device_main)
        sol_CFEM, sol_CFEM_host = solver.solve_iter_assemble_free(FEM.A_fun_singleGPU, b, x0,
                            Grad_N_til_list, JxW_list, Elem_patch_dofs_list, D, device_main, Dirichlet, tol=1e-10) # singleGPU
        print(f"C-HiDeNN assembly-free iterative solver took: {time.time() - start_solve:.2f} seconds")
        # mem_report('After solving C-HiDeNN', gpu_idx)


        if bool_plot:
            
            ### Post-processing
            if "convergence" in input_name:
                
                ## compute shape functions for Gauss_num_norm
                shape_vals, shape_grads_physical_host, JxW_host = FEM.get_FEM_shape_fun(XY, Elem_nodes, elem_type, 
                                                            Gauss_num_norm, quad_num_norm, max_array_size_block, device_main)
                ## Measure mises stress at Gauss points norm
                energy_norm = FEM.get_error_norm(sol_CFEM, XY, XY_host, Elem_nodes, Elem_nodes_host, 
                                                shape_fun_dicts_CFEM[0]["Elem_dofs_host"], D, shape_vals, 
                                                shape_fun_dicts_CFEM[0]["shape_vals"], shape_fun_dicts_CFEM[0]["shape_grads_physical_host"], JxW_host, 
                                                max_array_size_block, config, device_main)
                


            ## Save results as .vtk format
            uvw_host = sol_CFEM_host.reshape(nnode,var)
            cell_array = np.hstack([np.full((nelem, 1), nodes_per_elem), Elem_nodes_host])  # Add cell size prefix
            cell_array = cell_array.flatten()
            
            if "C3D4" in elem_type:
                cell_types = np.full(nelem, pv.CellType.TETRA, dtype=np.uint8)
            elif "C3D8" in elem_type:
                cell_types = np.full(nelem, pv.CellType.HEXAHEDRON, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cell_array, cell_types, XY_host)
            
            ### Add scalar values (displacement) to nodes
            grid.point_data["u"] = uvw_host[:,0]
            grid.point_data["v"] = uvw_host[:,1]
            grid.point_data["w"] = uvw_host[:,2]

            if "convergence" in input_name:
                # Compute exact displacement using v_u_exact from FEM.py
                uvw_exact = FEM.v_u_exact(XY_host, device_main)
                grid.point_data["u_exact"] = uvw_exact[:, 0]
                grid.point_data["v_exact"] = uvw_exact[:, 1]
                grid.point_data["w_exact"] = uvw_exact[:, 2]

            
            ### Visualize the mesh
            if importlib.util.find_spec("GPUtil") is None:    
                plotter = pv.Plotter()
                plotter.add_mesh(grid, show_edges=True, opacity=0.5, scalars="u", cmap="coolwarm", scalar_bar_args={"title": "Disp u"})
                plotter.show_axes()  # Add XYZ axis
                plotter.show()
            
            ### Export the mesh
            grid.save(os.path.join(parent_dir, f'output_files/{config["INPUT_PARAM"]["software"]}/{input_name}_C-HiDeNN_s{s_patch}_p{p_order}.vtk'))

