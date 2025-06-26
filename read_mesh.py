import numpy as np
import os
import time
import re # Regular espression operations.

# Read mesh from abaqus input file
# written by Chanwook Park (chanwookpark2024@u.northwestern.edu)

def read_mesh_Abaqus_3D(parent_dir, inp_name):
    """ Reads 3D abaqus input file. Currently only supports reading mesh info. Not boundary conditions
    --- inputs --- 
    parent_dir
    inp_name
    --- outputs ---
    iXY: (nnode, 1+dim), the first column is node index
    iElem_nodes: (nelem, 1+nodes_per_elem), elemental node indices, the first column is node index
    """
    # start_time = time.time()
    
    path = os.path.join(parent_dir, 'input_files/Abaqus')
    path = os.path.join(path, inp_name+".inp")
    file = open(path, 'r')
    lines = file.readlines()
    list_XY = []
    iElem_nodes_list, elem_type_list = [], []
    node_bool, element_bool = True, True
    MATERIAL = {}
    BC = {"BC_disp":[], "BC_trac":[]}
    for count, line in enumerate(lines):
        if '*Node' in line and node_bool: # distinguish from reference point node
            count += 1
            node_bool = True
            while node_bool:
                if '*Element' in lines[count+1]:
                    node_bool = False
                line_list =  [float(item.strip()) for item in lines[count].strip().split(',')] 
                list_XY.append(line_list[0:]) 
                count += 1
            iXY = np.array(list_XY, dtype=np.float64)
            iXY[:,0] -= 1 # for indexing
        
        if '*Element' in line and element_bool:
            match = re.search(r'type=([^\n]+)', line)
            elem_type = match.group(1)
            elem_type_list.append(elem_type)
            # print(elem_type)
            count += 1
            elem_bool = True
            list_elem_nodes = []
            while elem_bool:
                if (lines[count+1].startswith('*') and elem_type != 'C3D20') or (
                    lines[count+2].startswith('*') and elem_type == 'C3D20'):
                    elem_bool = False
                if lines[count][-2] == ',':
                    line_elem = lines[count] + lines[count+1]
                    count += 2
                else:
                    line_elem = lines[count]
                    count += 1
                line_list = line_list = [int(item.strip()) for item in line_elem.strip().split(',')]
                
                list_elem_nodes.append(line_list[0:])
            iElem_nodes = np.array(list_elem_nodes, dtype=np.int64) - 1 # for python numbering
            iElem_nodes_list.append(iElem_nodes)
            # print(iElem_nodes.shape)
        if '*End Part' in line: node_bool, element_bool = False, False
        
        if '*End Instance' in line:
            count += 2
            set_bool = True
            nsets = {}
            elsets = {}
            current_set = None
            current_type = None
            while set_bool:
                if '*End Assembly' in lines[count+1]:
                    set_bool = False
                
                line = lines[count].strip()
                if line.startswith('*Nset'):
                    match = re.search(r'nset=([^,]+)', line)
                    if match:
                        set_name = match.group(1)
                        if 'generate' in lines[count]:
                            set_name += '_generate'
                        nsets[set_name] = []
                        current_set = set_name
                            
                        current_type = 'nset'
                elif line.startswith('*Elset'):
                    match = re.search(r'elset=([^,]+)', line)
                    if match:
                        set_name = match.group(1)
                        if 'generate' in lines[count]:
                            set_name += '_generate'
                        elsets[set_name] = []
                        current_set = set_name
                        current_type = 'elset'
                elif line.startswith('*'):
                    current_set = None
                    current_type = None
                elif current_set:
                    indices = [int(x) for x in line.split(',') if x.strip()]
                    if current_type == 'nset':
                        nsets[current_set].extend(indices)
                    elif current_type == 'elset': 
                        elsets[current_set].extend(indices)                
                count += 1
            
            ## convert from list to array with python indexing (-1)
            for key, value in nsets.items():
                value_np = np.array(value, dtype=np.int64)
                if 'generate' in key: # generate sets
                    value_np = np.arange(value_np[0],value_np[1]+1,value_np[2], dtype=np.int64)
                nsets[key] = value_np - 1
            # Remove "_generate" from any nset keys
            keys_to_modify = [key for key in nsets if key.endswith("_generate")]
            for key in keys_to_modify:
                new_key = key.replace("_generate", "")
                nsets[new_key] = nsets.pop(key)

            for key, value in elsets.items():
                value_np = np.array(value, dtype=np.int64)
                if 'generate' in key: # generate sets
                    value_np = np.arange(value_np[0],value_np[1]+1,value_np[2], dtype=np.int64)
                elsets[key] = value_np - 1
            
            # Remove "_generate" from any nset keys
            keys_to_modify = [key for key in elsets if key.endswith("_generate")]
            for key in keys_to_modify:
                new_key = key.replace("_generate", "")
                elsets[new_key] = elsets.pop(key)
            continue
        
        ## read material properties
        if line.strip().startswith('** MATERIALS'):
            MATERIAL_bool = True
            while MATERIAL_bool:
                count += 1
                line = lines[count].strip()
                if line.startswith('*Material'):
                    count += 1
                    line1 = lines[count].strip()
                    count += 1
                    line = lines[count].strip()
                    line2 = [x.strip() for x in line.split(',')]
                    MATERIAL["Type"] = line1[1:]
                    MATERIAL["E"] = float(line2[0])
                    MATERIAL["nu"] = float(line2[1])
                next_line = lines[count+1].strip()
                if next_line.startswith('** BOUNDARY CONDITIONS'):
                    MATERIAL_bool = False
                

        ## read displacement boundary conditions
        if line.strip().startswith('** BOUNDARY CONDITIONS'):
            disp_BC_bool = True
            while disp_BC_bool:
                
                # Read the next line for node set and type
                count += 1
                line = lines[count].strip()
                if line.startswith('** Name'):
                    
                    line1 = [x.strip() for x in line.split(' ')]
                    count += 2
                    line = lines[count].strip()
                    line2 = [x.strip() for x in line.split(',')]
                    BC['BC_disp'].append({"Name": line1[2], 
                                          "Type": line1[4],
                                          "nset": line2[0],
                                          "Type_detail": line2[1]})
                next_line = lines[count+1].strip()
                if next_line.startswith('** ---'):
                    disp_BC_bool = False
                    
        ## read traction boundary conditions
        if line.strip().startswith('** LOADS'):
            trac_bool = True
            while trac_bool:

                # Read the next line for node set and type
                count += 1
                line = lines[count].strip()
                if line.startswith('** Name'):
                    
                    line1 = [x.strip() for x in line.split(' ')]
                    count += 2
                    line = lines[count].strip()
                    line2 = [x.strip() for x in line.split(',')]
                    BC['BC_trac'].append({"Name": line1[2],
                                          "elset": line2[0],
                                          "Type": line2[1],
                                          "Magnitude": float(line2[2]),
                                          "Vector": [float(line2[3]), float(line2[4]), float(line2[5])]})
                next_line = lines[count+1].strip()
                if next_line.startswith('** OUTPUT REQUESTS'):
                    trac_bool = False

                #     nset, bc_type = parts
                #     # Store in dictionary with upper-case type
                #     disp1['disp1'] = {"nset": nset, "type": bc_type.upper()}
                # break  # Remove if you expect multiple BCs
    
    
    return iXY, iElem_nodes_list, elem_type_list, nsets, elsets, MATERIAL, BC
