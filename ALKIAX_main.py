'''
Date: 10.01.2024
Author: Abdullah Tokmak (abdullah.tokmak@aalto.fi)
This is the main execution file.
'''

from ALKIAX_ground_truth import ground_truth_dimensions, ground_truth_type, ground_truth
from ALKIAX_kernel import matern_kernel
import numpy as np
from ALKIAX_functions import \
    root_node, head_nodes, compute_M_list, relevant_dict_total_f
from ALKIAX_approximation import node_approximation_procedure
import time
import pickle
import concurrent.futures
import os
from anytree import PreOrderIter
import datetime

'''Hyperparameters
epsilon (float): maximum allowed error for the approximation framework. Connection to RMPC: The maximum input error allowed by the input robustness of the MPC scheme.
round_n_digits (int): to omit cases like  0.16666666666 != 0.16666666667. Take a high value to ensure accuracy but stay below machine accurary (e.g. 14)
gt_string (string): String corresponding to ground truth. Currently implemented: 'waterfall_2D', 'CSTR_python', 'plasma_python', 'sine_2D', 'sine_3D'
    ->x_dim (int): state-dimension of the MPC, input-dimension of the approximating function
    ->y_dim (int): input-dimension of the MPC, output-dimension of the approximating function
number_of_head_nodes (int): Initial partitioning of the global domain. Has to be of form i**x_dim, i=1,2,3,4, ...
p_min (int): Minimum number of equidistant samples per axis is given by (1+2**(p_min))**x_dim
cond_max (float): Maximum condition number used throughout the  approximation procedure
C_ell (float): Factor with which the length scale of the kernel will be multiplied.
kernel (ALKIAX_kernel.matern_kernel object): chosen kernel. Current option is only the Matérn kernel.
parallel (boolean): Do we execute ALKIA-X parallelly (True) or sequentially (False)?
max_storage_termination (int): in mega-byte, upper bound on the size of the file where the samples of the approximation procedure are stored.
    -> Additional stopping criterion to epsilon
    -> Useful for capacity constraints of, e.g., microcontrollers
    -> Set max_storage_termination = np.inf to only break when sufficient accuracy is guaranteed
'''
epsilon = 5.1e-3  # max allowed error
round_n_digits = 14  # rounding
gt_string = 'CSTR_python'
x_dim, y_dim = ground_truth_dimensions(gt_string)
number_of_head_nodes = 1
p_min = 2
cond_max = 1.15e8
C_ell = 0.8
kernel = matern_kernel(sigma=1, ell=C_ell, nu=3/2)
parallel = True
max_storage_termination = np.infty


'''
Parameters:
global domain: Unit cube.
lambda_bar and C_Gamma: later added to RKHS norm of the approximating function.
'''

global_domain = np.array(list([[0, 1]]*x_dim))
M_min = (1+2**(p_min))
lambda_bar = 2+(1/(M_min-1))
C_Gamma = epsilon/(2**lambda_bar+1)

'''
C_root: root node corresponding to the global domain
dict_head_nodes: dictionary containing number_of_head_nodes nodes, which are the head nodes. All are direct children of the root node and correspond to the first level of sub-domains.
M_list: List of number of equidistant samples per axis, e.g., [5,9,17,33]. The maximum number ensures that the condition number is bounded by {cond_max}.
'''
C_root = root_node(global_domain)
dict_head_nodes = head_nodes(C_root, number_of_head_nodes, round_n_digits)
M_list = compute_M_list(M_min, cond_max, kernel, C_root)
# For plasma python, you may consider using M_list=[5, 9, 17] directly
# since the computation of the condition number with 33^3 samples might crash.
print(f'M_max={M_list[-1]}')

# Determine type of ground truth
gt_type = ground_truth_type(gt_string)

# Determining the maximum tree depth from max_storage_termination
if max_storage_termination == np.inf:
    max_tree_depth = np.inf
else:
    factor_samples_storage = 1e-5  # number_of_samples * (xdim+ydim) * factor_samples_storage  required_storage (in MB)
    required_storage = 0
    max_number_of_samples = 0
    depth = 0
    while required_storage < max_storage_termination:
        depth += 1
        number_of_samples = 2**(depth*x_dim)*number_of_head_nodes*M_list[-1]**x_dim
        if number_of_samples*factor_samples_storage*(x_dim+y_dim) > max_storage_termination:
            break
        else:
            max_number_of_samples = number_of_samples
            max_tree_depth = depth  # after this depth, stop the approximation procedure
    print(max_tree_depth, max_number_of_samples)

# Initialization of dictionaries and sets
dict_nodes = {}
dict_gt = {}
dict_eng = {}
dict_all_nodes = {}
set_done_nodes = set()


def node_approximation_parallel(input):
    current_node, tree_depth = input
    ''' Approximate the ground truth on the current node
    Args:
        current_node (anytree.node): the node that is approximated
        tree_depth (int): current depth of tree
    Returns:
        list_nodes_to_add (list): nodes that need to be approximated resulting from the approximation of current_node
        node_to_remove (anytree.node): same as current_node; removed after the node approximation is finished.
    '''

    if gt_type == 'toy_example':
        if 'current_gt' in dict_gt.keys():
            gt = dict_gt['current_gt']
        else:
            gt = ground_truth(gt_string)
    elif gt_type == 'MPC Python':
        if len(dict_gt.keys()) == 0:
            gt = ground_truth(gt_string)  # go without dictionary
            gt.get_solver()
            dict_gt['current_gt'] = gt
        else:
            gt = dict_gt['current_gt']
    list_nodes_to_add, node_to_remove = node_approximation_procedure(current_node, gt,
                                        epsilon, C_Gamma, round_n_digits,
                                        M_list, C_ell, kernel,
                                        x_dim, y_dim, max_tree_depth, tree_depth)
    return list_nodes_to_add, node_to_remove


'''Execution
Sequentially go through each head node.
Create a list of tasks that contains the nodes that we want to approximate on that level.
After each level, we delete unnecessary data, remove the approximated node, and add the new nodes.
If one head node is done, we save it locally and delete it from the variables.
After all head nodes are done, load the head nodes again and build the parent relationship between the head nodes and the root node.
Then, we do the post processing, sample the ground truth, and heuristically determine the maximal error.
Finally, save important values in the mat file for further usage.
'''

current_time = time.time()
try:
    os.mkdir('head_nodes')
except FileExistsError:
    pass
head_node_names = []
level_head = 0
for head_node in dict_head_nodes.values():
    level = 0
    level_head += 1
    dict_nodes = {}
    dict_nodes[head_node.name] = head_node
    while len(dict_nodes) > 0:
        now = datetime.datetime.now()
        print(f'At head node: {level_head}. Now at depth {level} with {len(dict_nodes)} tasks. The time of day is: {now.time()}')
        input = [(list(dict_nodes.values())[i], level) for i in range(len(dict_nodes.values()))]
        if parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(node_approximation_parallel, input)
            list_of_results = list(results)
        else:  # especially important for debugging
            list_of_results = []
            for task in input:
                result = node_approximation_parallel(task)
                list_of_results.append(result)
        level += 1
        for result in list_of_results:
            nodes_to_add = result[0]
            node_to_delete = result[1]
            # Ad the children to the list of tasks
            for i in range(len(nodes_to_add)):
                node_to_add = nodes_to_add[i]
                node_to_add.dict_total_f = relevant_dict_total_f(node_to_add, node_to_delete, round_n_digits)
                dict_nodes[node_to_add.name] = node_to_add

            # Delete unnecessary stuff
            del node_to_delete.__dict__['dict_total_f']
            if not node_to_delete.is_head:
                node_to_delete_name_backwards = node_to_delete.name[::-1]
                node_to_delete_parent_name = node_to_delete_name_backwards[node_to_delete_name_backwards.find('_'):][1:][::-1]
                node_to_delete.parent = dict_all_nodes[node_to_delete_parent_name]
                # No need to investigate the parent node anymore
                set_done_nodes.add(node_to_delete_parent_name)
            dict_all_nodes[node_to_delete.name] = node_to_delete
            # Delete node_to_delete from dict_nodes, which contains the tasks
            del dict_nodes[node_to_delete.name]
        for done_node in set_done_nodes:
            if not dict_all_nodes[done_node].is_head:
                # Delete done nodes from all nodes
                del dict_all_nodes[done_node]
            set_done_nodes = set()
        # TODO: Difference between dict_nodes and dict_all_nodes and the necessaty?
    # Delete 'infeasible_domain' and 'name' entries for nodes to delete
    if not node_to_delete.is_head:
        del node_to_delete.parent.__dict__['infeasible_domain']
        # del node_to_delete.parent.__dict__['name']
    for current_node in PreOrderIter(dict_all_nodes[head_node.name]):
        if not current_node.is_head and not current_node.is_leaf:
            del current_node.__dict__['is_head']

    # Saving the head nodes
    with open(f'./head_nodes/{head_node.name}.pickle', 'wb') as handle:
        pickle.dump(dict_all_nodes[head_node.name], handle)
    # Remove saved head node
    del dict_all_nodes[head_node.name]
    head_node_names.append(head_node.name)
    del locals()['head_node']

print('Sampling procedure done!')
# Loading all head nodes and removing them from local folder
for head_node_name in head_node_names:
    with open(f'./head_nodes/{head_node_name}.pickle', 'rb') as handle:
        dict_head_nodes[head_node_name] = pickle.load(handle)
        os.remove(f'./head_nodes/{head_node_name}.pickle')
# Creating the parent-relationship between root node and all head nodes
for head_node in dict_head_nodes.values():
    head_node.parent = C_root
# Saving the root node before post-processing
time_elapsed_total = time.time() - current_time
print(f'Total time: {time_elapsed_total}')
C_root.time_elapsed = time_elapsed_total
C_root.epsilon = epsilon
#with open('C_root_sine_2D.pickle', 'wb') as handle:
#    pickle.dump(C_root, handle, protocol=pickle.HIGHEST_PROTOCOL)
