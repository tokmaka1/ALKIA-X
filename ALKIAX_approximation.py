# Start: 23.08.2022
# Author: Abdullah Tokmak
# Contact: atokmak@ethz.ch
# Piece-wise defined approximation framework with localized RKHS norms

from ALKIAX_kernel import matern_kernel
import numpy as np
import sys
import pickle
from ALKIAX_functions import cube, grid, update_infeasible_points_dict, \
    create_children, current_node_is_infeasible
import datetime
sys.setrecursionlimit(10000)  # 10.000 instead of 1000


def node_approximation_procedure(current_node, gt, epsilon, C_Gamma,
                                 round_n_digits, M_list, C_ell,
                                 kernel, x_dim, y_dim, max_tree_depth,
                                 tree_depth):
    '''ddddd
    Args:
        current_node (anytee.node): node to approximate
        gt (ALKIAX_ground_truth): ground truth function
        epsilon (float): guaranteed error bound
        C_Gamma (float): parameter for RKHS norm over-estimation
        round_n_digits (int): rounding parameter
        M_list (list): list contaning number of samples per axis
        C_ell (float): lengthscale hyperparameter
        kernel (ALKIAX_kernel): chosen kernel
        x_dim (int): state-dimension of the MPC, input-dimension of the approximating function
        y_dim (int): input-dimension of the MPC, output-dimension of the approximating function
        max_tree_depth (int): maximum tree depth to satisfy memory upper bound
        tree_depth (int): current tree depth

    Returns:
        nodes_to_add (list): list containing nodes to approximate next
        node_to_remove (anytee.node): equal to current node
    '''
    now = datetime.datetime.now()
    print(f'Node {current_node.name} at time {now.time()}')  # visualizing the progress
    ell = C_ell*(current_node.D[0][1] - current_node.D[0][0])  # length scale
    kernel = matern_kernel(sigma=1, ell=ell, nu=3/2)
    nodes_to_add = []
    dict_total_f = current_node.dict_total_f  # dictionary containing function evaluations
    total_fev = 0  # count function evaluations
    current_node.infeasible_domain = 0  # initial assumption: feasible nodes

    # A priori determine value of power function
    dict_P_max = {}
    for M_power in M_list:
        X_S_power, _ = cube(M_power, current_node)
        x_c_S_power = sum(X_S_power)/(2**x_dim)
        P_max = kernel.gaussian_process.power_function()(X_S_power, x_c_S_power)
        dict_P_max[M_power] = P_max

    M = M_list[0]
    X_D = grid(M, current_node)
    if current_node.is_head:
        # If head node, then we need to sample complete X_D
        fX_D, infeasible_points = gt.get_function_values(X_D)
        # Save function evaluation in dictionary dict_total_f
        for i_root in range(fX_D.shape[0]):
            dict_total_f[tuple(np.around(X_D[i_root], round_n_digits))] = \
                [fX_D[i_root]]
        # Update dict_total_f with feasibility information
        dict_total_f = update_infeasible_points_dict(dict_total_f, infeasible_points, X_D, round_n_digits)
        total_fev += len(fX_D)

    else:  # not a head node
        fX_D = np.zeros([M**x_dim, y_dim])
        # For this M, all samples can be obtained from the parent node
        for i_child in range(X_D.shape[0]):
            if tuple(np.around(X_D[i_child], round_n_digits)) \
             in dict_total_f.keys():
                fX_D[i_child] = dict_total_f[tuple(np.around(X_D[i_child], round_n_digits))][0]
            else:
                with open('dict_total_f.pickle', 'wb') as handle:
                    pickle.dump(dict_total_f, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)            
                # print(X_D[i_child])
                raise Exception('Problem with getting the values from parent')
    mean = np.mean(fX_D, axis=0)
    success_flag = 1  # flag to track whether approximation is sufficiently accurate

    # Check feasibility
    if current_node_is_infeasible(X_D, dict_total_f, round_n_digits):
        # No need to create children
        current_node.infeasible_domain = 1
        dict_total_f = {}  # we don't need the values anymore
    else:
        current_node.infeasible_domain = 0

    # For loop: approximate each dimension individually
    for m in range(y_dim):
        if current_node.infeasible_domain == 1:
            break  # break out of the for-loop because the domain is infeasible
        list_gamma_hat_t = [0]*y_dim  # RKHS norm of the shifted localized approximating function + C_Gamma
        list_Gamma_bar = [0]*y_dim  # RKHS norm upper bound on shifted ground truth (heuristic)
        while True:  # Approximation procedure
            if kernel.ell < 1e-12:  # Numerical "error"
                raise Exception('Length scale is too small. O(1e-12).')

            if M == M_list[0] and m == 0:  # First dimension and first step of sampling
                # Reason: Never break out of the while-loop before executing the first step
                for m_gamma in range(y_dim):  # each output dimension
                    # RKHS norm of approximate
                    fX_D_t = fX_D - mean if y_dim == 1 else fX_D[:, m_gamma] - mean[m_gamma]
                    coeffs_t = kernel.gaussian_process.fit(X_D, fX_D_t)
                    gamma_hat_t = kernel.gaussian_process.RKHS_norm(fX_D_t, coeffs_t) + C_Gamma
                    list_gamma_hat_t[m_gamma] = [gamma_hat_t]  # First entry into list
            elif m == 0 and M == M_list[1]:  # First dimension, second step of sampling
                for m_gamma in range(y_dim):  # We already have the samples, determine Gamma_bar for all m dimensions
                    # RKHS norm of approximate and heuristic RKHS norm upper bound
                    fX_D_t = fX_D - mean if y_dim == 1 else fX_D[:, m_gamma] - mean[m_gamma]
                    coeffs_t = kernel.gaussian_process.fit(X_D, fX_D_t)
                    gamma_hat_t = kernel.gaussian_process.RKHS_norm(fX_D_t, coeffs_t) + C_Gamma
                    list_gamma_hat_t[m_gamma].append(gamma_hat_t)  # Add to list
                    tau = np.log(list_gamma_hat_t[m_gamma][1]/list_gamma_hat_t[m_gamma][0])/(1/M_list[0]-1/M_list[1])
                    list_Gamma_bar[m_gamma] = list_gamma_hat_t[m_gamma][0]*np.exp(tau/M_list[0])  # exponential extrapolation
            if M != M_list[0] and dict_P_max[M] * list_Gamma_bar[m] < epsilon:  # one Gamma_bar per output dimension
                # never break out in the first iteration step
                break
                # break out of while loop (current dimension m approximated sufficiently accurate)
                # but still continuing with the for loop (approximate other dimensions)
            M_test = 2*M - 1  # number of equidistant samples for the next step
            if (M_test > M_list[-1]) or (list_Gamma_bar[m] * dict_P_max[M_list[-1]] >= epsilon and tree_depth != max_tree_depth):
                '''If condition for breaking out of while-loop
                First condition: we would need too many samples (condition number)
                Second condition: No chance for accurate enough approximate, even if we continue until M_list[-1]
                BUT: only consider second condition if we are not at the maximum depth level
                Remember: to have a successful approximation, all output dimensions have to be approximated successfully.
                '''
                children_node_list = create_children(current_node, round_n_digits)
                for child in children_node_list:
                    child.is_head = False
                    nodes_to_add.append(child)
                success_flag = 0  # we were not successful to approximate accurate enough
                break  # break out of while-loop
            # Acquire new samples
            M = M_test
            X_D = grid(M, current_node)  # equidistant samples
            fX_D = np.ones([X_D.shape[0], y_dim]) * np.inf
            X_D_to_evaluate = np.zeros_like(X_D)
            index_to_evaluate = 0  # where to evaluate the function?
            for i_x in range(X_D.shape[0]):
                if tuple(np.around(X_D[i_x], round_n_digits)) in dict_total_f.keys():
                    # No need to sample it again
                    fX_D[i_x] = dict_total_f[tuple(np.around(X_D[i_x], round_n_digits))][0]
                else:
                    # We need to sample it again
                    X_D_to_evaluate[index_to_evaluate] = X_D[i_x]
                    index_to_evaluate += 1
            X_D_to_evaluate = X_D_to_evaluate[:index_to_evaluate]
            if index_to_evaluate != 0:  # function evaluation
                fX_D_to_evaluate, infeasible_points = gt.get_function_values(X_D_to_evaluate)
                index_to_evaluate = 0
                for i_f in range(len(fX_D)):
                    if fX_D[i_f, 0] == np.inf:  # add function evaluation here
                        fX_D[i_f] = fX_D_to_evaluate[index_to_evaluate]
                        index_to_evaluate += 1
                        dict_total_f[tuple(np.around(X_D[i_f], round_n_digits))] = [fX_D[i_f]]  # supplement dictionary with new function evaluation
                total_fev += len(fX_D_to_evaluate)  # count new samples
                dict_total_f = update_infeasible_points_dict(dict_total_f, infeasible_points, X_D_to_evaluate, round_n_digits)
    if tree_depth == max_tree_depth:
        nodes_to_add = []  # do not approximate any child nodes!
        success_flag = 1  # save the values
    if current_node.infeasible_domain == 0 and success_flag == 1:
        current_node.fX_D = fX_D
        current_node.X_D = X_D
        current_node.mean = mean  # very important to shift/re-shift later
        current_node.ell = kernel.ell

    current_node.dict_total_f = dict_total_f
    current_node.total_fev = total_fev
    node_to_remove = current_node
    return nodes_to_add, node_to_remove
