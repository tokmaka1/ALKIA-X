from ALKIAX_ground_truth import waterfall_2D, sine_2D, sine_3D, plasma_python, cstr_python, ground_truth
from ALKIAX_plot import plot_gt_2D_CSTR, plot_prediction_2D, plot_gt_2D, plot_subdomains_2D_CSTR, generate_plots_plasma
from ALKIAX_functions import approximation, binary_number, decimal_number, grid
from ALKIAX_kernel import matern_kernel
import numpy as np
from tqdm import tqdm
from anytree import PreOrderIter  # depth-first search
import pickle
import time
from plasma_MPC_get_samples import closed_loop_MPC, plasma_MPC_get_samples
from plasma_MPC_get_solver import plasma_MPC_get_solver
from scipy.io import loadmat


def post_processing(C, round_n_digits, division_points, gt_string):
    '''Determine the shifted localized approximating functions
    Args:
        C (anytree.node): head node corresponding to global domain
        round_n_digits (int): for numerical rounding reasons
        division_points (int): How many equidistant points per axis for evaluation
        gt_string (string): CSTR_python, sine_2D or plasma_python
    Returns:
        hX (array): approx. function evaluated at X_evaluation
        X_evaluation (array): array of equidistant samples
        relevant_fev (int): number of relevant function evaluations
        C (anytree.node): head node containing the local approximating functions
        fX_evaluation (array): ground truth evaluated at X_evaluation
    '''
    dict_f = {}
    x_dim = C.D.shape[0]  # get the domain from the node; no need for it to be a separate input
    relevant_nodes = []  # leaf node and not infeasible domain
    total_fev = 0
    for current_node in PreOrderIter(C):
        if not current_node.is_root:
            total_fev += current_node.total_fev
        if current_node.is_leaf and not current_node.infeasible_domain:
            relevant_nodes.append(current_node)
            for i in range(current_node.X_D.shape[0]):
                if tuple(np.around(current_node.X_D[i], round_n_digits)) not in dict_f.keys():
                    dict_f[tuple(np.around(current_node.X_D[i], round_n_digits))] = current_node.fX_D[i]
    y_dim = relevant_nodes[0].fX_D.shape[1]
    relevant_fev = len(dict_f.keys())  # the functions evaluations that we use
    print(f'Relevant fev: {relevant_fev}')
    print(f'Total fev: {total_fev}')

    for current_node in tqdm(relevant_nodes):
        # Find the samples that belong to that one cube
        chosen_kernel = matern_kernel(sigma=1, ell=current_node.ell, nu=3/2)
        piecewise_approximation_t = {}  # dict to save the shifted/transformed approx. functions
        M = int(round((current_node.X_D.shape[0]**(1/x_dim))))
        number_of_functions_per_axis = M - 1
        number_of_functions = number_of_functions_per_axis**x_dim
        for decimal_number_function in range(number_of_functions):  # iterate through each function
            binary_number_function = binary_number(decimal_number=decimal_number_function, x_dim=x_dim, base=number_of_functions_per_axis)
            X_S_binary = [0]*2**x_dim  # binary numbers for sorting reasons
            for i in range(len(X_S_binary)):
                binary_number_addition = binary_number(decimal_number=i, x_dim=x_dim, base=2)
                X_S_binary[i] = list(np.array(binary_number_function) + np.array(binary_number_addition))
            decimal_numbers_X_S = [0]*2**x_dim  # we have 2**x_dim points within a cube
            for i in range(len(decimal_numbers_X_S)):
                # corresponding decimal number to obtain that sample from X_D
                decimal_numbers_X_S[i] = decimal_number(X_S_binary[i], base=M)
            # we now need the 2**x_dim points of X and f following that decimal number
            piecewise_approximation_t[decimal_number_function] = [0]*y_dim  # One function per output dimension
            X_S = current_node.X_D[decimal_numbers_X_S]  # these 2**x_dim samples are needed to build the approx. function
            for m in range(y_dim):
                fX_S_t = current_node.fX_D[decimal_numbers_X_S, m] - current_node.mean[m] if len(range(y_dim)) > 1 else current_node.fX_D[decimal_numbers_X_S, m] - current_node.mean
                coeffs_t = chosen_kernel.gaussian_process.fit(X_S, fX_S_t)  # get coefficients of RKHS function
                hX_S_t = chosen_kernel.gaussian_process.predict(X_S, coeffs_t)
                piecewise_approximation_t[decimal_number_function][m] = hX_S_t  # shifted/transformed approximating function. Localized, only valid on that square/cube
        current_node.piecewise_approximation_t = piecewise_approximation_t

    hX = np.zeros([int(division_points**x_dim), y_dim])
    X_evaluation = grid(division_points, C)
    for _ in range(3):  # run three times for online evaluation time
        current_time = time.time()
        for i in tqdm(range(X_evaluation.shape[0])):
            hX[i] = approximation(X_evaluation[i], C, round_n_digits, y_dim)  # evaluate approx. function
        print(f'Time needed for approximating function for equidistant grid per step: {(time.time()-current_time)/X_evaluation.shape[0]*10**6} micro seconds.')
    if gt_string == 'plasma_python':
        return hX, X_evaluation, relevant_fev, total_fev, C
    elif gt_string == 'CSTR_python':
        gt = cstr_python()
        gt.get_solver()
        # fX_evaluation, infeasible_points = gt.get_function_values(X_evaluation)
        fX_evaluation = loadmat('dict_values_RMPC_python.mat')['fX_evaluation'].T.flatten()
        infeasible_points = loadmat('dict_values_RMPC_python.mat')['infeasible_points'].flatten()
        max_error = 0
        for ii in range(hX.flatten().shape[0]):
            if not np.isnan(hX.flatten()[ii]):
                error = abs(hX.flatten()[ii]-fX_evaluation[ii])
                if error >= max_error:
                    max_error = error
        print(f'Maximum error is {max_error} with a priori guaranteed error {C.epsilon}.')
        return hX, X_evaluation, relevant_fev, total_fev, C, fX_evaluation, infeasible_points
    elif gt_string == 'sine_2D' or gt_string == 'waterfall_2D' or gt_string == 'ackley':
        gt = ground_truth(gt_string)
        fX_evaluation, _ = gt.get_function_values(X_evaluation)
        return hX, X_evaluation, relevant_fev, total_fev, C, fX_evaluation


def closed_loop_approx_plasma(num_iter, C, x=np.array([35, 58, 0]), const=0.5, dt=0.5):
    '''
    Args:
        num_iter (int): number of time steps for the closed-loop
        C (anytree.node): head node corresponding to global domain
        x (array): initial state
        const (float): constant for the MPC solver
        dt (float): sampling time [s]
    Returns:
        X (list): state trajectory
        u_list (list): input trajectory
    '''
    xs = [38, 43.5, 0]  # shifting of states
    us = [3, 3]  # shifting of inputs
    X = np.zeros([num_iter+1, 3])
    X[0, :] = x.flatten()
    u_list = np.zeros([num_iter, 2])
    x1_abs, x2_abs, x3_abs = x  # init
    for i in range(num_iter):
        x1_abs, x2_abs, x3_abs = X[i, :]
        # Normalize
        x1_normalized = (x1_abs-25)/(42.5-25)
        x2_normalized = (x2_abs-20)/(80-20)
        x3_normalized = (x3_abs-0)/(11-0)
        u1, u2 = approximation(np.array([x1_normalized, x2_normalized, x3_normalized]), C, round_n_digits=14, y_dim=2)
        u_list[i, :] = np.array([u1, u2]).flatten()
        # Dymamics equations
        x1_abs_new = xs[0] + 0.42*(x1_abs-xs[0]) + 0.68*(x2_abs-xs[1]) + 1.58*(u1-us[0]) - 1.02*(u2-us[1])
        x2_abs_new = xs[1] - 0.06*(x1_abs-xs[0]) + 0.26*(x2_abs-xs[1]) + 0.73*(u1-us[1]) + 0.03*(u2-us[1])
        x3_abs_new = x3_abs + (const**(43-x1_abs))*dt  # that is the problem!
        X[i+1, :] = np.array([x1_abs_new, x2_abs_new, x3_abs_new]).flatten()
    return X, u_list


if __name__ == '__main__':
    gt_string = 'ackley'
    if gt_string == 'sine_2D':
        with open('C_root_sine_2D.pickle', 'rb') as handle:
            C = pickle.load(handle)
        hX, X_evaluation, relevant_fev, total_fev, C, fX_evaluation = post_processing(C, round_n_digits=14, division_points=300, gt_string=gt_string)
        # Plot the ground truth and the prediction
        # plot_gt_2D(X_evaluation, fX_evaluation)
        # plot_prediction_2D(X_evaluation, hX.flatten())
        # Determine maximum evaluation error
        max_error = max(abs(hX.flatten()-fX_evaluation))
        print(f'The maximum error is {max_error}, the guaranteed error was {C.epsilon}.')
    elif gt_string == 'ackley':
        with open('C_root_ackley_5e-3.pickle', 'rb') as handle:
            C = pickle.load(handle)
        hX, X_evaluation, relevant_fev, total_fev, C, fX_evaluation = post_processing(C, round_n_digits=14, division_points=500, gt_string=gt_string)
        # Plot the ground truth and the prediction
        # plot_gt_2D(X_evaluation, fX_evaluation)
        # plot_prediction_2D(X_evaluation, hX.flatten())
        # Determine maximum evaluation error
        max_error = max(abs(hX.flatten()-fX_evaluation))
        print(f'The maximum error is {max_error}, the guaranteed error was {C.epsilon}.')

    
    elif gt_string == 'plasma_python':
        t = time.time()
        solver, lb_relaxed, ub_relaxed, con_lb, con_ub, n, m, N = plasma_MPC_get_solver()
        print(f'solver loaded in {time.time()-t} seconds.')
        time_steps = 30
        # Closed-loop control with MPC
        evaluation_times_MPC = []
        for i in range(3):
            current_time = time.time()
            X_MPC, u_list_MPC = closed_loop_MPC(time_steps, solver, lb_relaxed, ub_relaxed, con_lb, con_ub, n, m, N)
            new_current_time = time.time()
            time_MPC = (new_current_time-current_time)/len(u_list_MPC)*10**3
            print(f'Time needed per MPC closed-loop step is: {time_MPC} milli seconds.')
            evaluation_times_MPC.append(time_MPC)
        print(f'Average time needed per MPC closed-loop step is: {np.mean(evaluation_times_MPC)} with an std of {np.std(evaluation_times_MPC)}.')

        with open('C_root_plasma.pickle', 'rb') as handle:
            C = pickle.load(handle)
        _, _, _, _, C = post_processing(C, round_n_digits=14, division_points=50, gt_string=gt_string)
        evaluation_times_approx = []
        for i in range(3):
            current_time = time.time()
            X_approx, u_list_approx = closed_loop_approx_plasma(time_steps, C)
            new_current_time = time.time()
            time_approx_MPC = (new_current_time-current_time)/len(u_list_approx)*10**6
            print(f'Time needed per approximate MPC closed-loop step is: {time_approx_MPC} micro seconds.')
            evaluation_times_approx.append(time_approx_MPC)
        print(f'Average time needed per approximate MPC closed-loop step is: {np.mean(evaluation_times_approx)} with an std of {np.std(evaluation_times_approx)}.')
        # Check constraint violation
        if np.all(X_approx >= np.array([25, 20, 0])) and np.all(X_approx <= np.array([42.5, 80, 11])):
            print('All state constraints are satisfied!')
        if np.all(u_list_approx >= np.array([1.5, 1])) and np.all(u_list_approx <= np.array([8, 6])):
            print('All input constraints are satisfied!')

        # Compare closed-loop trajectory of AMPC using the MPC
        u_list_MPC_for_approx_traj, _ = plasma_MPC_get_samples(solver, X_approx, lb_relaxed, ub_relaxed, con_lb, con_ub, n, m, N)
        u_list_MPC_for_approx_traj = u_list_MPC_for_approx_traj[:-1]  # last step not required
        diff_1 = abs(u_list_MPC_for_approx_traj[0, :] - u_list_approx[0, :])
        diff_2 = abs(u_list_MPC_for_approx_traj[1, :] - u_list_approx[1, :])
        RMSE_1 = sum(diff_1)/len(diff_1)
        RMSE_2 = sum(diff_2)/len(diff_2)
        print(f'The RMSEs are {RMSE_1} and {RMSE_2}, respectively. The maximum error (infinity norm) is {max(max(diff_1), max(diff_2))*10**3}e-3.')
        generate_plots_plasma(X_MPC, X_approx, u_list_MPC, u_list_approx)
    elif gt_string == 'CSTR_python':
        with open('C_root_CSTR.pickle', 'rb') as handle:
            C = pickle.load(handle)
        hX, X_evaluation, relevant_fev, total_fev, C, fX_evaluation, infeasible_points = post_processing(C, 14, 300, gt_string)
        plot_gt_2D_CSTR(X_evaluation, fX_evaluation)
        plot_subdomains_2D_CSTR(C, infeasible_points, X_evaluation)
