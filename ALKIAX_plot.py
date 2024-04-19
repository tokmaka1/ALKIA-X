import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cs
from matplotlib import cm
from anytree import PreOrderIter
from scipy.spatial import ConvexHull
import tikzplotlib


def plot_gt_2D(X, fX):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(X[:, 0], X[:, 1], fX)
    surf = ax.plot_trisurf(X[:, 0], X[:, 1], fX, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_zlabel('$f(x)$')
    plt.title('Ground Truth')
    plt.show()


def plot_error(X, hX, fX, epsilon):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    error = np.abs(hX-fX)

    # Envelope the function values with plus and minus epsilon
    error_bound = [epsilon]*X.shape[0]

    # Plot the original surface
    ax.plot_trisurf(X[:, 0], X[:, 1], error, alpha=1, cmap=cm.jet, linewidth=0)

    # Plot the enveloped surfaces with very transparent color
    ax.plot_trisurf(X[:, 0], X[:, 1], error_bound, color='black', alpha=0.1, linewidth=0)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_zlabel('Error')
    plt.title('Ackley function')
    plt.show()



def plot_gt_2D_CSTR(X_evaluation, fX_evaluation):
    X_evaluation_reshaped = X_evaluation.copy()
    x_begin = [-0.2, -0.2]
    x_end = [0.2, 0.2]
    for i in range(X_evaluation_reshaped.shape[0]):
        for k in range(X_evaluation_reshaped.shape[1]):
            X_evaluation_reshaped[i, k] = X_evaluation_reshaped[i, k]*x_end[k] + (1-X_evaluation_reshaped[i, k])*x_begin[k]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    filtered_indices = np.logical_and(fX_evaluation >= 0, fX_evaluation <= 2.1)  # only plot that part
    ax.plot_trisurf(X_evaluation_reshaped[:, 0][filtered_indices], X_evaluation_reshaped[:, 1][filtered_indices], fX_evaluation[filtered_indices])
    surf = ax.plot_trisurf(X_evaluation_reshaped[:, 0][filtered_indices], X_evaluation_reshaped[:, 1][filtered_indices], fX_evaluation[filtered_indices], cmap=cm.jet, linewidth=0)
    xlim = (-0.2, 0.2)
    xticks = [-0.2, -0.1, 0, 0.1, 0.2]
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.view_init(elev=25, azim=-125)
    plt.show()


def plot_prediction_2D(X, hX):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(X[:, 0], X[:, 1], hX)
    surf = ax.plot_trisurf(X[:, 0], X[:, 1], hX, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.set_zlabel('$h(x)$')
    plt.title('Predicted function')
    plt.show()


# Plot sub-domain partitioning and feasible domains
def plot_subdomains_2D_CSTR(C, infeasible_points, X_evaluation):
    plt.figure()
    x_begin = [-0.2, -0.2]
    x_end = [0.2, 0.2]
    X_evaluation_reshaped = X_evaluation.copy()
    x_begin = [-0.2, -0.2]
    x_end = [0.2, 0.2]
    for i in range(X_evaluation_reshaped.shape[0]):
        for k in range(X_evaluation_reshaped.shape[1]):
            X_evaluation_reshaped[i, k] = X_evaluation_reshaped[i, k]*x_end[k] + (1-X_evaluation_reshaped[i, k])*x_begin[k]
    for current_node in PreOrderIter(C):
        shifted_D = current_node.D
        D = shifted_D.copy()
        D[0, 0] = shifted_D[0, 0]*x_end[0] + (1-shifted_D[0, 0])*x_begin[0]
        D[0, 1] = shifted_D[0, 1]*x_end[1] + (1-shifted_D[0, 1])*x_begin[1]
        D[1, 0] = shifted_D[1, 0]*x_end[0] + (1-shifted_D[1, 0])*x_begin[0]
        D[1, 1] = shifted_D[1, 1]*x_end[1] + (1-shifted_D[1, 1])*x_begin[1]
        plt.plot([D[0][0], D[0][1], D[0][1], D[0][0], D[0][0]], [D[1][0], D[1][0], D[1][1], D[1][1], D[1][0]], '-k', linewidth=0.4)
        if current_node.is_leaf and not current_node.infeasible_domain:
            plt.fill([D[0][0], D[0][1], D[0][1], D[0][0], D[0][0]], [D[1][0], D[1][0], D[1][1], D[1][1], D[1][0]], color='magenta', alpha=0.2)
    feasible_boolean = infeasible_points == 0
    X_feasible = X_evaluation_reshaped[feasible_boolean]  # plot in [-0.2, 0.2] \times [-0.2, 0.2] grid
    points = list(zip(X_feasible[:, 0], X_feasible[:, 1]))
    hull = ConvexHull(points)
    vertices = hull.vertices
    plt.fill(X_feasible[:, 0][vertices], X_feasible[:, 1][vertices], color='lightgray', alpha=0.8)
    plt.show()


def generate_plots_plasma(X_MPC, X_approx, u_list_MPC, u_list_approx):
    plt.figure()
    plt.plot(range(len(X_MPC)), X_MPC[:, 0], color='blue')
    plt.plot(range(len(X_MPC)), X_approx[:, 0], color='magenta')
    plt.plot(range(len(X_MPC)), 42.5*np.ones(len(X_MPC)), color='black')
    plt.xlabel('Time steps')
    plt.ylabel('$x_1$')
    plt.show()

    plt.figure()
    plt.plot(range(len(X_MPC)), X_MPC[:, 1], color='blue')
    plt.plot(range(len(X_MPC)), X_approx[:, 1], color='magenta')
    plt.xlabel('Time steps')
    plt.ylabel('$x_2')
    plt.show()

    plt.figure()
    plt.plot(range(len(X_MPC)), X_MPC[:, 2], color='blue')
    plt.plot(range(len(X_MPC)), X_approx[:, 2], color='magenta')
    plt.plot(range(len(X_MPC)), 10*np.ones(len(X_MPC)), color='green')
    plt.xlabel('Time steps')
    plt.ylabel('$x_3$')
    plt.show()

    plt.figure()
    plt.step(range(len(u_list_MPC)), u_list_MPC[:, 0], color='blue')
    plt.step(range(len(u_list_MPC)), u_list_approx[:, 0], color='magenta')
    plt.xlabel('Time steps')
    plt.ylabel('$u_1$')
    plt.show()

    plt.figure()
    plt.step(range(len(u_list_MPC)), u_list_MPC[:, 1], color='blue')
    plt.step(range(len(u_list_MPC)), u_list_approx[:, 1], color='magenta')
    plt.xlabel('Time steps')
    plt.ylabel('$u_2$')
    plt.show()
