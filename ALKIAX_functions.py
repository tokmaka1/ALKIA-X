import numpy as np
from anytree import Node


def create_children(parent_node, round_n_digits):
    '''
    Args:
        parent_node (anytee.node): the node whose children we create
        round_n_digits (int): number of digits for rounding
    Returns:
        children_node_list (list): list of 2**x_dim children nodes of parent_node
    '''
    parent_node_width = parent_node.D[0][1]-parent_node.D[0][0]
    width = parent_node_width/2  # width of child
    x_dim = np.shape(parent_node.D)[0]
    children_node_list = [0] * (2**x_dim)  # number of children
    number_of_nodes_per_axis = 2  # divide into cubes
    for i in range(len(children_node_list)):
        D, name = node_domain_name(width, i, number_of_nodes_per_axis, parent_node.D)
        current_node = Node(f'{parent_node.name}_{name}')
        current_node.D = D
        children_node_list[i] = current_node
    return children_node_list


def cube(M, current_node):
    # TODO: understand the binary logic a bit more and put into readme.
    '''Create cube with 2**x_dim samples. Size of the cube is determined by M and the domain of current node.
    Args:
        M (int): number of equidistant samples per axis on current_node. This determines the edge length of the cube.
        current_node (anytree.node): node of interest
    Returns:
        X_S (array): one cube with 2**x_dim samples within the domain of the node of interest.
    '''
    if M < 2:
        raise Exception('Start with at least 2 samples per axis!')
    x_dim = current_node.D.shape[0]  # dimension of the input
    delta_x = (current_node.D[0][1]-current_node.D[0][0])/(M-1)  # grid size
    X_S = np.zeros((2**x_dim, x_dim))  # wlog: start at [0,...,0]. We are just interested in one of the uniform local cubes.
    for number_of_point in range(X_S.shape[0]):  # 2**x_dim
        # "binary" logic of numbering the samples within the cube
        x = X_S[number_of_point, :]
        rest_division = number_of_point  # starts with 0 of course
        exp = x_dim
        entry_of_point = 0
        while rest_division != 0:
            exp = exp - 1
            int_division = int(rest_division/2**exp)
            x[entry_of_point] = delta_x*int_division
            rest_division = rest_division % (2**exp)
            entry_of_point += 1
        X_S[number_of_point] = x
    return X_S, delta_x


def grid(M, current_node):
    '''Create grid with M equidistant samples per axis on the domain of the node of interest
    Args:
        M (int): number of equidistant samples per axis on current_node. This determines the number of samples for the grid.
        current_node (anytree.node): node of interest
    Returns:
        X_D (array): M equidistant samples per axis within the domain of the node of interest.
    '''
    x_dim = current_node.D.shape[0]  # dimension of the input
    X_D = np.zeros((M**x_dim, x_dim)) + current_node.D[:, 0]  # initialization
    for number_of_point in range(X_D.shape[0]):
        x = X_D[number_of_point, :]
        rest_division = number_of_point
        exp = x_dim
        entry_of_point = 0
        while rest_division != 0:
            # "binary" logic of numbering
            exp = exp - 1
            int_division = int(rest_division/M**exp)
            x[entry_of_point] = current_node.D[entry_of_point][0] + (current_node.D[entry_of_point][1] - current_node.D[entry_of_point][0])/(M-1)*int_division
            rest_division = rest_division % (M**exp)
            entry_of_point += 1
        X_D[number_of_point] = x
    return X_D


def binary_number(decimal_number, x_dim, base):
    '''Binary logic: convert decimal number into corresponding "binary number"
    Args:
        decimal_number (int): needed to extract entry from e.g. the array X_D
        x_dim (int): dimension of the input
        base (int): base. In binary numbers, the base is always 2. For sorting reasons, our base can differ from 2.
    Returns:
        binary_number (list): corresponding binary number. This can also be an n-ary number. For sorting/locating reasons.
    '''
    rest_division = decimal_number
    binary_number = [0]*x_dim
    exp = x_dim
    entry_of_point = 0
    while rest_division != 0:
        # "binary" logic of numbering
        exp = exp - 1
        int_division = int(rest_division/base**exp)
        rest_division = rest_division % (base**exp)
        binary_number[entry_of_point] = int_division
        entry_of_point += 1
    return binary_number


def decimal_number(binary_number, base):
    '''Recriprocal function to def binary_number()
    Args:
        binary_number (list): corresponding binary number
        base (int): Base of the binary/n-ary counting.
    Returns:
        decimal_number (int): corresponding decimal number
    '''
    decimal_number = 0
    for i in range(len(binary_number)):
        exp = len(binary_number) - i - 1
        decimal_number += binary_number[i]*base**exp
    return decimal_number


def update_infeasible_points_dict(dict_total_f, infeasible_points, X_D, round_n_digits):
    for i_infeasible in range(len(infeasible_points)):  # range(infeasible_points.shape[0]):
        dict_total_f[tuple(np.around(X_D[i_infeasible], round_n_digits))].append(infeasible_points[i_infeasible])   # just append to list
    return dict_total_f


def current_node_is_infeasible(X_D, infeasible_points_dict, round_n_digits):
    '''Check whether the node of interest only contains infeasible samples
    Args:
        X_D (array): array of samples to investigate infeasibility
        infeasible_points_dict (dict): dictionary with sample (key) and boolean (value) for infeasibility
        round_n_digits (int): number of digits for rounding
    Returns:
        (boolean): Node is infeasible (True) or feasible (False)
    '''
    # if M < 9:
    #     warnings.warn('Hard-coding!')
    #     return False
    # else:
    for i_XD in range(X_D.shape[0]):
        if infeasible_points_dict[tuple(np.around(X_D[i_XD], round_n_digits))][1] == 0:
            return False
    # TODO: a bit unnecessary do go through all of them right??
    return True


def head_nodes(C_root, number_of_head_nodes, round_n_digits):
    '''Given root node, return first-level children, which are called head nodes
    Args:
        C_root (anytee.node): root node
        number_of_head_nodes (int): number of first-level children of the root node
        round_n_digits (int): number of digits for rounding
    Returns:
        dict_head_nodes (dict): dictionary with head node names (keys) and the head nodes (values)
    '''
    x_dim = C_root.D.shape[0]
    if np.around((number_of_head_nodes)**(1/x_dim), round_n_digits) != float(int(np.around(number_of_head_nodes**(1/x_dim), round_n_digits))):
        raise Exception('Global domain has to be divided into cubes!')
    number_of_head_nodes_per_axis = int(np.around(number_of_head_nodes**(1/x_dim), round_n_digits))
    C_root_width = C_root.D[0][1]-C_root.D[0][0]
    width = C_root_width/number_of_head_nodes_per_axis  # the children nodes always correspond to uniform cubes
    dict_head_nodes = {}
    for number_of_node in range(number_of_head_nodes):
        local_domain, node_name = node_domain_name(width, number_of_node, number_of_head_nodes_per_axis, C_root.D)
        current_node = Node(f'C_{node_name}')  # create the node
        current_node.D = local_domain
        current_node.is_head = True
        current_node.dict_total_f = {}  # initialization
        dict_head_nodes[current_node.name] = current_node  # save in dictionary
    return dict_head_nodes


def root_node(D):
    '''Given the global domain, create the root node
    Args:
        D (array): global domain
    Returns:
        C_root (anytree.node): root node
    '''
    C_root = Node('C')
    C_root.D = D
    C_root.dict_total_f = {}  # initialization
    return C_root


def node_domain_name(width, number_of_node, number_children_per_axis, parent_domain):
    '''
    Args:
        width (float): width of the node whose domain and name we determine
        number_of_node (int): numbering of the node in decimal logic
        number_children_per_axis (int): number of child nodes per axis
        parent_domain (array): domain of the parent node
    Returns:
        local_domain (array): domain of the node of interest
        node_name_decimal (int): number of that node, for sorting and identifying
    '''
    rest_division = number_of_node
    node_name_index = 0
    x_dim = parent_domain.shape[0]
    node_name = [0]*x_dim  # for n-ary
    exp = x_dim
    while rest_division != 0:
        # "binary" logic of filling the node_name list
        exp = exp - 1
        int_division = int(rest_division/number_children_per_axis**exp)
        node_name[node_name_index] = int_division
        rest_division = rest_division % (number_children_per_axis**exp)
        node_name_index += 1
    local_domain = np.zeros([x_dim, 2])  # initialization
    for k in range(x_dim):
        local_domain[k][0] = np.array(parent_domain[k][0])+node_name[k]*width  # beginning
        local_domain[k][1] = np.array(parent_domain[k][0])+node_name[k]*width+width  # end
    node_name_decimal = decimal_number(node_name, base=number_children_per_axis)
    '''
    base=number_children_per_axis:
        - number_of_children_per_axis=2 for all except for root
        - for root: we can have more or less child/head nodes
    '''
    return local_domain, node_name_decimal


def compute_M_list(M_min, cond_max, kernel, C_root):
    '''List of equidistant samples per axis such that cond<cond_max
    Args:
        M_min (int): minimum number of samples per axis
        cond_max (float): maximum condition number
        kernel (ALKIAX_kernel.matern_kernel object): chosen kernel with starting length scale C_ell
        C_root (anytree.node): root node corresponding to global domain
    Returns:
        M_list (list): list containing integers of number of samples per axis
    '''
    cond = 0
    M = int((M_min + 1)/2)
    M_list = []
    while True:
        M = 2*M-1
        X_D = grid(M, C_root)
        K, _ = kernel.gaussian_process.covariance_matrix(X_D)
        cond = np.linalg.cond(K)
        if cond < cond_max:
            M_list.append(M)
        else:
            break
    return M_list


def relevant_dict_total_f(child_node, parent_node, round_n_digits):
    # TODO: is this efficient? If yes, add a brief description
    dict_total_f = {}
    for entry in parent_node.dict_total_f:
        is_relevant_flag = 1
        for i in range(len(entry)):
            if not (np.around((entry[i] - child_node.D[i][0]), round_n_digits) >= 0 and  np.around((entry[i] - child_node.D[i][1]), round_n_digits) <= 0):
                is_relevant_flag = 0
                break
        if is_relevant_flag == 1:
            dict_total_f[entry] = parent_node.dict_total_f[entry]
    return dict_total_f


def approximation(x, C, round_n_digits, y_dim):
    '''
    Args:
        x (array): input to be evaluated by the approximating function
        C (anytree.node): root node
        round_n_digits (int): number of digits for rounding
        y_dim (int): output dimension
    Returns:
        h_x (float): approximating functioon evaluated at x. If infeasible, NaN is returned.
    '''
    nan = float('NaN')
    x_dim = C.D.shape[0]
    number_of_head_nodes = len(C.children)
    number_of_head_nodes_per_axis = int(np.around(number_of_head_nodes**(1/x_dim), round_n_digits))  # child nodes always correspond to uniform cubes
    binary_number_head = [0]*x_dim
    decimal_number_head = 0
    C_width = C.D[0][1]-C.D[0][0]  # width corresponding to the root node, usually equal to 1.
    width_head = C_width/(number_of_head_nodes_per_axis)  # width of head nodes
    for i in range(x_dim):  # for each dimension
        # "binary" sorting/finding logic
        binary_number_head[i] = int(np.around((x[i]-C.D[i][0])/width_head, round_n_digits)) \
            if int(np.around((x[i] - C.D[i][0])/width_head, round_n_digits)) < number_of_head_nodes_per_axis \
            else number_of_head_nodes_per_axis-1
        exp = x_dim-i-1
        decimal_number_head += number_of_head_nodes_per_axis**exp*binary_number_head[i]
    current_node = C.children[decimal_number_head]
    # current_node is the head node that contains x in one of its leaf nodes
    not_finished = True
    while not_finished:  # finding leaf node that contains x
        D = current_node.D
        if current_node.children == ():  # leaf node
            not_finished = False  # we are done and have the relevant leaf
            if current_node.infeasible_domain == 1:
                return nan  # no value for infeasible domain
            else:
                break  # relevant leaf node found
        else:
            node_number_decimal = 0
            for i in range(x_dim):
                node_number_step = x[i] >= D[i][0] + (D[i][1] - D[i][0])/2
                node_number_decimal += node_number_step*(2**(x_dim- i-1))
            current_node = current_node.children[node_number_decimal]  # one level deeper
    # current_node is now the leaf node containing x
    piecewise_approximation_t = current_node.piecewise_approximation_t  # shifted localized approximating functions
    D = current_node.D
    M = int(np.round((current_node.X_D.shape[0])**(1/x_dim)))
    delta_x = (D[0][1] - D[0][0])/(M-1)  # grid size
    grid_number_decimal = 0
    for i in range(x_dim):
        grid_number_step = int((x[i] - D[i][0])/delta_x)  # how many times does it fit in the square
        if grid_number_step == M-1:  # corner case at the of boundary
            grid_number_step = M-2  # we "push back to previous cube"
        grid_number_decimal += grid_number_step*((M-1)**(x_dim-i-1))
    h_S = piecewise_approximation_t[grid_number_decimal]  # shifted approx. function of local cube that contains x
    h_x = np.zeros(y_dim)
    for m in range(y_dim):  # each output dimension
        h_x[m] = h_S[m](x) + current_node.mean[m] if len(range(y_dim)) > 1 else h_S[m](x) + current_node.mean  # re-shift
    return h_x  # if feasible return h_x, if not return nan is returned
