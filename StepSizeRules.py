import numpy as np


def armijo(xk, search_direction, directional_derivative, function_gradient, objective_function_value, gamma=0.001, beta=0.5):
    """
    :param xk: The current point.
    :param search_direction: The current search direction.
    :param directional_derivative: The current directional derivative in the search direction.
    :param function_gradient: Function that returns the value of the objective function.
    :param objective_function_value: The current objective function value.
    :param gamma: Constant 0 < gamma < 0.5 of Armijo condition.
    :param beta: Constant of Armijo condition.
    :return: Step size satisfying the Armijo condition.
    """
    step_size = 1
    while (function_gradient(xk + step_size * search_direction, True) > \
            objective_function_value + gamma * step_size * directional_derivative).all():
        step_size *= beta
    return step_size


def powell_wolfe(xk, search_direction, directional_derivative, function_gradient, objective_function_value, gamma=0.001, eta=0.9):
    """
    :param xk: The current point.
    :param search_direction: The current search direction.
    :param directional_derivative: The current directional derivative in the search direction.
    :param function_gradient: Function that returns the value of the objective function.
    :param objective_function_value: The current objective function value.
    :param gamma: Constant 0 < gamma < 0.5 of Powell-Wolfe condition.
    :param eta: Constant gamma < eta < 1 of the Powell-Wolfe condition.
    :return: Step size satisfying the Powell-Wolfe condition.
    """
    if function_gradient(xk + search_direction, True) <= objective_function_value + gamma * directional_derivative:
        if np.matmul(np.transpose(function_gradient(xk + search_direction, False, True)), search_direction)\
                >= eta * directional_derivative:
            return 1
        else:
            t_u = 2
            while function_gradient(xk + t_u * search_direction, True) <= \
                    objective_function_value + gamma * directional_derivative * t_u:
                t_u *= 2
            t_l = 0.5 * t_u
    else:
        t_l = 0.5
        while function_gradient(xk + t_l * search_direction, True) > \
                objective_function_value + gamma * directional_derivative * t_l:
            t_l *= 0.5
        t_u = 2 * t_l
    while np.matmul(np.transpose(function_gradient(xk + t_l * search_direction, False, True)), search_direction) < \
            eta * directional_derivative:
        t_c = 0.5 * (t_l + t_u)
        if function_gradient(xk + t_c * search_direction, True) <= \
                objective_function_value + gamma * directional_derivative * t_c:
            t_l = t_c
        else:
            t_u = t_c
    return t_l

