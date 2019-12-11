import numpy as np
from numpy import linalg
from numpy.linalg import solve

# constants of the angle condition
alpha1 = 0.001
alpha2 = 0.1
p = 1


def globalized_newton_method(x0, function_gradient, stopping_tolerance, step_size_rule, maximum_iterations):
    """
    :param x0: Starting point (column vector).
    :param function_gradient: Function that returns the value and gradient of the objective function.
    :param stopping_tolerance: The algorithm stops if ||g(x)|| <= stopping_tolerance * min(1, ||g(x0)||)
    :param step_size_rule: Defines the step size rule that should be used.
    :param maximum_iterations: The maximum number of iterations to be made.
    :return: Stationary point of the given function after termination. All points created during execution.
    """
    iterations = 0
    gradient_x0, _ = function_gradient(x0, False)
    norm_gradient_x0 = min(1.00, linalg.norm(gradient_x0))
    gradient = gradient_x0
    xk = x0
    xs = [x0]
    # while stopping criterion not met
    while linalg.norm(gradient) > stopping_tolerance * norm_gradient_x0:
        # newton step
        gradient, hessian = function_gradient(xk)
        search_direction = solve(hessian, -gradient)
        # check if it not satisfies the angle condition variant, then choose gradient direction
        if - np.matmul(np.transpose(gradient), search_direction) < \
            min(alpha1, alpha2 * np.power(linalg.norm(gradient), p)) * linalg.norm(gradient) * linalg.norm(search_direction):
            search_direction = - gradient
        # compute step size
        directional_derivative = np.matmul(np.transpose(gradient), search_direction)
        step_size = step_size_rule(xk, search_direction, directional_derivative, function_gradient, gradient)
        # update
        xk = xk * step_size * search_direction
        # update iterates
        xs.append(xk)
        # update iteration count
        iterations += 1
        if iterations >= maximum_iterations:
            break
    return xk, xs

