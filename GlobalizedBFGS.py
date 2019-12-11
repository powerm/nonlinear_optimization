import numpy as np
import StepSizeRules
from numpy.linalg import norm

gamma = 0.001
eta = 0.9


def globalized_BFGS(x, function_gradient, stopping_tolerance, maximum_iterations):
    iterations = 0
    n = x.shape[0]
    B = np.eye(n)
    xs = [x]
    while norm(function_gradient(x, gradient_only=True)) > stopping_tolerance:
        function_value, gradient = function_gradient(x)
        search_direction = - np.matmul(B, gradient)
        directional_derivative = np.matmul(np.transpose(gradient), search_direction)
        step_size = StepSizeRules.powell_wolfe(x, search_direction, directional_derivative, function_gradient, function_value)
        d = step_size * search_direction  # = x_next - x
        x_next = x + d
        y = function_gradient(x_next, gradient_only=True) - gradient
        # update B using BFGS update
        B_y = np.matmul(B, y)
        B = B + (np.matmul(d - B_y, np.transpose(d)) + np.matmul(d, np.transpose(d - B_y))) / (np.matmul(np.transpose(y), d)) \
            - (np.matmul(np.transpose(d - B_y), y) / (np.square(np.matmul(np.transpose(y), d)))) * np.matmul(d, np.transpose(d))
        # update iterates
        x = x_next
        xs.append(x)
        # update iteration count
        iterations += 1
        if iterations >= maximum_iterations:
            break
    return x, xs

