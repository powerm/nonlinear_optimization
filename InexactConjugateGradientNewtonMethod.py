import StepSizeRules
import numpy as np
from numpy.linalg import norm

# constants
alpha = 0.001
vu = 0.01


def inexact_conjugate_gradient_newton_method(x_k, function_gradient_hessian):
    x = x_k
    y = 0
    g = function_gradient_hessian(x, gradient_only=True)
    d = g
    while True:
        _, gradient, hessian = function_gradient_hessian(x)
        gradient_norm = norm(gradient)
        # relative residuum is small enough
        if norm(g) <= min(vu, gradient_norm) * gradient_norm:
            return y
        # direction of non-positive curvature
        if np.matmul(np.matmul(np.transpose(d), hessian), d) <= 0:
            return y - np.sign(np.matmul(np.transpose(gradient), d)) * gradient_norm * (d / norm(d))
        a = (np.matmul(np.transpose(g), g)) / np.matmul(np.matmul(np.transpose(d), hessian), d)
        y_next = y - a * d
        g_next = g - a * np.matmul(hessian, d)
        # direction of descent becomes insufficient
        if - np.matmul(np.transpose(gradient), y_next) < min(alpha, gradient_norm) * gradient_norm * norm(y_next):
            return y
        beta = np.matmul(np.transpose(g_next), g_next) / np.matmul(np.transpose(g), g)
        d_next = g_next + beta * d
        # update
        y = y_next
        g = g_next
        d = d_next


def general_descent_inexact_newton_conjugate_gradient(x0, function_gradient_hessian, stopping_tolerance, maximum_iterations):
    iterations = 0
    x = x0
    # keep track of iterates
    xs = [x0]
    # while not close to a stationary point
    while norm(function_gradient_hessian(x, gradient_only=True)) > stopping_tolerance:
        # compute a direction of descent
        search_direction = inexact_conjugate_gradient_newton_method(x, function_gradient_hessian)
        # compute a step size
        objective_function_value, gradient, _ = function_gradient_hessian(x)
        step_size = StepSizeRules.armijo(x, search_direction, np.matmul(np.transpose(gradient), search_direction),
                                         function_gradient_hessian, objective_function_value)
        # update
        x = x + step_size * search_direction
        xs.append(x)
        # update iteration count
        iterations += 1
        if iterations >= maximum_iterations:
            break
    return x, xs