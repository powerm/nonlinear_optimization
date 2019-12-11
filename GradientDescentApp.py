import GradientDescent as GD
import StepSizeRules
import numpy as np
import Examples
import Plotting


x_result, iterates = GD.gradient_steepest_descent(np.array((9, 3)), Examples.quadratic_function, 0.001,
                                                  StepSizeRules.armijo, 5000)
print("result for quadratic function", x_result)
Plotting.plot_iterates("gradient_descent_quadratic_armijo", Examples.quadratic_function, iterates)

x_result, iterates = GD.gradient_steepest_descent(np.array((0, -3)), Examples.rosenbrock, 0.001,
                                                  StepSizeRules.powell_wolfe, 5000)
print("result for rosenbrock function", x_result)
Plotting.plot_iterates("gradient_descent_rosenbrock_powell", Examples.rosenbrock, iterates)
