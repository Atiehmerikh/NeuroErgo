import sympy as sym
import numpy as np


class polynomial_generator:

    def squared_error(self, real_value, predicted_value):
        error = 0
        for i in range(len(real_value)):
            error += (predicted_value[i] - real_value[i]) ** 2
        error = sym.sympify(error)
        return 1 / (len(real_value)) * error

    def polynomial_coefficients_calculator(self,training_points,real_partial_REBA_values):
        a_2 = sym.Symbol('a_2')
        a_1 = sym.Symbol('a_1')
        a_0 = sym.Symbol('a_0')

        polynomial =[]
        for i in range(0,len(training_points)):
            polynomial.append(a_2*training_points[i]**2+a_1*training_points[i]+a_0)

        sq_error = sym.sympify(sym.expand(self.squared_error(real_partial_REBA_values,polynomial)))
        df_da2 = sym.diff(sq_error,a_2)
        df_da1 = sym.diff(sq_error,a_1)
        df_da0 = sym.diff(sq_error,a_0)

        a = np.array([[df_da2.args[1].args[0], df_da2.args[2].args[0], df_da2.args[3].args[0]],
                      [df_da1.args[1].args[0], df_da1.args[2].args[0], df_da1.args[3].args[0]],
                      [df_da0.args[1].args[0], df_da0.args[2].args[0], df_da0.args[3].args[0]]
                      ],dtype='float')
        b = np.array([df_da2.args[0], df_da1.args[0], df_da0.args[0]],dtype='float')
        # [a_2,a_1,a_0]
        coeffs = np.linalg.solve(a, b)
        return coeffs




