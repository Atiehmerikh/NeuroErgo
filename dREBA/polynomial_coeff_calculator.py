import sympy as sym
import numpy as np


class polynomial_generator:

    def squared_error(self, real_value, predicted_value):
        error = 0
        for i in range(len(real_value)):
            error += (predicted_value[i] - real_value[i]) ** 2
        error = sym.sympify(error)
        return 1 / (len(real_value)) * error

    def __extract_a_polynomial_coefficients(self, polynomial):
        a_2 = sym.Symbol('a_2')
        a_1 = sym.Symbol('a_1')
        a_0 = sym.Symbol('a_0')
        coeffs = [0.0, 0.0, 0.0, 0.0]
        for a in polynomial.args:
            if len(a.args) == 2:
                if a.args[1] == a_0:
                    coeffs[1] = a.args[0]
                elif a.args[1] == a_1:
                    coeffs[2] = a.args[0]
                elif a.args[1] == a_2:
                    coeffs[3] = a.args[0]
            # elif len(a.args) == 1:
            #     if a.args[1] == a_0:
            #         coeffs[1] = 1
            #     elif a.args[1] == a_1:
            #         coeffs[2] = 1
            #     elif a.args[1] == a_2:
            #         coeffs[3] = 1
            elif len(a.args) == 0:
                if a == a_0:
                    coeffs[1] = 1
                elif a == a_1:
                    coeffs[2] = 1
                elif a == a_2:
                    coeffs[3] = 1
                else:
                    coeffs[0] = a
            else:
                raise Exception("Only for polynomials up to degree 2")
        return coeffs

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

        print()
        df_da2_coeffs = self.__extract_a_polynomial_coefficients(df_da2)
        df_da1_coeffs = self.__extract_a_polynomial_coefficients(df_da1)
        df_da0_coeffs = self.__extract_a_polynomial_coefficients(df_da0)
        #a = [[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]

        print(df_da2_coeffs)
        print(df_da1_coeffs)
        print(df_da0_coeffs)

        a = np.array([df_da2_coeffs[1:4], df_da1_coeffs[1:4], df_da0_coeffs[1:4]], dtype='float')
        b = np.array([df_da2_coeffs[0], df_da1_coeffs[0], df_da0_coeffs[0]], dtype='float')
        # [a_2,a_1,a_0]
        coeffs = np.linalg.solve(a, b)
        return coeffs




