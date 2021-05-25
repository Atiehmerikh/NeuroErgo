import numpy as np

import polynomial_coeff_calculator as pCoeff
import Dreba_coeff_calculator as DREBA
import sympy as sym


def dReba(M,N,A):
    # M is a n to m matrix,
    # N is a n to 1 matrix
    # A is a m to 3 matrix
    # n is number of postures
    # m is number of body degrees
    total_error = 0
    n = len(N)
    for i in range(n):
        # for each posture
        m_dREBA = DREBA.Dreba_error_each_posture()
        each_posture_error = m_dREBA.dReba_error(A, M[i, :], N[i])
        total_error += each_posture_error

    total_error = sym.sympify(sym.expand((1/n)*total_error))
    w = sym.symbols('w_0:13')
    df_dw=[]
    a = []
    b =[]
    row =[]
    for i in range(len(w)):
        df_dw.append(sym.diff(total_error, w[i]))
    for i in range(len(df_dw)):
        for j in range(1,len(df_dw)+1):
            elem = df_dw[i].args[j].args[0]
            row.append(elem)
        a.append(row)
        b.append(df_dw[i].args[0])

    dREBA_coeffs = np.linalg.solve(a, b)
    print(dREBA_coeffs)

if __name__ == '__main__':
    # polynomial coefficients:(training point,partial reba score)
    m_pcoeff = pCoeff.polynomial_generator()
    Neck_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([0,10,20,80],[0,2,1,2])





