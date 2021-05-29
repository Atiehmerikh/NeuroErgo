import numpy as np

import polynomial_coeff_calculator as pCoeff
import Dreba_coeff_calculator as DREBA
import sympy as sym


def dReba_coeff_generator(M,N,A):
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
        each_posture_error = m_dREBA.dReba_symbolic_error(A, M[i, :], N[i])
        total_error += each_posture_error

    total_error = sym.sympify(sym.expand((1/n)*total_error))
    w = sym.symbols('w_0:21')
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

def dREBA_polynomial_matrix_generator():
    A = np.empty((0, 2), int)

    # polynomial coefficients:(training point,partial reba score)
    m_pcoeff = pCoeff.polynomial_generator()
    Neck_flex_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-60,0,20], [2,1,2])
    Neck_side_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-54,0, 54], [1,0,1])
    Neck_twist_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-60,0, 60], [1,0,1])

    trunk_flex_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-30,0,20,60], [3,1,2,4])
    trunk_side_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-40,0, 40],[1,0,1] )
    trunk_twist_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator( [-35,0, 35], [1,0,1])

    leg_bending_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([0,30,60], [1,1,2])

    right_upper_arm_front_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-20, 0, 20, 45,90], [2,1,2,3,4])
    right_upper_arm_side_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-2,0],[1,0,1])
    right_shoulder_rise_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([0, 30],[1,0,1])

    left_upper_arm_front_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-20, 0, 20, 45],  [2,1,2,3,4])
    left_upper_arm_side_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-2,0],[1,0,1])
    left_shoulder_rise_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([0, 30],[1,0,1])

    right_lower_arm_front_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator(
        [0, 60, 100], [2,1,2])

    left_lower_arm_front_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator(
        [0, 60, 100], [2,1,2])

    right_wrist_flex_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-53,-15,15], [2,1,1])
    right_wrist_side_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-40,0, 30], [1,0,1])
    right_wrist_twist_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator( [-90,0, 90], [1,0,1])

    left_wrist_flex_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-53,-15,15], [2,1,1])
    left_wrist_side_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-40,0, 30], [1,0,1])
    left_wrist_twist_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator( [-90,0, 90] , [1,0,1])

    np.append(A, np.array([Neck_flex_polynomial_coefficients]), axis=0)
    np.append(A, np.array([Neck_side_polynomial_coefficients]), axis=0)
    np.append(A, np.array([Neck_twist_polynomial_coefficients]), axis=0)

    np.append(A, np.array([trunk_flex_polynomial_coefficients]), axis=0)
    np.append(A, np.array([trunk_side_polynomial_coefficients]), axis=0)
    np.append(A, np.array([trunk_twist_polynomial_coefficients]), axis=0)

    np.append(A, np.array([leg_bending_polynomial_coefficients]), axis=0)

    np.append(A, np.array([right_upper_arm_front_adduction_polynomial_coefficients]), axis=0)
    np.append(A, np.array([right_upper_arm_side_adduction_polynomial_coefficients]), axis=0)
    np.append(A, np.array([right_shoulder_rise_polynomial_coefficients]), axis=0)

    np.append(A, np.array([left_upper_arm_front_adduction_polynomial_coefficients]), axis=0)
    np.append(A, np.array([left_upper_arm_side_adduction_polynomial_coefficients]), axis=0)
    np.append(A, np.array([left_shoulder_rise_polynomial_coefficients]), axis=0)

    np.append(A, np.array([right_lower_arm_front_adduction_polynomial_coefficients]), axis=0)
    np.append(A, np.array([left_lower_arm_front_adduction_polynomial_coefficients]), axis=0)

    np.append(A, np.array([right_wrist_flex_polynomial_coefficients]), axis=0)
    np.append(A, np.array([right_wrist_side_polynomial_coefficients]), axis=0)
    np.append(A, np.array([right_wrist_twist_polynomial_coefficients]), axis=0)

    np.append(A, np.array([left_wrist_flex_polynomial_coefficients]), axis=0)
    np.append(A, np.array([left_wrist_side_polynomial_coefficients]), axis=0)
    np.append(A, np.array([left_wrist_twist_polynomial_coefficients]), axis=0)

    return A


if __name__ == '__main__':
   # reading the input file(M,N)
   A = dREBA_polynomial_matrix_generator()
   dREBA_coeffs = dReba_coeff_generator(M,N,A)

   each_posture_error=[]
   n = len(N)
   for i in range(n):
       m_dREBA = DREBA.Dreba_error_each_posture()
       each_posture_error.append(m_dREBA.dReba_error(A,M[i,:],N[i],dREBA_coeffs))

    print(each_posture_error)