import polynomial_coeff_calculator as pCoeff
import Dreba_coeff_calculator as DREBA

if __name__ == '__main__':
    m_pcoeff = pCoeff.polynomial_generator()
    # polynomial coefficients:(training point,partial reba score)
    Neck_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([0,10,20,80],[0,2,1,2])



    # number of posture for training
    m = 10
    total_error =0
    for i in range(m):
        m_dREBA = DREBA.Dreba_coeff()
        posture_error = m_dREBA.Dreba_expression(poly_coeff,posture_angles,porsture_total_REBA)
        total_error += posture_error

    total_error = sym.sympify(sym.expand(total_error))
    w = sym.symbols('w_0:13')
    for i in range(len(w)):
        df_dwi = sym.diff(total_error, w[i])

