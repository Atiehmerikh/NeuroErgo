import sympy as sym


class Dreba_error_each_posture:
    def dReba_symbolic_error(self, polynomial_coeff, posture_angles, posture_real_REBA):
        w = sym.symbols('w_0:13')
        dREBA = 0
        # number of body segments: Neck flexion, neck side, ...
        m = 21
        Q =[]
        for i in range(m):
            Q.append(polynomial_coeff[i][0]*posture_angles[i]**2 +polynomial_coeff[i][1]*posture_angles[i]
                     +polynomial_coeff[i][2])
        for j in range(m):
            dREBA += w[j]*Q[j]

        dREBA = sym.sympify(sym.expand(dREBA))
        sq_error = sym.sympify(sym.expand(dREBA-posture_real_REBA)**2)
        return sq_error

    def dReba_error(self, polynomial_coeff, posture_angles, posture_real_REBA, dREBA_coeffs):
        dREBA = 0
        # number of body segments: Neck flexion, neck side, ...
        m = 21
        Q =[]
        for i in range(m):
            Q.append(polynomial_coeff[i][0]*posture_angles[i]**2 +polynomial_coeff[i][1]*posture_angles[i]
                     +polynomial_coeff[i][2])
        for j in range(m):
            dREBA += dREBA_coeffs[j]*Q[j]

        sq_error = sym.sympify(sym.expand(dREBA-posture_real_REBA)**2)
        return sq_error
