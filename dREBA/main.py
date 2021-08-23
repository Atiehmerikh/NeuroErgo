import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from tqdm import tqdm
from scipy.stats import truncnorm
import csv

import REBA.body_part_reba_calculator.Degree_to_REBA.neck_reba_score as REBA_neck
import REBA.body_part_reba_calculator.Degree_to_REBA.trunk_reba_score as REBA_trunk
import REBA.body_part_reba_calculator.Degree_to_REBA.leg_reba_score as REBA_leg
import REBA.body_part_reba_calculator.Degree_to_REBA.upperarm_reba_score as REBA_UA
import REBA.body_part_reba_calculator.Degree_to_REBA.lowerarm_reba_score as REBA_LA
import REBA.body_part_reba_calculator.Degree_to_REBA.wrist_reba_score as REBA_wrist
import REBA.body_part_reba_calculator.partial_REBA_to_total_REBA as REBA

import polynomial_coeff_calculator as pCoeff
import Dreba_coeff_calculator as DREBA
import sympy as sym
import random
import _pickle as cPickle

def retrieve_from_pickle(file_address):
    f = open(file_address, "rb")
    p = cPickle.Unpickler(f)
    seqs_list = p.load()
    return seqs_list

def store_in_pickle(file_address, data):
    p = cPickle.Pickler(open(file_address, "wb")) 
    p.fast = True 
    p.dump(data)

def calc_total_reba(x):
    return REBA.partial_to_total_REBA([REBA_neck.NeckREBA(list(x[0:3])).neck_reba_score(),\
                                       REBA_trunk.TrunkREBA(list(x[3:6])).trunk_reba_score(),\
                                       REBA_leg.LegREBA([x[6],x[6]]).leg_reba_score(), \
                                       REBA_UA.UAREBA(list(x[7:13])).upper_arm_reba_score(),\
                                       REBA_LA.LAREBA(list(x[13:15])).lower_arm_score(),\
                                       REBA_wrist.WristREBA(list(x[15:])).wrist_reba_score()]).find_total_REBA()

def dReba_coeff_generator(M,N,A):
    # M is a n to m matrix,
    # N is a n to 1 matrix
    # A is a m to 3 matrix
    # n is number of postures
    # m is number of body degrees
    total_error = 0
    n = len(N)
    m_dREBA = DREBA.Dreba_error_each_posture()
    error_list = []
    for i in tqdm(range(n)):
        # for each posture
        each_posture_error = m_dREBA.dReba_symbolic_error(A, M[i, :], N[i])
        error_list.append(each_posture_error)
    
    print("all errors have been collected")

    total_error = sym.Add(*error_list)

    print("total error is computed")

    total_error = sym.sympify(sym.expand((1/n)*total_error))
    w = sym.symbols('w_0:21')
    df_dw=[]
    a = []
    b =[]
    for i in range(len(w)):
        df_dw.append(sym.diff(total_error, w[i]))

    for i in range(len(df_dw)):
        row =[]
        for j in range(1,len(df_dw)+1):
            if(len(df_dw[i].args) != 22):
                raise Exception("a polynomial with some zero coefficients")
            elem = df_dw[i].args[j].args[0]
            row.append(elem)
        a.append(row)
        b.append(df_dw[i].args[0])

    print("before linear solving")

    a = np.array(a, dtype='float')
    b = np.array(b, dtype='float')

    dREBA_coeffs = np.linalg.solve(a, b)
    return dREBA_coeffs

def dREBA_polynomial_matrix_generator():
    A = []

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
    right_upper_arm_side_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-3.5,0],[1,0])
    right_shoulder_rise_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([0, 30.5],[0,1])

    left_upper_arm_front_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-20, 0, 20, 45],  [2,1,2,3])
    left_upper_arm_side_adduction_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([-3.5,0],[1,0])
    left_shoulder_rise_polynomial_coefficients = m_pcoeff.polynomial_coefficients_calculator([0, 30.5],[0,1])

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

    A.append(Neck_flex_polynomial_coefficients)
    A.append(Neck_side_polynomial_coefficients)
    A.append(Neck_twist_polynomial_coefficients)

    A.append(trunk_flex_polynomial_coefficients)
    A.append(trunk_side_polynomial_coefficients)
    A.append(trunk_twist_polynomial_coefficients)

    A.append(leg_bending_polynomial_coefficients)

    A.append(right_upper_arm_front_adduction_polynomial_coefficients)
    A.append(right_upper_arm_side_adduction_polynomial_coefficients)
    A.append(right_shoulder_rise_polynomial_coefficients)

    A.append(left_upper_arm_front_adduction_polynomial_coefficients)
    A.append(left_upper_arm_side_adduction_polynomial_coefficients)
    A.append(left_shoulder_rise_polynomial_coefficients)

    A.append(right_lower_arm_front_adduction_polynomial_coefficients)
    A.append(left_lower_arm_front_adduction_polynomial_coefficients)

    A.append(right_wrist_flex_polynomial_coefficients)
    A.append(right_wrist_side_polynomial_coefficients)
    A.append(right_wrist_twist_polynomial_coefficients)

    A.append(left_wrist_flex_polynomial_coefficients)
    A.append(left_wrist_side_polynomial_coefficients)
    A.append(left_wrist_twist_polynomial_coefficients)

    return np.array(A)

def train_dREBA(sample_size):
    random.seed(1)
    qss = [[-60,0,20], [-54,0, 54], [-60,0, 60],\
          [-30,0,20,60], [-40,0, 40], [-35,0, 35],\
          [0,30,60],\
          [-20,0,20,45], [-20, 0, 20, 45], [-2,0], [-2,0], [0, 30], [0, 30],\
          [0, 60, 100], [0, 60, 100],\
          [-53,-15,15], [-53,-15,15], [-40,0, 30], [-40,0, 30], [-90,0, 90], [-90,0, 90]]


    samples = np.zeros(shape=(sample_size, 21))
    samples_REBA = np.zeros(shape=(sample_size,))
    for i in tqdm(range(sample_size)):      
        a_sample = np.zeros(shape=(21,))
        for j, qs in enumerate(qss):
            a_sample[j] = random.sample(qs,1)[0]
        samples[i,:] = a_sample
        samples_REBA[i] = calc_total_reba(a_sample)

    generator = dREBA_polynomial_matrix_generator()
    dREBA_coeffs = dReba_coeff_generator(samples,samples_REBA,generator)
    return dREBA_coeffs



def generate_samples(sample_size):
    random.seed(2)
    qss = [[-60,0,20], [-54,0, 54], [-60,0, 60],\
          [-30,0,20,60], [-40,0, 40], [-35,0, 35],\
          [0,30,60],\
          [-20,0,20,45], [-20, 0, 20, 45], [-2,0], [-2,0], [0, 30], [0, 30],\
          [0, 60, 100], [0, 60, 100],\
          [-53,-15,15], [-53,-15,15], [-40,0, 30], [-40,0, 30], [-90,0, 90], [-90,0, 90]]


    samples = np.zeros(shape=(sample_size, 21))
    samples_REBA = np.zeros(shape=(sample_size,))
    for i in tqdm(range(sample_size)):      
        a_sample = np.zeros(shape=(21,))
        for j, qs in enumerate(qss):
            minimum = min(qs)
            maximum = max(qs)
            mean_val = (minimum + maximum)/2
            std_val = (maximum - minimum)/6
            a, b = (min(qs) - mean_val) / std_val, (max(qs) - mean_val) / std_val
            a_sample[j] = truncnorm.rvs(a, b, size = 1)[0]
            #a_sample[j] = random.sample(list(range(min(qs), max(qs)+1)),1)[0]
        samples[i,:] = a_sample
        samples_REBA[i] = calc_total_reba(a_sample)
    
    with open('./data/M_test_2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    print("M-test is wirtten.")

    with open('./data/N_test_2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(samples_REBA)
if __name__ == '__main__':
    generate_samples(1000000)
    # dREBA_coeffs = train_dREBA(100000)
    # print(dREBA_coeffs)
    # store_in_pickle("data/coeffs_10000.pkl",dREBA_coeffs)
   # reading the input file(M,N)
#    A = dREBA_polynomial_matrix_generator()

#    dREBA_coeffs = dReba_coeff_generator(M,N,A)

#    each_posture_error=[]
#    n = len(N)
#    for i in range(n):
#        m_dREBA = DREBA.Dreba_error_each_posture()
#        each_posture_error.append(m_dREBA.dReba_error(A,M[i,:],N[i],dREBA_coeffs))

#     print(each_posture_error)