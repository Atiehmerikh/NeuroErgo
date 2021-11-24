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

# import polynomial_coeff_calculator as pCoeff
# import Dreba_coeff_calculator as DREBA
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
    qss = [[-60,30], [-54, 54], [-60,0, 60],\
          [-30, 70], [-40, 40], [-35, 35],\
          [0, 150],\
          [-47,170], [-47, 170], [-2,200], [-2,200], [0, 30], [0, 30],\
          [0, 150], [0, 150],\
          [-53,47], [-53,47], [-40, 30], [-40, 30], [-90, 90], [-90, 90]]


    samples = np.zeros(shape=(sample_size, 21))
    samples_REBA = np.zeros(shape=(sample_size,))
    for i in tqdm(range(sample_size)):      
        a_sample = np.zeros(shape=(21,))
        for j, qs in enumerate(qss):
            minimum = qs[0]
            maximum = qs[1]
            mean_val = (minimum + maximum)/2
            std_val = (maximum - minimum)/6
            a, b = (min(qs) - mean_val) / std_val, (max(qs) - mean_val) / std_val
            a_sample[j] = truncnorm.rvs(a, b, size = 1)[0]
            #a_sample[j] = random.sample(list(range(min(qs), max(qs)+1)),1)[0]
        samples[i,:] = a_sample
        samples_REBA[i] = calc_total_reba(a_sample)
    

    return samples, samples_REBA

if __name__ == '__main__':
    
    joint_samples_train, reba_scores_train = generate_samples(21)
    with open('./matlab/data/input/M.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(joint_samples_train)
    with open('./matlab/data/input/N.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([reba_scores_train])


    # joint_samples_test, reba_scores_test = generate_samples(1000000)
    # with open('./dREBA_matlab/data/input/M_test.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(joint_samples_test)
    # with open('./dREBA_matlab/data/input/N_test.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(reba_scores_test)
