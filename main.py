import REBA.body_part_reba_calculator.Degree_to_REBA.neck_reba_score as REBA_neck
import REBA.body_part_reba_calculator.Degree_to_REBA.trunk_reba_score as REBA_trunk
import REBA.body_part_reba_calculator.Degree_to_REBA.leg_reba_score as REBA_leg
import REBA.body_part_reba_calculator.Degree_to_REBA.upperarm_reba_score as REBA_UA
import REBA.body_part_reba_calculator.Degree_to_REBA.lowerarm_reba_score as REBA_LA
import REBA.body_part_reba_calculator.Degree_to_REBA.wrist_reba_score as REBA_wrist
import REBA.body_part_reba_calculator.partial_REBA_to_total_REBA as REBA

import numpy as np
import _pickle as cPickle

import multiprocessing as mp
from itertools import product, chain

from tensorflow.keras.layers import Dense, Dropout, Concatenate, concatenate
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Adadelta, Adagrad, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
import tensorflow as tf

from tqdm import tqdm
import math
import gzip
import shutil
import os
import time
import pandas as pd
from functools import partial

# libraries for blackbox optimization
from human_forward_kinematic import *  
import localsolver 

def retrieve_from_pickle(file_address):
    f = open(file_address, "rb")
    p = cPickle.Unpickler(f)
    seqs_list = p.load()
    return seqs_list

def store_in_pickle(file_address, data):
    p = cPickle.Pickler(open(file_address, "wb")) 
    p.fast = True 
    p.dump(data)


def find_largest_power_of_ten(x):
    return int(math.log10(x))
# Neck
def neck_ranges():
    neck_flexion_extension_samples = list(range(-60, 31))
    neck_side_flexion_samples = list(range(-54, 55))
    neck_rotation_samples = list(range(-60, 61))

    return neck_flexion_extension_samples, neck_side_flexion_samples, neck_rotation_samples

def neck_learning_model():
    activation = 'tanh'
    model = Sequential()
    model.add(Dense(3, input_dim=3, activation=activation, name = "neck_model"))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(1))

    return model

def neck_training_model():

    model = neck_learning_model()
    model.compile(optimizer=SGD(lr=0.01), loss='mse')

    neck_flexion_extension_samples, neck_side_flexion_samples, neck_rotation_samples = neck_ranges()
    for e in tqdm(range(60)):
        for i in neck_flexion_extension_samples:
            num_of_data = len(neck_side_flexion_samples) * len(neck_rotation_samples)
            X_train = np.zeros(shape=(num_of_data, 3))
            y_train = np.zeros(shape=(num_of_data,))
            counter = 0
            for j in neck_side_flexion_samples:
                for k in neck_rotation_samples:
                    m_neck =REBA_neck.NeckREBA([i,j,k])
                    X_train[counter, :] = [i,j,k]
                    y_train[counter] = m_neck.neck_reba_score()
                    counter +=1
            model.fit(X_train, y_train, verbose=0)

    model.save('./data/neck_DNN.model')

def neck_model_test():
    neck_flexion_extension_samples, neck_side_flexion_samples, neck_rotation_samples = neck_ranges()
    model = load_model('./data/neck_DNN.model')

    abs_sum = 0
    for i in tqdm(neck_flexion_extension_samples):
        num_of_data = len(neck_side_flexion_samples) * len(neck_rotation_samples)
        X_train = np.zeros(shape=(num_of_data, 3))
        y_train = np.zeros(shape=(num_of_data,))
        counter = 0
        for j in neck_side_flexion_samples:
            for k in neck_rotation_samples:
                m_neck =REBA_neck.NeckREBA([i,j,k])
                m_neck_reba_score = m_neck.neck_reba_score()
                X_train[counter, :] = [i,j,k]
                y_train[counter] = m_neck_reba_score
                counter += 1

        pred = model.predict(X_train)
        for y_true, y_pred in zip(y_train, pred):
            abs_sum += math.fabs(y_true - y_pred)

    return (abs_sum, len(neck_flexion_extension_samples) * len(neck_side_flexion_samples) * len(neck_rotation_samples))

# trunk
def trunk_ranges():
    trunk_flexion_extension_samples = range(-30, 71)
    trunk_side_flexion_samples = range(-40, 41)
    trunk_rotation_samples = range(-35, 36)

    return trunk_flexion_extension_samples, trunk_side_flexion_samples, trunk_rotation_samples

def trunk_learning_model():
    activation = 'softplus'
    model = Sequential()
    model.add(Dense(3, input_dim=3, activation=activation, name = "trunk_model"))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(1))

    return model

def trunk_training_model():
    model = trunk_learning_model()
    model.compile(optimizer=Adam(lr=0.00001), loss='mse')

    trunk_flexion_extension_samples, trunk_side_flexion_samples, trunk_rotation_samples = trunk_ranges()

    for e in tqdm(range(400)):
        for i in trunk_flexion_extension_samples:
            num_of_data = len(trunk_side_flexion_samples) * len(trunk_rotation_samples)
            X_train = np.zeros(shape=(num_of_data, 3))
            y_train = np.zeros(shape=(num_of_data,))
            counter = 0
            for j in trunk_side_flexion_samples:
                for k in trunk_rotation_samples:
                    m_trunk = REBA_trunk.TrunkREBA([i,j,k])
                    X_train[counter, :] = [i,j,k]
                    y_train[counter] = m_trunk.trunk_reba_score()
                    counter += 1
            model.fit(X_train, y_train, verbose=0)

    model.save('./data/trunk_DNN.model')

def trunk_model_test():
    trunk_flexion_extension_samples, trunk_side_flexion_samples, trunk_rotation_samples = trunk_ranges()
    model = load_model('./data/trunk_DNN.model')

    abs_sum = 0
    for i in tqdm(trunk_flexion_extension_samples):
        num_of_data = len(trunk_side_flexion_samples) * len(trunk_rotation_samples)
        X_train = np.zeros(shape=(num_of_data, 3))
        y_train = np.zeros(shape=(num_of_data,))
        counter = 0
        for j in trunk_side_flexion_samples:
            for k in trunk_rotation_samples:
                m_trunk = REBA_trunk.TrunkREBA([i,j,k])
                X_train[counter, :] = [i,j,k]
                y_train[counter] = m_trunk.trunk_reba_score()
                counter += 1

        pred = model.predict(X_train)
        for y_true, y_pred in zip(y_train, pred):
            abs_sum += math.fabs(y_true - y_pred)

    return (abs_sum, len(trunk_flexion_extension_samples) * len(trunk_side_flexion_samples) * len(trunk_rotation_samples))

# Legs
def leg_ranges():
    legs_flexion_samples = range(0, 151)
    return legs_flexion_samples

def leg_learning_model():
    activation = 'tanh'
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation=activation, name = "leg_model"))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1))

    return model

def leg_training_model():
    model = leg_learning_model()
    model.compile(optimizer=SGD(lr=0.01), loss='mse')

    legs_flexion_samples = leg_ranges()
    
    for e in tqdm(range(100)):
        num_of_data = len(legs_flexion_samples)
        X_train = np.zeros(shape=(num_of_data, 1))
        y_train = np.zeros(shape=(num_of_data,))
        counter = 0
        for i in legs_flexion_samples:    
            m_leg = REBA_leg.LegREBA([i,i])
            X_train[counter, :] = [i]
            y_train[counter] = m_leg.leg_reba_score()
            counter += 1
        model.fit(X_train, y_train, verbose=0)

    model.save('./data/leg_DNN.model')

def leg_model_test():
    legs_flexion_samples = leg_ranges()
    model = load_model('./data/leg_DNN.model')

    num_of_data = len(legs_flexion_samples)
    X_train = np.zeros(shape=(num_of_data, 1))
    y_train = np.zeros(shape=(num_of_data,))
    counter = 0
    abs_sum = 0
    for i in legs_flexion_samples:    
        m_leg = REBA_leg.LegREBA([i,i])
        X_train[counter, :] = [i]
        y_train[counter] = m_leg.leg_reba_score()
        counter += 1

    pred = model.predict(X_train)
    for y_true, y_pred in zip(y_train, pred):
        abs_sum += math.fabs(y_true - y_pred)
    return (abs_sum, len(legs_flexion_samples))

# Upper Arm
def upper_arm_ranges():
    right_upper_arm_flexion_extension_samples = [-47, 165, 170] + [*range(-45, 171, 9)]
    left_upper_arm_flexion_extension_samples = [-47, 165, 170] + [*range(-45, 171, 9)]
    right_upper_arm_adduction_abduction_samples = [-2, -1] + [*range(0, 201, 10)]
    left_upper_arm_adduction_abduction_samples = [-2, -1] + [*range(0, 201, 10)]
    right_shoulder_raise_samples = [*range(0, 31, 6)]
    left_shoulder_raise_samples = [*range(0, 31, 6)]
    return right_upper_arm_flexion_extension_samples, left_upper_arm_flexion_extension_samples, \
           right_upper_arm_adduction_abduction_samples, left_upper_arm_adduction_abduction_samples, \
           right_shoulder_raise_samples, left_shoulder_raise_samples

def upper_arm_learning_model():
    activation = 'tanh'
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation=activation, name = "upper_arm_model"))
    model.add(Dense(6, activation=activation))
    model.add(Dense(6, activation=activation))
    model.add(Dense(6, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1))

    return model

def upper_arm_training_model():
    model = upper_arm_learning_model()
    model.compile(optimizer=SGD(lr=0.001), loss='mse')

    right_upper_arm_flexion_extension_samples , left_upper_arm_flexion_extension_samples, \
    right_upper_arm_adduction_abduction_samples, left_upper_arm_adduction_abduction_samples, \
    right_shoulder_raise_samples, left_shoulder_raise_samples  = upper_arm_ranges()

    for e in tqdm(range(40)):
        for i in right_upper_arm_flexion_extension_samples:
            for j in left_upper_arm_flexion_extension_samples:
                num_of_data = len(right_upper_arm_adduction_abduction_samples) * len(left_upper_arm_adduction_abduction_samples) *\
                            len(right_shoulder_raise_samples) * len(left_shoulder_raise_samples)
                X_train = np.zeros(shape=(num_of_data, 6))
                y_train = np.zeros(shape=(num_of_data,))
                counter = 0
                for k in right_upper_arm_adduction_abduction_samples:
                    for l in left_upper_arm_adduction_abduction_samples:
                        for m in right_shoulder_raise_samples:
                            for n in left_shoulder_raise_samples:
                                m_UA = REBA_UA.UAREBA([i, j,k,l,m,n])
                                X_train[counter, :] = [i, j,k,l,m,n]
                                y_train[counter] = m_UA.upper_arm_reba_score()
                                counter += 1
                model.fit(X_train, y_train, verbose=0)

    model.save('./data/upper_arm_DNN.model')

def upper_arm_model_test():
    right_upper_arm_flexion_extension_samples , left_upper_arm_flexion_extension_samples, \
    right_upper_arm_adduction_abduction_samples, left_upper_arm_adduction_abduction_samples, \
    right_shoulder_raise_samples, left_shoulder_raise_samples  = upper_arm_ranges()

    model = load_model('./data/upper_arm_DNN.model')

    abs_sum = 0
    for i in right_upper_arm_flexion_extension_samples:
        for j in left_upper_arm_flexion_extension_samples:
            num_of_data = len(right_upper_arm_adduction_abduction_samples) * len(left_upper_arm_adduction_abduction_samples) *\
                          len(right_shoulder_raise_samples) * len(left_shoulder_raise_samples)
            X_train = np.zeros(shape=(num_of_data, 6))
            y_train = np.zeros(shape=(num_of_data,))
            counter = 0
            for k in right_upper_arm_adduction_abduction_samples:
                for l in left_upper_arm_adduction_abduction_samples:
                    for m in right_shoulder_raise_samples:
                        for n in left_shoulder_raise_samples:
                            m_UA = REBA_UA.UAREBA([i, j,k,l,m,n])
                            X_train[counter, :] = [i, j,k,l,m,n]
                            y_train[counter] = m_UA.upper_arm_reba_score()
                            counter += 1

            pred = model.predict(X_train)
            for y_true, y_pred in zip(y_train, pred):
                abs_sum += math.fabs(y_true - y_pred)
    return (abs_sum, len(right_upper_arm_flexion_extension_samples) * len(left_upper_arm_flexion_extension_samples) * \
                     len(right_upper_arm_adduction_abduction_samples) * len(left_upper_arm_adduction_abduction_samples) *\
                     len(right_shoulder_raise_samples) * len(left_shoulder_raise_samples))


# Lower Arm
def lower_arm_ranges():
    right_lower_arm_flexion_samples = range(0, 151)
    left_lower_arm_flexion_samples = range(0,151)
    return right_lower_arm_flexion_samples, left_lower_arm_flexion_samples

def lower_arm_learning_model():
    activation = 'tanh'
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation=activation, name = "lower_arm_model"))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1))

    return model

def lower_arm_training_model():
    model = lower_arm_learning_model()
    model.compile(optimizer=Nadam(lr=0.001), loss='mse')

    right_lower_arm_flexion_samples, left_lower_arm_flexion_samples = lower_arm_ranges()
    for e in tqdm(range(100)):
        num_of_data = len(right_lower_arm_flexion_samples) * len(left_lower_arm_flexion_samples)
        X_train = np.zeros(shape=(num_of_data, 2))
        y_train = np.zeros(shape=(num_of_data,))
        counter = 0
        for i in right_lower_arm_flexion_samples:
            for j in left_lower_arm_flexion_samples:
                m_LA = REBA_LA.LAREBA([i,j])
                X_train[counter, :] = [i, j]
                y_train[counter] = m_LA.lower_arm_score()
                counter += 1

        model.fit(X_train, y_train, verbose=0)

    model.save('./data/lower_arm_DNN.model')

def lower_arm_model_test():
    right_lower_arm_flexion_samples, left_lower_arm_flexion_samples = lower_arm_ranges()
    model = load_model('./data/lower_arm_DNN.model')
    num_of_data = len(right_lower_arm_flexion_samples) * len(left_lower_arm_flexion_samples)
    X_train = np.zeros(shape=(num_of_data, 2))
    y_train = np.zeros(shape=(num_of_data,))
    counter = 0
    
    for i in right_lower_arm_flexion_samples:
        for j in left_lower_arm_flexion_samples:
            m_LA = REBA_LA.LAREBA([i,j])
            X_train[counter, :] = [i, j]
            y_train[counter] = m_LA.lower_arm_score()
            counter += 1
    pred = model.predict(X_train)

    abs_sum = 0
    for y_true, y_pred in zip(y_train, pred):
        abs_sum += math.fabs(y_true - y_pred)
    
    return (abs_sum, num_of_data)

# Wrist
def wrist_ranges():
    right_wrist_flexion_extension_samples = [-53,  47] + [*range(-45, 46, 9)]
    left_wrist_flexion_extension_samples =  [-53,  47] + [*range(-45, 46, 9)]
    right_wrist_side_adduction_abduction_samples = range(-40, 31, 10)
    left_wrist_side_adduction_abduction_samples = range(-40, 31, 10)
    right_wrist_rotation_samples = range(-90, 91, 10)
    left_wrist_rotation_samples = range(-90, 91, 10)
    return right_wrist_flexion_extension_samples, left_wrist_flexion_extension_samples, \
    right_wrist_side_adduction_abduction_samples, left_wrist_side_adduction_abduction_samples, \
    right_wrist_rotation_samples, left_wrist_rotation_samples

def wrist_learning_model():
    activation = 'tanh'
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation=activation, name = "wrist_model"))
    model.add(Dense(6, activation=activation))
    model.add(Dense(6, activation=activation))
    model.add(Dense(6, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1))

    return model

def wrist_training_model():
    model = wrist_learning_model()
    model.compile(optimizer=SGD(lr=0.001), loss='mse')

    right_wrist_flexion_extension_samples, left_wrist_flexion_extension_samples, \
    right_wrist_side_adduction_abduction_samples, left_wrist_side_adduction_abduction_samples, \
    right_wrist_rotation_samples, left_wrist_rotation_samples = wrist_ranges()

    for e in tqdm(range(40)):
        for i in right_wrist_flexion_extension_samples:
            for j in left_wrist_flexion_extension_samples:
                num_of_data = len(right_wrist_side_adduction_abduction_samples) * len(left_wrist_side_adduction_abduction_samples) *\
                            len(right_wrist_rotation_samples) * len(left_wrist_rotation_samples)
                X_train = np.zeros(shape=(num_of_data, 6))
                y_train = np.zeros(shape=(num_of_data,))
                counter = 0
                for k in right_wrist_side_adduction_abduction_samples:
                    for l in left_wrist_side_adduction_abduction_samples:
                        for m in right_wrist_rotation_samples:
                            for n in left_wrist_rotation_samples:
                                m_wrist = REBA_wrist.WristREBA([i, j,k,l,m,n])
                                X_train[counter, :] = [i, j,k,l,m,n]
                                y_train[counter] = m_wrist.wrist_reba_score()
                                counter += 1
                model.fit(X_train, y_train, verbose=0)

    model.save('./data/wrist_DNN.model')

def wrist_model_test():
    right_wrist_flexion_extension_samples, left_wrist_flexion_extension_samples, \
    right_wrist_side_adduction_abduction_samples, left_wrist_side_adduction_abduction_samples, \
    right_wrist_rotation_samples, left_wrist_rotation_samples = wrist_ranges()

    model = load_model('./data/wrist_DNN.model')
    abs_sum = 0
    for i in right_wrist_flexion_extension_samples:
            for j in left_wrist_flexion_extension_samples:
                num_of_data = len(right_wrist_side_adduction_abduction_samples) * len(left_wrist_side_adduction_abduction_samples) *\
                            len(right_wrist_rotation_samples) * len(left_wrist_rotation_samples)
                X_train = np.zeros(shape=(num_of_data, 6))
                y_train = np.zeros(shape=(num_of_data,))
                counter = 0
                for k in right_wrist_side_adduction_abduction_samples:
                    for l in left_wrist_side_adduction_abduction_samples:
                        for m in right_wrist_rotation_samples:
                            for n in left_wrist_rotation_samples:
                                m_wrist = REBA_wrist.WristREBA([i, j,k,l,m,n])
                                X_train[counter, :] = [i, j,k,l,m,n]
                                y_train[counter] = m_wrist.wrist_reba_score()
                                counter += 1
                pred = model.predict(X_train)

                for y_true, y_pred in zip(y_train, pred):
                    abs_sum += math.fabs(y_true - y_pred)
    
    return (abs_sum, len(right_wrist_flexion_extension_samples) * len(left_wrist_flexion_extension_samples) * \
                     len(right_wrist_side_adduction_abduction_samples) * len(left_wrist_side_adduction_abduction_samples) *\
                     len(right_wrist_rotation_samples) * len(left_wrist_rotation_samples))

# Partial to Total REBA
def total_reba_from_partial_ranges():
    neck = [1,2,3]
    trunk = [1,2,3,4,5]
    leg = [1,2,3,4]
    upper_arm = [1,2,3,4,5,6]
    lower_arm = [1,2]
    wrist = [1,2,3]
    return (neck, trunk, leg, upper_arm, lower_arm, wrist)

def total_reba_from_partial_learning_model():
    activation = 'tanh'
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation=activation))
    # model.add(Dense(6, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1))

    return model

def total_reba_from_partial_training_model():
    model = total_reba_from_partial_learning_model()
    model.compile(optimizer=SGD(lr=0.001), loss='mse')

    neck, trunk, leg, upper_arm, lower_arm, wrist = total_reba_from_partial_ranges()

    for e in tqdm(range(300)):
        num_of_data = len(neck) * len(trunk) * len(leg) * len(upper_arm) * len(lower_arm) * len(wrist)
        X_train = np.zeros(shape=(num_of_data, 6))
        y_train = np.zeros(shape=(num_of_data,))
        counter = 0
        for i in neck:
            for j in trunk:
                for k in leg:
                    for l in upper_arm:
                        for m in lower_arm:
                            for n in wrist:
                                m_REBA = REBA.partial_to_total_REBA([i,j,k,l,m,n])
                                X_train[counter, :] = [i, j,k,l,m,n]
                                y_train[counter] = m_REBA.find_total_REBA()
                                counter += 1

        model.fit(X_train, y_train, verbose=0)

    model.save('./data/total_reba_from_partial_DNN.model')

def total_reba_from_partial_model_test():
    neck, trunk, leg, upper_arm, lower_arm, wrist = total_reba_from_partial_ranges()
    model = load_model('./data/total_reba_from_partial_DNN.model')
    abs_sum = 0
    num_of_data = len(neck) * len(trunk) * len(leg) * len(upper_arm) * len(lower_arm) * len(wrist)
    X_train = np.zeros(shape=(num_of_data, 6))
    y_train = np.zeros(shape=(num_of_data,))
    counter = 0
    for i in neck:
        for j in trunk:
            for k in leg:
                for l in upper_arm:
                    for m in lower_arm:
                        for n in wrist:
                            m_REBA = REBA.partial_to_total_REBA([i,j,k,l,m,n])
                            X_train[counter, :] = [i, j,k,l,m,n]
                            y_train[counter] = m_REBA.find_total_REBA()
                            counter += 1

    pred = model.predict(X_train)

    for y_true, y_pred in zip(y_train, pred):
        abs_sum += math.fabs(y_true - y_pred)

    return(abs_sum, num_of_data)


# Approximate REBA
neck_model                    =  load_model('./data/neck_DNN.model')
trunk_model                   =  load_model('./data/trunk_DNN.model')
leg_model                     =  load_model('./data/leg_DNN.model')
upper_arm_model               =  load_model('./data/upper_arm_DNN.model')
lower_arm_model               =  load_model('./data/lower_arm_DNN.model')
wrist_model                   =  load_model('./data/wrist_DNN.model')
total_reba_from_partial_model =  load_model('./data/total_reba_from_partial_DNN.model')
                                 

def neuro_neck_REBA(neck_vector):
    return neck_model.predict([neck_vector])[0][0]

def neuro_trunk_REBA(trunk_vector):
    return trunk_model.predict([trunk_vector])[0][0]

def neuro_leg_REBA(leg_vector):
    return leg_model.predict([leg_vector])[0][0]

def neuro_upper_arm_REBA(upper_arm_vector):
    return upper_arm_model.predict([upper_arm_vector])[0][0]

def neuro_lower_arm_REBA(lower_arm_vector):
    return lower_arm_model.predict([lower_arm_vector])[0][0]

def neuro_wrist_REBA(wrist_vector):
    return wrist_model.predict([wrist_vector])[0][0]

def neuro_REBA (neck_vector, trunk_vector, leg_vector, upper_arm_vector, lower_arm_vector, wrist_vector):
    return total_reba_from_partial_model.predict(np.array([[neuro_neck_REBA(neck_vector),           \
                                                  neuro_trunk_REBA(trunk_vector),         \
                                                  neuro_leg_REBA(leg_vector),             \
                                                  neuro_upper_arm_REBA(upper_arm_vector), \
                                                  neuro_lower_arm_REBA(lower_arm_vector), \
                                                  neuro_wrist_REBA(wrist_vector)]]))[0][0]


def create_super_model():
    # neck_model = neck_learning_model()
    # trunk_model = trunk_learning_model()
    # leg_model = leg_learning_model()
    # upper_arm_model = upper_arm_learning_model()
    # lower_arm_model = lower_arm_learning_model()
    # wrist_model = wrist_learning_model()
    # partial_to_total_model = total_reba_from_partial_learning_model()

    model_concat = concatenate([neck_model.output, trunk_model.output, leg_model.output, upper_arm_model.output, lower_arm_model.output, wrist_model.output])
    # model_concat = Dense(1, activation='softmax')(model_concat)
    # model = Model(inputs=[model1.input, model2.input], outputs=model_concat)

    composed_model = Model(
        inputs=[neck_model.input, trunk_model.input, leg_model.input, upper_arm_model.input, lower_arm_model.input, wrist_model.input],
        outputs=[total_reba_from_partial_model(model_concat)]
    )


    return composed_model
    #return Concatenate()([neck_model, trunk_model, leg_model.output, upper_arm_model, lower_arm_model, wrist_model])

def identity(x):
    return list(x)

def calc_total_reba(x, y):
    return [REBA.partial_to_total_REBA([REBA_neck.NeckREBA(list(x[0:3])).neck_reba_score(),\
                                       REBA_trunk.TrunkREBA(list(x[3:6])).trunk_reba_score(),\
                                       REBA_leg.LegREBA([x[6],x[6]]).leg_reba_score(), \
                                       REBA_UA.UAREBA(list(y[0:6])).upper_arm_reba_score(),\
                                       REBA_LA.LAREBA(list(y[6:8])).lower_arm_score(),\
                                       REBA_wrist.WristREBA(list(y[8:])).wrist_reba_score()]).find_total_REBA()]

def generate_super_model_training_data():
    counter = 1
    num_of_data = 4 * 4 * 2 * 2 * 2 * 2 *\
                  3 * 3 *\
                  3 * 3 * 3 * 3 * 3 * 3
        #                    3 * 3 * 3 *\
        #                    4 * 3 * 3 *\
        #                    3 *\
        #                    5 * 5 * 2 * 2 * 2 * 2 *\
        #                    3 * 3 *\
        #                    4 * 4 * 3 * 3 * 3 * 3
    data = {
        'neck_model_input': np.zeros(shape=(num_of_data, 3)),
        'trunk_model_input': np.zeros(shape=(num_of_data, 3)),
        'leg_model_input': np.zeros(shape=(num_of_data, 1)), 
        'upper_arm_model_input': np.zeros(shape=(num_of_data, 6)), 
        'lower_arm_model_input': np.zeros(shape=(num_of_data, 2)), 
        'wrist_model_input': np.zeros(shape=(num_of_data, 6)),
        'y':{}
    }
    y = {'sequential': np.zeros(shape=(num_of_data, 1))}
    for sample_part_1 in product([-60,0,20], [-54,0, 54], [-60,0, 60],\
                                      [-30,0,20,60], [-40,0, 40], [-35,0, 35],\
                                      [0,30,60]):

        with mp.Pool(16) as workers:
            sample_part_2 = np.array(workers.map(identity, product(
                                        [-20,0,20,45], [-20, 0, 20, 45], [-2,0], [-2,0], [0, 30], [0, 30],\
                                        [0, 60, 100], [0, 60, 100],\
                                        [-53,-15,15], [-53,-15,15], [-40,0, 30], [-40,0, 30], [-90,0, 90], [-90,0, 90])))
                                        
            

            data['neck_model_input'][:, :] = np.tile(np.array(list(sample_part_1[0:3])), (num_of_data,1)).tolist()
            data['trunk_model_input'][:, :] = np.tile(np.array(list(sample_part_1[3:6])), (num_of_data,1)).tolist()
            data['leg_model_input'][:, :] = np.tile(np.array([sample_part_1[6]]), (num_of_data,1)).tolist()
            data['upper_arm_model_input'][:, :] = sample_part_2[:, 0:6].tolist()
            data['lower_arm_model_input'][:, :] = sample_part_2[:, 6:8].tolist() #np.tile(np.array(list(sample_part_1[7:9])), (num_of_data,1)).tolist() #sample_part_2[:, 6:8].tolist()
            data['wrist_model_input'][:, :] = sample_part_2[:, 8:].tolist()
            # y['sequential'][counter-1, :] = [REBA.partial_to_total_REBA([REBA_neck.NeckREBA(list(sample[0:3])).neck_reba_score(),\
            #                                                             REBA_trunk.TrunkREBA( list(sample[3:6])).trunk_reba_score(),\
            #                                                             REBA_leg.LegREBA([sample[6],sample[6]]).leg_reba_score(), \
            #                                                             REBA_UA.UAREBA(list(sample[7:13])).upper_arm_reba_score(),\
            #                                                             REBA_LA.LAREBA(list(sample[13:15])).lower_arm_score(),\
            #                                                             REBA_wrist.WristREBA(list(sample[15:21])).wrist_reba_score()]).find_total_REBA()]
            y['sequential'][:, :] = np.apply_along_axis(lambda y: calc_total_reba(sample_part_1, y), axis=1, arr=sample_part_2).tolist()
            data['y'] = y
            file_number = ''
            file_number = file_number.join(['0']* (3- find_largest_power_of_ten(counter))) + str(counter)
            file_name = './data/super_samples/' + file_number + '.pickle'
            store_in_pickle(file_name, data)
            
            with open(file_name, 'rb') as f_in, gzip.open('./data/super_samples/' + file_number + ".gz", 'wb') as f_out:
                f_out.writelines(f_in)
                os.remove(file_name)
            counter += 1

        

        print(counter)

def super_model_train():

    super_model = create_super_model()
    super_model.compile(optimizer=SGD(lr=0.001), loss='mse')

    # uncomment if you have a pretrained model
    #super_model = load_model('./data/super_model_DNN.model')

    print("training is started!")
    for i in tqdm(range(1, 2917)):
        file_number = ''
        file_number = file_number.join(['0']* (3- find_largest_power_of_ten(i))) + str(i)
        file_name = './data/super_samples/' + file_number + '.pickle'
        zipped_file_name = './data/super_samples/' + file_number + '.gz'
        data = {}
        with gzip.open(zipped_file_name, 'rb') as f_in:
            with open(file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                data = retrieve_from_pickle(file_name)
                os.remove(file_name)
        
        super_model.fit(data, data['y'], verbose = 0)
        super_model.save('./data/super_model_DNN.model')


def super_model_training_error():
    
    super_model = load_model('./data/super_model_DNN.model')
    abs_sum = 0
    num_of_data = 0

    print("testing is started!")
    for i in tqdm(range(1, 2917)):
        file_number = ''
        file_number = file_number.join(['0']* (3- find_largest_power_of_ten(i))) + str(i)
        file_name = './data/super_samples/' + file_number + '.pickle'
        zipped_file_name = './data/super_samples/' + file_number + '.gz'
        data = {}
        with gzip.open(zipped_file_name, 'rb') as f_in:
            with open(file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                data = retrieve_from_pickle(file_name)
                os.remove(file_name)
        
        pred = super_model.predict(data)
        y_train = data['y']['sequential']
        abs_sum = 0
        num_of_data += len(y_train)

        abs_sum += np.sum(np.absolute(np.subtract(y_train, pred)))

    return(abs_sum/ num_of_data, abs_sum, num_of_data)


def justify_reba_prediction(pred):
    p = pred
    if(p > 15):
        p = 15
    elif(p < 1):
        p = 1
    else:
        p = round(p)
    return p

def super_model_test_error():
    
    super_model = load_model('./data/super_model_DNN.model')
    abs_sum = 0
    num_of_data = 1000000
    data = {
        'neck_model_input': np.zeros(shape=(num_of_data, 3)),
        'trunk_model_input': np.zeros(shape=(num_of_data, 3)),
        'leg_model_input': np.zeros(shape=(num_of_data, 1)), 
        'upper_arm_model_input': np.zeros(shape=(num_of_data, 6)), 
        'lower_arm_model_input': np.zeros(shape=(num_of_data, 2)), 
        'wrist_model_input': np.zeros(shape=(num_of_data, 6))
    }
    feature_data = pd.read_csv('./dREBA/matlab/data/input/M_test.csv', header=None)

    data['neck_model_input'][:, :] = feature_data.iloc[:, 0:3].values.tolist()
    data['trunk_model_input'][:, :] = feature_data.iloc[:, 3:6].values.tolist()
    data['leg_model_input'][:, :] = np.reshape(feature_data.iloc[:, 6].values.tolist(), (num_of_data,1)).tolist()
    data['upper_arm_model_input'][:, :] = feature_data.iloc[:, 7:13].values.tolist()
    data['lower_arm_model_input'][:, :] = feature_data.iloc[:, 13:15].values.tolist()
    data['wrist_model_input'][:, :] = feature_data.iloc[:, 15:21].values.tolist()

    target_data = pd.read_csv('./dREBA/matlab/data/input/N_test.csv', header=None)


    pred = super_model.predict(data)
    pred = list(chain(*pred))
    for i in range(len(pred)):
        pred[i] = justify_reba_prediction(pred[i])


    y_target = target_data.iloc[:,1].values.tolist()
    abs_sum = 0

    errors = np.absolute(np.subtract(y_target, pred))
    f = open("./data/neuro_errors.csv", "w")
    f2 = open("./data/neuro_estimation.csv", "w")
    for i in range(num_of_data-1):
        f.write(str(errors[i]))
        f2.write(str(pred[i]))
        f.write('\n')
        f2.write('\n')
    f.write(str(errors[num_of_data-1]))
    f2.write(str(pred[num_of_data-1]))
    f.close()
    f2.close()
    abs_sum = np.sum(errors)

    return(abs_sum/ num_of_data, abs_sum, num_of_data)





# for black-box optmization 
def objective_function(context, initial_joint):
    initial = initial_joint
    num_of_data = 1
    data = {
        'neck_model_input': np.zeros(shape=(num_of_data, 3)),
        'trunk_model_input': np.zeros(shape=(num_of_data, 3)),
        'leg_model_input': np.zeros(shape=(num_of_data, 1)), 
        'upper_arm_model_input': np.zeros(shape=(num_of_data, 6)), 
        'lower_arm_model_input': np.zeros(shape=(num_of_data, 2)), 
        'wrist_model_input': np.zeros(shape=(num_of_data, 6))
    }

    angles=[]
    for i in range(21):
        #angles.append(context.get(i))
        angles.append(context[i])

    data['neck_model_input'][:, :] = [angles[0:3]]
    data['trunk_model_input'][:, :] = [angles[3:6]]
    data['leg_model_input'][:, :] = [[angles[6]]]
    data['upper_arm_model_input'][:, :] = [angles[7:13]]
    data['lower_arm_model_input'][:, :] = [angles[13:15]]
    data['wrist_model_input'][:, :] = [angles[15:21]]

    pred = super_model_for_optimization.predict(data)
    pred = list(chain(*pred))[0]
    pred = justify_reba_prediction(pred)

    fk = forward_kinematics(angles)
    end_effector_position = fk.total_human_body_FK()
    dist = np.linalg.norm([initial[i] - context[i] for i in range(len(initial))])
    return 0.25 * pred + 0.75 * dist


def real_data_provider():
    joints = [
        [-2.521828455, 13.86098367, 0.0,
         26.74368395, 5.343683243, 0.0, 
         max(88.56745626,	96.6036250487339),	41.92672174, 9.598638383,	
         73.62039271, 62.54834548, 0.0, 
         0.0, 55.66712169, 59.06936301, 
         24.35608804, -12.04060774,-1.719131321,	
         10.02046955, -1.719131321,	10.02046955],
        [26.48733624, 22.14641858, 0.0,
         15.90242956, 1.008251178, 0.0,		
         max(16.66457137, 126.5839527),	58.80180918, 70.12312593,
         100.7194402, 94.01398722, 17.39755071,
         29.40632989, 47.15635696, 29.07315401,
         6.783288906, 0.0, -2.579181052,
         -0.05729578906, -2.579181052, -0.05729578906],
        [29.20921193, 31.87697653, 0.0,
         -1.08868532, 0.5246335074,	0.0,
         max(79.68848194, 102.4155273),	38.00100076, 33.59184473,
         71.51840573, 79.39716419, 21.34605395,
         15.18876822, 37.1554291, 94.87602487,
         24.49464847, 0.0, -1.833778,
         0.05729578906, -1.833778, 0.05729578906],
        [-3.497213697, 23.97266054, 0.0,
         19.99877181, 36.02931359, 0.0,		
         max(60.19828072, 96.20003713),	49.90925975, 18.73754177,
         54.83030708, 5.731967965, 28.29426008,
         0.0, 15.84580755, 24.49464847,	
         16.4635952, -52.3952303, 8.916796253,	
         27.66184465, 8.916796253, -37.66184465],
        [0.8594689248, 14.74802917,	0.0,
         -19.99877181, 4.02526877,	0.0,
         max(10.5798441, 11.18676388), 48.54741689,	28.1154286,
         76.70292825, 41.6688274, 29.57306994,	
         28.75173915, 99.09082809, 16.8632196,	
         -52.34466073,	9.598638383, 7.816449009, 
         2.980724879, 7.816449009,	2.980724879],
        [11.82949905, 13.34000882, 0.0,	
         -5.336218588,	0.5667250047, 0.0,
         max(84.66378141, 100.2532871),	43.6972753,	46.44899776,
         69.51268489, 79.0472158, 28.29426008,
         11.5954424, 31.78833062, 67.16988933,	
         12.57811866, 8.506146953, 2.980724879,
         -8.568979552, 2.980724879,	-8.568979552],
        [13.59162277, 38.21833344,	0.0,
         43.15701348, 30.74733817,	0.0,
         max(111.7773031, 107.5176753),	55.18001173, 23.9358894,
         58.33175675, 70.60978761,	3.497213697,	
         29.39775711, 44.02801981,	61.37988921,
         -42.18332997,	15.6345989,	-8.047846247,
         -10.83608944,	-8.047846247,	-10.83608944],
        [29.69256357, 48.0752575, 0.0,	
         -14.41834524, 4.45809523, 0.0,
         max(144.7856001, 95.45131981),	19.60981069, 69.63496512,
         15.20360409, 39.91474948, 29.88639405,
         9.671555142, 2.562558733, 97.29626797,
         8.506146953, 41.32292462, -4.991042573,
         -15.01073396, -4.991042573, -15.01073396],
        [5.451319812, 28.53904924,	0.0,	
         6.776690914, 7.908910471,	0.0,
         max(137.7314156, 137.4764848),	67.97568716, 69.57383715,
         86.44538186, 92.00576193,	14.0046121,	
         9.55533142, 43.44790027, 41.06193147,
         38.71544899, -11.47834095,	-4.588565736,
         -10.48627587,	-4.588565736,	-10.48627587],
        [-57.03467226,	53.95608175, 0.0,	
         35.59133528, 4.86590026, 0.0,
         max(73.08216595, 47.5458498), 65.03907961, 17.82423582, 
         63.64062321, 76.82064807, 12.94408154, 
         14.18183373, 27.37700618, 85.06646877, 
         -6.783288906, 35.11491567, 6.776690914, 
         10.42801239, 6.776690914, 10.42801239],
        [-14.77358515, 14.32367983,	0.0,
         -18.24011776, 2.288359392,	0.0,
         max(5.126400082, 13.34447671),	29.1908472,	45.81319361,
         16.8632196, 47.77838594, 29.68755299,	
         29.84386808, 21.56518502, 9.598638383,
         31.57012867, 18.37748066,	-6.430619681,
         2.349785605, -6.430619681,	2.349785605],
        [-3.726853142, 20.51640523,	0.0,
         8.800821189, 7.970146712, 0.0,
         max(85.23897314, 95.33621859),	59.80132232, 44.43869463,
         90.1718876, 93.72685314, 3.095477741,
         7.065272931, 52.62710189,	100.2532871,	
         14.30364872, 0.0, 4.76102686,
         0.2291837292,	4.76102686,	0.2291837292]
    ]
    return joints
    



if __name__ == "__main__": 

    #super_model_test_error()
    
    # tf.compat.v1.disable_eager_execution()
    # super_model = load_model('./data/super_model_DNN.model')

    # first = K.gradients(neck_model.outputs, neck_model.inputs)
    # first = K.gradients(first, super_model.inputs)
    # print(first)
    ###    A balck-box optimization method ###
    super_model_for_optimization = load_model('./data/super_model_DNN.model')
    joints = real_data_provider()
    # print(objective_function([0] * 21))
    # qss = [[-60,0,20], [-54,0, 54], [-60,0, 60],\
    #       [-30,0,20,60], [-40,0, 40], [-35,0, 35],\
    #       [0,30,60],\
    #       [-20,0,20,45], [-20, 0, 20, 45], [-2,0], [-2,0], [0, 30], [0, 30],\
    #       [0, 60, 100], [0, 60, 100],\
    #       [-53,-15,15], [-53,-15,15], [-40,0, 30], [-40,0, 30], [-90,0, 90], [-90,0, 90]]
    qss = [[-60,30], [-54, 54], [-60,0, 60],\
          [-30, 70], [-40, 40], [-35, 35],\
          [0, 150],\
          [-47,170], [-47, 170], [-2,200], [-2,200], [0, 30], [0, 30],\
          [0, 150], [0, 150],\
          [-53,47], [-53,47], [-40, 30], [-40, 30], [-90, 90], [-90, 90]]
    
    all_solutions = []
    for joint in joints:
        with localsolver.LocalSolver() as ls:
            model = ls.get_model()
    
            for i, qs in enumerate(qss):
                minimum = qs[0]
                maximum = qs[-1]
                globals()['x%s' % i] = eval(f'model.float({minimum},{maximum})')
            
            obj_func_instance = partial(objective_function, initial_joint = joint)
            f = model.create_double_blackbox_function(obj_func_instance)
            call = model.call()
            call.add_operand(f)

            for i in range(len(qss)):
                eval(f'call.add_operand(x{i})')

            model.minimize(call)
            model.close()

            ls.get_param().set_time_limit(100)
            ls.solve()
            sol = ls.get_solution()
            
            solution = []
            for i in range(len(qss)):
                solution.append(eval('sol.get_value(x' + str(i) +')'))
            all_solutions.append(solution)
            for i in range(len(qss)):
                eval('print("x{} = {}".format('+ str(i) + ',sol.get_value(x' + str(i) +')))')
            print("obj = {}".format(sol.get_value(call)))

    for solution in all_solutions:
        print(calc_total_reba(solution[0:7], solution[7:]))
        
        
    #np.random.seed(42)
    #print(super_model_test_error())

#    0.5-0.5   0.25-0.75     0.95-0.05
# 6	  [8]         [8]          [8]
# 5	  [9]         [8]          [8]
# 4	  [3]         [3]          [4]
# 6	  [8]         [8]         [10]
# 6	  [6]         [5]          [5]
# 3	  [8]         [5]          [4]
# 6	  [9]         [8]          [9]
# 6	  [8]         [9]          [9]
# 6	  [8]         [8]          [8]
# 9	  [9]         [10]         [9]
# 6	  [8]         [5]          [4]
# 4   [8]         [8]          [9]

