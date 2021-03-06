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

from tqdm import tqdm

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
def objective_function(context, initial_joint, task_w, posture_w):
    initial = initial_joint
    angles=[]
    for i in range(21):
        #angles.append(context.get(i))
        angles.append(context[i])
    num_of_data = 1
    data = {
        'neck_model_input': tf.Variable([angles[0:3]], dtype=tf.float32),
        'trunk_model_input': tf.Variable([angles[3:6]], dtype=tf.float32),
        'leg_model_input': tf.Variable([[angles[6]]], dtype=tf.float32),
        'upper_arm_model_input': tf.Variable([angles[7:13]], dtype=tf.float32),
        'lower_arm_model_input': tf.Variable([angles[13:15]], dtype=tf.float32),
        'wrist_model_input': tf.Variable([angles[15:21]], dtype=tf.float32)
    }
    
    with tf.GradientTape() as tape:
        pred = super_model_for_optimization(data)
    grad = tape.gradient(pred, data)
    
    c_posuture_derivitive = list(np.array(grad['neck_model_input'])[0]) + list(np.array(grad['trunk_model_input'])[0]) + list(np.array(grad['leg_model_input'])[0]) +\
               list(np.array(grad['upper_arm_model_input'])[0]) + list(np.array(grad['lower_arm_model_input'])[0]) + list(np.array(grad['wrist_model_input'])[0])

    # fk = forward_kinematics(angles)
    # end_effector_position = fk.total_human_body_FK()
    c_task_derivitive = [2*(context[i] - initial[i])  for i in range(len(initial))]
    gradient_norm = np.linalg.norm([(posture_w * c_posuture_derivitive[i] + task_w * c_task_derivitive[i]) for i in range(len(initial))])
    return gradient_norm


def real_data_provider():
    joints_set = [51.5360792322674,50.7433097956106,0,
                  -3.26763048572082,0.332623327271285,0,
                  140.805032574017,31.7883306170516,23.3645749994502,
                  -73.8591253414813,74.9892660368735,13.238204725126,
                  17.4576031237221,48.2409115317895,75.1078730717884,
                  19.4383706628683,7.69281245155988,-17.8183364120746,
                  1.14599199838859,-17.8183364120746,1.14599199838859]
    
    return joints_set
    



if __name__ == "__main__": 
    
    super_model_for_optimization = load_model('./data/super_model_DNN.model')
    joints = real_data_provider()
    qss = [[-60,30], [-54, 54], [-60,0, 60],\
          [-30, 70], [-40, 40], [-35, 35],\
          [0, 150],\
          [-47,170], [-47, 170], [-2,200], [-2,200], [0, 30], [0, 30],\
          [0, 150], [0, 150],\
          [-53,47], [-53,47], [-40, 30], [-40, 30], [-90, 90], [-90, 90]]

    
    ###    A balck-box optimization method ###
    postures_ws = [0.25]
    
    for pw in postures_ws:
        print("------------------------pw:" + str(pw))
        all_solutions = []
        # print(np.array(joints).shape)
        for joint in tqdm(joints):
            with localsolver.LocalSolver() as ls:
                model = ls.get_model()
        
                for i, qs in enumerate(qss):
                    minimum = qs[0]
                    maximum = qs[-1]
                    globals()['x%s' % i] = eval(f'model.float({minimum},{maximum})')
                
                obj_func_instance = partial(objective_function, initial_joint = joint, task_w = 1-pw, posture_w = pw)
                f = model.create_double_blackbox_function(obj_func_instance)
                call = model.call()
                call.add_operand(f)

                for i in range(len(qss)):
                    eval(f'call.add_operand(x{i})')

                model.minimize(call)
                model.close()

                ls.get_param().set_time_limit(50)
                ls.get_param().set_nb_threads(8)
                ls.get_param().set_verbosity(0)
                ls.solve()
                sol = ls.get_solution()
                
                solution = []
                for i in range(len(qss)):
                    solution.append(eval('sol.get_value(x' + str(i) +')'))
                all_solutions.append(solution)
                for i in range(len(qss)):
                    eval('print("x{} = {}".format('+ str(i) + ',sol.get_value(x' + str(i) +')))')
                print("obj = {}".format(sol.get_value(call)))

    #     with open("./data/solutions_" + str(pw) + ".txt", "w") as solution_file:
    #         for solution in all_solutions:
    #             r = calc_total_reba(solution[0:7], solution[7:])[0]
    #             solution_file.write(str(r))
    #             solution_file.write('\n')
        

