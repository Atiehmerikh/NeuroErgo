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
from itertools import product
#from sklearn.model_selection import train_test_split
#import tensorflow
from tensorflow.keras.layers import Dense, Dropout, Concatenate, concatenate
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Adadelta, Adagrad, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tqdm import tqdm
import math
import gzip
import shutil
import os
import time

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
    activation = 'relu'
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


# print(neuro_REBA([-50, -50, 10], [-10, -20, 30], [20], [10, 10, 10, 10, 10, 10], [50, 50], [10, 10, 10, 10, 10, 10]))

# m  = REBA.partial_to_total_REBA([REBA_neck.NeckREBA([-50, -50, 10]).neck_reba_score(),
#                             REBA_trunk.TrunkREBA([-10, -20, 30]).trunk_reba_score(),
#                             REBA_leg.LegREBA([20,20]).leg_reba_score(),
#                             REBA_UA.UAREBA([10, 10, 10, 10, 10, 10]).upper_arm_reba_score(),
#                             REBA_LA.LAREBA([50,50]).lower_arm_score(),
#                             REBA_wrist.WristREBA([10, 10, 10, 10, 10, 10]).wrist_reba_score()]).find_total_REBA()

#print(m)


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

        if counter > 2830:
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
        else:
            counter += 1
            continue    

        

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


def super_model_test():
    
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


np.random.seed(42)
print(super_model_test())
#generate_super_model_training_data()


# for i in product([-60,0,20, 30], [-54,0, 54], [-60,0, 60],[-30,0,20,60, 70], [-40,0, 40], [-35,0, 35], [0,30,60,150]):
#     print(i)
#     break

# def calc_stuff(x):
#     return list(x)

# pool = mp.Pool(4)
# out1 = pool.map(calc_stuff, product([-60,0,20, 30], [-54,0, 54], [-60,0, 60],[-30,0,20,60, 70], [-40,0, 40], [-35,0, 35], [0,30,60,150]))
# print(type(out1))
# neck_training_model()
# print(neck_model_test())
# trunk_training_model()
# print(trunk_model_test())
# leg_training_model()
# print(leg_model_test())
# upper_arm_training_model()
# print(upper_arm_model_test())
# lower_arm_training_model()
# print(lower_arm_model_test())
# wrist_training_model()
# print(wrist_model_test())
# total_reba_from_partial_learning_model()
# print(total_reba_from_partial_model_test())
