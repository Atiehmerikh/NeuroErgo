import REBA.body_part_reba_calculator.Degree_to_REBA.neck_reba_score as REBA_neck
import REBA.body_part_reba_calculator.Degree_to_REBA.trunk_reba_score as REBA_trunk
import REBA.body_part_reba_calculator.Degree_to_REBA.leg_reba_score as REBA_leg
import REBA.body_part_reba_calculator.Degree_to_REBA.upperarm_reba_score as REBA_UA
import REBA.body_part_reba_calculator.Degree_to_REBA.lowerarm_reba_score as REBA_LA
import REBA.body_part_reba_calculator.Degree_to_REBA.wrist_reba_score as REBA_wrist
import REBA.body_part_reba_calculator.partial_REBA_to_total_REBA as REBA

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Adadelta, Adagrad, SGD
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tqdm import tqdm
import math

# Neck
def neck_ranges():
    neck_flexion_extension_samples = list(range(-60, 31))
    neck_side_flexion_samples = list(range(-54, 55))
    neck_rotation_samples = list(range(-60, 61))

    return neck_flexion_extension_samples, neck_side_flexion_samples, neck_rotation_samples

def neck_learning_model():

    activation = 'tanh'
    model = Sequential()
    model.add(Dense(3, input_dim=3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(1))
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
    model.add(Dense(3, input_dim=3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    # model.add(Dense(6, activation=activation))
    # model.add(Dense(9, activation=activation))
    # model.add(Dense(12, activation=activation))
    # model.add(Dense(9, activation=activation))
    # model.add(Dense(6, activation=activation))
    # model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(3, activation=activation))
    model.add(Dense(1))
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
    model.add(Dense(1, input_dim=1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1))
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
    right_upper_arm_flexion_extension_samples = [-47, -46] + [*range(-45, 171, 5)]
    left_upper_arm_flexion_extension_samples = [-47, -46] + [*range(-45, 171, 5)]
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
    model.add(Dense(6, input_dim=6, activation=activation))
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
    model.add(Dense(2, input_dim=2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1))
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
# LA_sample =[]
# for i in right_lower_arm_flexion_samples:
#     for j in left_lower_arm_flexion_samples:
#         m_LA = REBA_LA.LAREBA([i,j])
#         LA_sample.append([i,j,m_LA.lower_arm_score()])

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
    model.add(Dense(6, input_dim=6, activation=activation))
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
# m_REBA = REBA.partial_to_total_REBA([neck_sample[0][len(neck_sample[0])-1],     1,2,3
#                                      trunk_sample[0][len(trunk_sample[0])-1],   1,2,3,4,5
#                                      leg_sample[0][len(leg_sample[0])-1],       1 -- 4
#                                      UA_sample[0][len(UA_sample[0])-1],         1 -- 6
#                                      LA_sample[0][len(LA_sample[0])-1],         1,2
#                                      wrist_sample[0][len(wrist_sample[0])-1]])  1,2,3
# print(m_REBA.find_total_REBA())






np.random.seed(42)
#neck_learning_model()
#print(neck_model_test())
#trunk_learning_model()
#print(trunk_model_test())
#leg_learning_model()
#print(leg_model_test())
#upper_arm_learning_model()
#print(upper_arm_model_test())
#lower_arm_learning_model()
#print(lower_arm_model_test())
#wrist_learning_model()
#print(wrist_model_test())
total_reba_from_partial_learning_model()
print(total_reba_from_partial_model_test())
