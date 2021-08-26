import pandas as pd
import numpy as np
import REBA.body_part_reba_calculator.partial_REBA_to_total_REBA as REBA


class Dataset:
    def __init__(self, angle):
        self.angle = angle

    def angle_ranges(self):
        # Neck
        neck_flexion_extension = [-60, 30]
        neck_side_flexion = [-54, 54]
        neck_rotation = [-60, 60]

        # trunk
        trunk_flexion_extension = [-30, 70]
        trunk_side_flexion = [-40, 40]
        trunk_rotation = [-35, 35]

        # Legs
        legs_flexion = [0, 150]

        # Upper Arm
        upper_arm_flexion_extension = [-47, 169.5]
        shoulder_raise = [0, 30]
        upper_arm_adduction_abduction = [-2, 200]

        # Lower Arm
        lower_arm_flexion = [0, 150]

        # Wrist
        wrist_flexion_extension = [-53, 47]
        wrist_side_adduction_abduction = [-40, 30]
        wrist_rotation = [-90, 90]

    def make_sample_from_range(self, body_part_degree_range, sample_rate):
        # here we derive some sample degree from the body part viable range
        init = body_part_degree_range[0]
        final = body_part_degree_range[1]

        sample_degrees = []
        for i in range(init, final, sample_rate):
            sample_degrees.append(i)

        return sample_degrees

    def store_data(self, samples_array, REBA):
        data = {'neck_flexion': [samples_array[0]],
                'neck_side_flexion': [samples_array[1]],
                'neck_rotation': [samples_array[2]],
                'trunk_flexion_extension': [samples_array[3]],
                'trunk_side_flexion': [samples_array[4]],
                'trunk_rotation': [samples_array[5]],
                'legs_flexion': [samples_array[6]],
                'upper_arm_flexion_extension': [samples_array[7]],
                'shoulder_raise': [samples_array[8]],
                'upper_arm_adduction_abduction': [samples_array[9]],
                'lower_arm_flexion': [samples_array[10]],
                'wrist_flexion_extension': [samples_array[11]],
                'wrist_side_adduction_abduction': [samples_array[12]],
                'wrist_rotation': [samples_array[13]],
                'REBA': [REBA]}
        # Convert the dictionary into DataFrame
        df = pd.DataFrame(data)
