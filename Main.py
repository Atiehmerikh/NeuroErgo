import REBA.body_part_reba_calculator.Degree_to_REBA.neck_reba_score as REBA_neck


def make_sample():
        # Neck
        neck_flexion_extension_samples = [-60,0,20, 30]
        neck_side_flexion_samples = [-54,0, 54]
        neck_rotation_samples = [-60,0, 60]
        neck_sample = []
        for i in neck_flexion_extension_samples:
            for j in neck_side_flexion_samples:
                for k in neck_rotation_samples:
                    m_neck =REBA_neck.NeckREBA([i,j,k])
                    neck_sample.append([i,j,k,m_neck.neck_reba_score()])

        # trunk
        trunk_flexion_extension_samples = [-30,0,20,60, 70]
        trunk_side_flexion_samples = [-40,0, 40]
        trunk_rotation_samples = [-35,0, 35]
        trunk_sample = []
        for i in trunk_flexion_extension_samples:
            for j in trunk_side_flexion_samples:
                for k in trunk_rotation_samples:
                    trunk_sample.append(
                        [i, j, k])

        # Legs
        legs_flexion_samples = [0,30,60,150]

        # Upper Arm
        upper_arm_flexion_extension_samples = [-47,-20,0,20,45,90, 170]
        shoulder_raise_samples = [0, 30]
        upper_arm_adduction_abduction_samples = [-2,0, 200]
        UA_sample = []
        for i in upper_arm_flexion_extension_samples:
            for j in shoulder_raise_samples:
                for k in upper_arm_adduction_abduction_samples:
                    UA_sample.append(
                        [i, j, k])

        # Lower Arm
        lower_arm_flexion_samples = [0,60,100, 150]

        # Wrist
        wrist_flexion_extension_samples = [-53,-15,15, 47]
        wrist_side_adduction_abduction_samples = [-40,0, 30]
        wrist_rotation_samples = [-90,0, 90]
        wrist_sample = []
        for i in wrist_flexion_extension_samples:
            for j in wrist_side_adduction_abduction_samples:
                for k in wrist_rotation_samples:
                    wrist_sample.append(
                        [i,j, k])



if __name__ == "__main__":
    # for i in range(1, 9, 2):
    #     print(i)

    make_sample()