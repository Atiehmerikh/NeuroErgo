import REBA.body_part_reba_calculator.Degree_to_REBA.neck_reba_score as REBA_neck
import REBA.body_part_reba_calculator.Degree_to_REBA.trunk_reba_score as REBA_trunk
import REBA.body_part_reba_calculator.Degree_to_REBA.leg_reba_score as REBA_leg
import REBA.body_part_reba_calculator.Degree_to_REBA.upperarm_reba_score as REBA_UA
import REBA.body_part_reba_calculator.Degree_to_REBA.lowerarm_reba_score as REBA_LA
import REBA.body_part_reba_calculator.Degree_to_REBA.wrist_reba_score as REBA_wrist
import REBA.body_part_reba_calculator.partial_REBA_to_total_REBA as REBA


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
                    m_trunk =REBA_trunk.TrunkREBA([i,j,k])
                    trunk_sample.append([i,j,k,m_trunk.trunk_reba_score()])

        # Legs
        legs_flexion_samples = [0,30,60,150]
        leg_sample = []
        for i in legs_flexion_samples:
            m_leg = REBA_leg.LegREBA([i,i])
            leg_sample.append([i, m_leg.leg_reba_score()])

        # Upper Arm
        right_upper_arm_flexion_extension_samples = [-47,-20,0,20,45,90, 170]
        left_upper_arm_flexion_extension_samples = [-47, -20, 0, 20, 45, 90, 170]
        right_upper_arm_adduction_abduction_samples = [-2,0, 200]
        left_upper_arm_adduction_abduction_samples = [-2,0, 200]
        right_shoulder_raise_samples = [0, 30]
        left_shoulder_raise_samples = [0, 30]
        UA_sample = []
        for i in right_upper_arm_flexion_extension_samples:
            for j in left_upper_arm_flexion_extension_samples:
                for k in right_upper_arm_adduction_abduction_samples:
                    for l in left_upper_arm_adduction_abduction_samples:
                        for m in right_shoulder_raise_samples:
                            for n in left_shoulder_raise_samples:
                                m_UA = REBA_UA.UAREBA([i, j,k,l,m,n])
                                UA_sample.append([i, j, k,l,m,n,m_UA.upper_arm_reba_score()])

        # Lower Arm
        right_lower_arm_flexion_samples = [0,60,100, 150]
        left_lower_arm_flexion_samples = [0,60,100, 150]
        LA_sample =[]
        for i in right_lower_arm_flexion_samples:
            for j in left_lower_arm_flexion_samples:
                m_LA = REBA_LA.LAREBA([i,j])
                LA_sample.append([i,j,m_LA.lower_arm_score()])

        # Wrist
        right_wrist_flexion_extension_samples = [-53,-15,15, 47]
        left_wrist_flexion_extension_samples = [-53,-15,15, 47]
        right_wrist_side_adduction_abduction_samples = [-40,0, 30]
        left_wrist_side_adduction_abduction_samples = [-40,0, 30]
        right_wrist_rotation_samples = [-90,0, 90]
        left_wrist_rotation_samples = [-90,0, 90]
        wrist_sample = []
        for i in right_wrist_flexion_extension_samples:
            for j in left_wrist_flexion_extension_samples:
                for k in right_wrist_side_adduction_abduction_samples:
                    for l in left_wrist_side_adduction_abduction_samples:
                        for m in right_wrist_rotation_samples:
                            for n in left_wrist_rotation_samples:
                                m_wrist = REBA_wrist.WristREBA([i, j,k,l,m,n])
                                wrist_sample.append([i, j,k,l,m,n, m_wrist.wrist_reba_score()])

        # m_REBA = REBA.partial_to_total_REBA([neck_sample[0][len(neck_sample[0])-1],
        #                                      trunk_sample[0][len(trunk_sample[0])-1],
        #                                      leg_sample[0][len(leg_sample[0])-1],
        #                                      UA_sample[0][len(UA_sample[0])-1],
        #                                      LA_sample[0][len(LA_sample[0])-1],
        #                                      wrist_sample[0][len(wrist_sample[0])-1]])
        # print(m_REBA.find_total_REBA())

if __name__ == "__main__":
    # for i in range(1, 9, 2):
    #     print(i)

    make_sample()