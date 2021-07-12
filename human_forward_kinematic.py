#   here we translate human joint angles to end effector pose
import math as m
import numpy as np


class forward_kinematics:
    def __init__(self, angles):
        self.hand_sided = "right"
        self.p_toe_right =[10,0,0]
        self.p_toe_left =[-10,0,0]
        self.p_shoulder_centre=[0,0,160]

        self.l_toe_knee=20
        self.l_knee_pelvic=20
        self.l_s_e = 10
        self.l_w_t = 10
        self.l_e_w = 10
        self.l_b_s = 10
        self.spine_length =50
        self.shoulder_length=20


        self.q_trunck_flexion = angles[3]
        self.q_trunk_side = angles[4]
        self.q_trunk_twist = angles[5]

        self.q_knee = angles[6]

        self.q_right_upper_arm_front_adduction = angles[7]
        self.q_right_upper_arm_side_adduction = angles[8]
        self.q_right_shoulder_rise = angles[9]
        self.q_left_upper_arm_front_adduction = angles[10]
        self.q_left_upper_arm_side_adduction = angles[11]
        self.q_left_shoulder_rise = angles[12]

        self.q_right_lower_arm_front_adduction = angles[13]
        self.q_left_lower_arm_front_adduction = angles[14]


        self.q_right_wrist_flex = angles[15]
        self.q_right_wrist_side = angles[16]
        self.q_right_wrist_twist = angles[17]
        self.q_left_wrist_flex = angles[18]
        self.q_left_wrist_side = angles[19]
        self.q_left_wrist_twist = angles[20]

    def Rx(self, theta):
        return np.matrix([[1, 0, 0],
                          [0, m.cos(theta), -m.sin(theta)],
                          [0, m.sin(theta), m.cos(theta)]])

    def Ry(self,theta):
        return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])

    def Rz(self,theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])

    def legs_FK(self):
        # when knee bends the legs move in z-y plane

        p_x = (self.p_toe_right[0] + self.p_toe_left[0]) / 2
        p_y = self.p_toe_right[1] + self.l_knee_pelvic * m.sin(self.q_knee)
        p_z = self.p_toe_right[2] + self.l_toe_knee + self.l_knee_pelvic * m.cos(self.q_knee)

        # position of Hip centre:
        return (p_x, p_y, p_z)

    def body_FK(self,hip_centre):
        # p_shoulder_centre = np.array([[1], [0], [0]])
        z_shoulder_centre = hip_centre[2] + self.spine_length
        shoulder = [self.shoulder_length,0,z_shoulder_centre]

        R = self.Rx(self.q_trunck_flexion) * self.Ry(self.q_trunk_side) * self.Rz(self.q_trunk_twist)
        R = (np.round(R, decimals=2))

        p_shoulder_new = R * shoulder
        # p_shoulder_new = R * self.p_shoulder

        # print(np.round(p_shoulder_centre_new, decimals=2))
        return p_shoulder_new

    def arm_FK(self,shoulder_position,hand_sided):
        if hand_sided=="right":
            # assumed thetas are given in gradient
            theta_1 =self.q_right_shoulder_rise
            theta_2 = self.q_right_lower_arm_front_adduction
            theta_3 = self.q_right_upper_arm_side_adduction
            theta_4 = self.q_right_lower_arm_front_adduction
            theta_5 = self.q_right_wrist_side
            theta_6 = self.q_right_wrist_flex
            theta_7 = self.q_right_wrist_twist
        else:
            theta_1 = self.q_left_shoulder_rise
            theta_2 = self.q_left_lower_arm_front_adduction
            theta_3 = self.q_left_upper_arm_side_adduction
            theta_4 = self.q_left_lower_arm_front_adduction
            theta_5 = self.q_left_wrist_side
            theta_6 = self.q_left_wrist_flex
            theta_7 = self.q_left_wrist_twist

        a3 = 20 # centimeter
        a4 = 25

        A_1 = np.matrix([[m.cos(theta_1),-m.sin(theta_1),0,0],[0,0,-1,0],[m.sin(theta_1),m.cos(theta_1),0,0],[0,0,0,1]])
        A_2 = np.matrix([[m.cos(theta_2),-m.sin(theta_2),0,0],[0,0,-1,0],[m.sin(theta_2),m.cos(theta_2),0,0],[0,0,0,1]])
        A_3 = np.matrix([[m.cos(theta_3),-m.sin(theta_3),0,a3],[0,0,1,0],[m.sin(theta_3),-m.cos(theta_3),0,0],[0,0,0,1]])
        A_4 = np.matrix([[m.cos(theta_4),-m.sin(theta_4),0,a4],[0,0,-1,0],[m.sin(theta_4),m.cos(theta_4),0,0],[0,0,0,1]])
        A_5 = np.matrix([[m.cos(theta_5),-m.sin(theta_5),0,0],[0,0,1,0],[m.sin(theta_5),-m.cos(theta_5),0,0],[0,0,0,1]])
        A_6 = np.matrix([[m.cos(theta_6),-m.sin(theta_6),0,0],[0,0,-1,0],[m.sin(theta_6),m.cos(theta_6),0,0],[0,0,0,1]])
        A_7 = np.matrix([[m.cos(theta_7),-m.sin(theta_7),0,0],[0,0,1,0],[-m.sin(theta_7),-m.cos(theta_7),0,0],[0,0,0,1]])
        T = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(A_1,A_2),A_3),A_4),A_5),A_6),A_7)
        print(T)
        # position of ee relative to shoulder
        ee_position = T[:,3]

        p_x = ee_position[0]+shoulder_position[0]
        p_y = ee_position[1]+shoulder_position[1]
        p_z = ee_position[2]+shoulder_position[2]

        # position relative to middle of toes

        return [p_x, p_y, p_z]

    def total_human_body_FK(self):
        hip_centre = self.legs_FK()
        shoulder_new = self.body_FK(hip_centre)
        ee_position = self.arm_FK(shoulder_new,self.hand_sided)
        return ee_position
        # shoulder new position
