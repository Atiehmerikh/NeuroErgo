import numpy as np


class partial_to_total_REBA:
    def __init__(self,REBA_partial_scores):
        self.REBA_partial_scores = REBA_partial_scores

    def reba_table_a(self):
        return np.array([
            [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
            [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
            [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
        ])

    def reba_table_b(self):
        return np.array([
            [[1, 2, 2], [1, 2, 3]],
            [[1, 2, 3], [2, 3, 4]],
            [[3, 4, 5], [4, 5, 5]],
            [[4, 5, 5], [5, 6, 7]],
            [[6, 7, 8], [7, 8, 8]],
            [[7, 8, 8], [8, 9, 9]],
        ])

    def reba_table_c(self):
        return np.array([
            [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
            [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
            [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
            [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
            [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
            [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
            [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
            [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
            [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
            [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
            [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
        ])

    def find_total_REBA(self):
        # Table A ( Neck X Trunk X Legs)
        table_a = self.reba_table_a()
        # Table B ( UpperArm X LowerArm X Wrist)
        table_b = self.reba_table_b()
        # Table C ( ScoreA X ScoreB)
        table_c = self.reba_table_c()

        # Because of REBA tables misinformation
        if self.REBA_partial_scores[0]>3:
            self.REBA_partial_scores[0] = 3
        if self.REBA_partial_scores[1]>5:
            self.REBA_partial_scores[1] = 5
        if self.REBA_partial_scores[2]>4:
            self.REBA_partial_scores[2] = 4
        if self.REBA_partial_scores[3]>6:
            self.REBA_partial_scores[3] = 6
        if self.REBA_partial_scores[4]>2:
            self.REBA_partial_scores[4] = 2
        if self.REBA_partial_scores[5]>3:
            self.REBA_partial_scores[5] = 3




        posture_score_a = table_a[self.REBA_partial_scores[0]-1][self.REBA_partial_scores[1]-1][self.REBA_partial_scores[2]-1]

        load = 0
        if 11 <= int(load) < 22:
            posture_score_a = posture_score_a + 1
        if 22 <= int(load):
            posture_score_a = posture_score_a + 2

        posture_score_b = table_b[self.REBA_partial_scores[3]-1][self.REBA_partial_scores[4]-1][self.REBA_partial_scores[5]-1]

        # step 11: coupling score
        coupling = 0
        # coupling = input("what is the coupling condition?(good(0) or fair(1) or poor(2) or unacceptable(3)? ")

        posture_score_b = posture_score_b + int(coupling)

        # step 12: look up score in table C
        posture_score_c = table_c[posture_score_a - 1][posture_score_b - 1]

        return posture_score_c
