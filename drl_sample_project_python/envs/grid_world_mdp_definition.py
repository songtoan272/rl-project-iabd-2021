import numpy as np
from drl_sample_project_python.do_not_touch.contracts import MDPEnv


class EnvGridWorldMDP(MDPEnv):
    def __init__(self, rows_count: int, cols_count: int, goal: (int, int), lost: (int, int)):
        assert(rows_count > 1 and cols_count > 1)
        self.rows_count = rows_count
        self.cols_count = cols_count
        self.cell_count = self.rows_count * self.cols_count
        self.S = np.arange(self.cell_count)
        self.A = np.array([0, 1, 2, 3], dtype=np.int)  # 0: Left, 1: Right, 2: Up, 3: Down
        self.R = np.array([-1, 0, 1], dtype=np.int)
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self.goal = goal
        self.lost = lost

        reward_same_case = 1  # 0: Malus si on essaie une action que l'on ne peut pas, 1: Rien

        for c in range(self.cell_count):
            if c % self.cols_count != 0:
                self.p[c, 0, c - 1, 1] = 1.0
            else:
                self.p[c, 0, c, reward_same_case] = 1.0

            if c % self.cols_count != self.cols_count - 1:
                self.p[c, 1, c + 1, 1] = 1.0
            else:
                self.p[c, 1, c, reward_same_case] = 1.0

            if c >= self.cols_count:
                self.p[c, 2, c - self.cols_count, 1] = 1.0
            else:
                self.p[c, 2, c, reward_same_case] = 1.0

            if c < self.cell_count - self.cols_count:
                self.p[c, 3, c + self.cols_count, 1] = 1.0
            else:
                self.p[c, 3, c, reward_same_case] = 1.0

        if self.goal[0] > 0:
            self.p[(self.goal[0] - 1) * self.cols_count + self.goal[1], 3, self.goal[0] * self.cols_count + self.goal[1], :] = 0.0
            self.p[(self.goal[0] - 1) * self.cols_count + self.goal[1], 3, self.goal[0] * self.cols_count + self.goal[1], 2] = 1.0
        if self.goal[0] < self.rows_count - 1:
            self.p[(self.goal[0] + 1) * self.cols_count + self.goal[1], 2, self.goal[0] * self.cols_count + self.goal[1], :] = .0
            self.p[(self.goal[0] + 1) * self.cols_count + self.goal[1], 2, self.goal[0] * self.cols_count + self.goal[1], 2] = 1.0
        if self.goal[1] > 0:
            self.p[self.goal[0] * self.cols_count + self.goal[1] - 1, 1, self.goal[0] * self.cols_count + self.goal[1], :] = 0.0
            self.p[self.goal[0] * self.cols_count + self.goal[1] - 1, 1, self.goal[0] * self.cols_count + self.goal[1], 2] = 1.0
        if self.goal[0] < self.cols_count - 1:
            self.p[self.goal[0] * self.cols_count + self.goal[1] + 1, 0, self.goal[0] * self.cols_count + self.goal[1], :] = 0.0
            self.p[self.goal[0] * self.cols_count + self.goal[1] + 1, 0, self.goal[0] * self.cols_count + self.goal[1], 2] = 1.0

        if self.lost[0] > 0:
            self.p[(self.lost[0] - 1) * self.cols_count + self.lost[1], 3, self.lost[0] * self.cols_count + self.lost[1], :] = 0.0
            self.p[(self.lost[0] - 1) * self.cols_count + self.lost[1], 3, self.lost[0] * self.cols_count + self.lost[1], 0] = 1.0
        if self.lost[0] < self.rows_count - 1:
            self.p[(self.lost[0] + 1) * self.cols_count + self.lost[1], 2, self.lost[0] * self.cols_count + self.lost[1], :] = 0.0
            self.p[(self.lost[0] + 1) * self.cols_count + self.lost[1], 2, self.lost[0] * self.cols_count + self.lost[1], 0] = 1.0
        if self.lost[1] > 0:
            self.p[self.lost[0] * self.cols_count + self.lost[1] - 1, 1, self.lost[0] * self.cols_count + self.lost[1], :] = 0.0
            self.p[self.lost[0] * self.cols_count + self.lost[1] - 1, 1, self.lost[0] * self.cols_count + self.lost[1], 0] = 1.0
        if self.lost[0] < self.cols_count - 1:
            self.p[self.lost[0] * self.cols_count + self.lost[1] + 1, 0, self.lost[0] * self.cols_count + self.lost[1], :] = 0.0
            self.p[self.lost[0] * self.cols_count + self.lost[1] + 1, 0, self.lost[0] * self.cols_count + self.lost[1], 0] = 1.0

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        if s == self.goal[0] * self.cols_count + self.goal[1] or s == self.lost[0] * self.cols_count + self.lost[1]:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]

    def view_state(self, s: int):
        pass