import numpy as np
from drl_sample_project_python.do_not_touch.contracts import DeepSingleAgentEnv


class EnvGridWorldDeepSingleAgent(DeepSingleAgentEnv):
    def __init__(self, rows_count: int, columns_count: int, max_steps: int, goal: (int, int), lost: (int, int)):
        assert(rows_count * columns_count >= 3)
        self.rows_count = rows_count
        self.columns_count = columns_count
        self.agent_pos = 0
        self.game_over = False
        self.current_score = 0.0
        self.current_step = 0
        self.max_steps = max_steps
        self.goals = [goal[0] * self.columns_count + goal[1], ]
        self.loses = [lost[0] * self.columns_count + lost[1], ]

    def state_description(self) -> np.ndarray:
        board = np.zeros(self.rows_count * self.columns_count)
        board[0] = -1
        board[self.rows_count * self.columns_count - 1] = 10
        board[self.agent_pos] = 1
        return board

    def state_description_length(self) -> int:
        return self.rows_count * self.columns_count

    def max_actions_count(self) -> int:
        return 4

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert(not self.game_over)
        assert(action_id in [0, 1, 2, 3])

        if action_id == 0:
            self.agent_pos -= 1
        elif action_id == 1:
            self.agent_pos += 1
        elif action_id == 2:
            self.agent_pos = (self.agent_pos // self.rows_count - 1) * self.rows_count + self.agent_pos % self.rows_count
        else:
            self.agent_pos = (self.agent_pos // self.rows_count + 1) * self.rows_count + self.agent_pos % self.rows_count

        if self.agent_pos in self.loses:
            self.game_over = True
            self.current_score = -1.0
        elif self.agent_pos in self.goals:
            self.game_over = True
            self.current_score = 1.0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.game_over = True

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        n = np.array([0, 1, 2, 3])  # 0: Left, 1: Right, 2: Up, 3: Down

        if self.agent_pos % self.rows_count == 0:  #Complètement à Gauche
            n = np.delete(n, np.where(n == 0))
        if self.agent_pos % self.rows_count == self.columns_count - 1:  #Complètement à Droite
            n = np.delete(n, np.where(n == 1))
        if self.agent_pos // self.rows_count == 0:  #Complètement en Haut
            n = np.delete(n, np.where(n == 2))
        if self.agent_pos // self.rows_count == self.rows_count - 1:  #Complètement en Bas
            n = np.delete(n, np.where(n == 3))
        return n

    def reset(self):
        self.agent_pos = self.rows_count // 2 * self.columns_count + self.columns_count // 2
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0

    def reset_random(self):
        self.agent_pos = np.random.randint(1, self.rows_count * self.columns_count - 1)
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0

