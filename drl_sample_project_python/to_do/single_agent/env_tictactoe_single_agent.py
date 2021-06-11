import random
import numpy as np
from drl_sample_project_python.do_not_touch.contracts import SingleAgentEnv


class EnvTicTacToeSingleAgent(SingleAgentEnv):
    def __init__(self, max_steps: int):
        assert(max_steps > 0)
        self.game_over = False
        self.current_score = 0.0
        self.current_step = 0
        self.max_steps = max_steps
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def state_id(self) -> int:
        num = ''
        for b in self.board:
            if b == 0:
                num += '1'
            elif b == 1:
                num += '2'
            else:
                num += '3'

        return int(num)

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert(not self.game_over)
        assert(action_id in [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.board[action_id] = 1

        r0 = self.board[0] + self.board[1] + self.board[2]
        r1 = self.board[3] + self.board[4] + self.board[5]
        r2 = self.board[6] + self.board[7] + self.board[8]

        c0 = self.board[0] + self.board[3] + self.board[6]
        c1 = self.board[1] + self.board[4] + self.board[7]
        c2 = self.board[2] + self.board[5] + self.board[8]

        d0 = self.board[0] + self.board[4] + self.board[8]
        d1 = self.board[2] + self.board[4] + self.board[6]

        if r0 == 3 or r1 == 3 or r2 == 3 or c0 == 3 or c1 == 3 or c2 == 3 or d0 == 3 or d1 == 3:
            self.game_over = True
            self.current_score = 1.0

        board_full = True
        for b in self.board:
            if b == 0:
                board_full = False
                break

        if board_full:
            self.game_over = True

        if not self.game_over:
            a = random.choice(self.available_actions_ids())

            self.board[a] = 10

            r0 = self.board[0] + self.board[1] + self.board[2]
            r1 = self.board[3] + self.board[4] + self.board[5]
            r2 = self.board[6] + self.board[7] + self.board[8]

            c0 = self.board[0] + self.board[3] + self.board[6]
            c1 = self.board[1] + self.board[4] + self.board[7]
            c2 = self.board[2] + self.board[5] + self.board[8]

            d0 = self.board[0] + self.board[4] + self.board[8]
            d1 = self.board[2] + self.board[4] + self.board[6]

            if r0 == 30 or r1 == 30 or r2 == 30 or c0 == 30 or c1 == 30 or c2 == 30 or d0 == 30 or d1 == 30:
                self.game_over = True
                self.current_score = -1.0

            board_full = True
            for b in self.board:
                if b == 0:
                    board_full = False
                    break

            if board_full:
                self.game_over = True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.game_over = True

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int8)

        # Jouer sur une case :
        # 0  1  2
        # 3  4  5
        # 6  7  8
        n = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)

        i = 0
        for b in self.board:
            if b != 0:
                n = np.delete(n, i)
            else:
                i += 1

        return n

    def reset(self):
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def reset_random(self):
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0
        c = [0, 1, 10]
        self.board = [random.choice(c), random.choice(c), random.choice(c),
                      random.choice(c), random.choice(c), random.choice(c),
                      random.choice(c), random.choice(c), random.choice(c)]

