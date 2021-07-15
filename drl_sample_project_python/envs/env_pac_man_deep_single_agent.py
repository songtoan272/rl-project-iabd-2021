import copy

import numpy as np
from drl_sample_project_python.do_not_touch.contracts import DeepSingleAgentEnv


class EnvPacManDeepSingleAgent(DeepSingleAgentEnv):
    def __init__(self, max_steps: int, level_file: str):
        self.filename = level_file
        self.max_steps = max_steps
        self.board, self.rows, self.cols, self.pacgum_count = self.init_board()
        self.initial_board = copy.deepcopy(self.board)
        self.reset()

    def state_description(self) -> np.ndarray:
        r = np.zeros(self.state_description_length())
        for n, b in enumerate(self.board):
            if b.isnumeric():
                r[n] = int(b)
            else:
                r[n] = ord(b)
        return r

    def state_description_length(self) -> int:
        return len(self.board)

    def max_actions_count(self) -> int:
        return 4

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.game_over)
        assert (action_id in [0, 1, 2, 3])

        old_pos = self.agent_pos

        if action_id == 0:
            if self.board[self.agent_pos - 1] == '1':
                self.current_pacgum += 1
                self.current_score += 1
                self.board[self.agent_pos - 1] = '2'
                self.agent_pos -= 1
            elif not self.board[self.agent_pos - 1].isnumeric():
                exit = self.find_wrapper_exit(self.board[self.agent_pos - 1], self.agent_pos - 1)
                self.board[exit - 1] = '2'
        elif action_id == 1:
            if self.board[self.agent_pos + 1] == '1':
                self.current_pacgum += 1
                self.current_score += 1
                self.board[self.agent_pos + 1] = '2'
                self.agent_pos += 1
            elif not self.board[self.agent_pos + 1].isnumeric():
                exit = self.find_wrapper_exit(self.board[self.agent_pos + 1], self.agent_pos + 1)
                self.board[exit + 1] = '2'
        elif action_id == 2:
            if self.board[self.agent_pos - self.cols] == '1':
                self.current_pacgum += 1
                self.current_score += 1
                self.board[self.agent_pos - self.cols] = '2'
                self.agent_pos -= self.cols
            elif not self.board[self.agent_pos - self.cols].isnumeric():
                exit = self.find_wrapper_exit(self.board[self.agent_pos - self.cols], self.agent_pos - self.cols)
                self.board[exit - self.cols] = '2'
        else:
            if self.board[self.agent_pos + self.cols] == '1':
                self.current_pacgum += 1
                self.current_score += 1
                self.board[self.agent_pos + self.cols] = '2'
                self.agent_pos += self.cols
            elif not self.board[self.agent_pos + self.cols].isnumeric():
                exit = self.find_wrapper_exit(self.board[self.agent_pos + self.cols], self.agent_pos + self.cols)
                self.board[exit + self.cols] = '2'

        self.board[old_pos] = '0'

        self.current_step += 1

        if self.current_pacgum == self.pacgum_count:
            self.game_over = True
            self.current_score += 1000

        if self.current_step == self.max_steps:
            self.game_over = True

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        n = np.array([0, 1, 2, 3])  # 0: Left, 1: Right, 2: Up, 3: Down

        if self.agent_pos % self.cols == 0:  # Complètement à Gauche
            n = np.delete(n, np.where(n == 0))
        elif self.board[self.agent_pos - 1] == '7':
            n = np.delete(n, np.where(n == 0))

        if self.agent_pos % self.cols == self.cols - 1:  # Complètement à Droite
            n = np.delete(n, np.where(n == 1))
        elif self.board[self.agent_pos + 1] == '7':
            n = np.delete(n, np.where(n == 1))

        if self.agent_pos // self.rows == 0:  # Complètement en Haut
            n = np.delete(n, np.where(n == 2))
        elif self.board[self.agent_pos - self.cols] == '7':
            n = np.delete(n, np.where(n == 2))

        if self.agent_pos // self.rows == self.cols - 1:  # Complètement en Bas
            n = np.delete(n, np.where(n == 3))
        elif self.board[self.agent_pos + self.cols] == '7':
            n = np.delete(n, np.where(n == 3))

        return n

    def reset(self):
        self.board = copy.deepcopy(self.initial_board)
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0
        self.current_pacgum = 0

        n = 0
        while n < self.rows * self.cols and self.board[n] != '2':
            n += 1
        if n == self.rows * self.cols:
            self.game_over = True
            return
        self.agent_pos = n

    def reset_random(self):
        self.reset()

    def set_state(self, state):
        self.board = state

    def init_board(self):
        board = []
        f = open(self.filename, 'r')
        line = f.readline().replace('\n', '')
        cols = len(line)
        rows = 0
        nb_pacgum = 0
        while line:
            row = []
            for n, l in enumerate(line):
                row.append(l)
                if l == '1':
                    nb_pacgum += 1
            board.append(row)
            rows += 1
            line = f.readline().replace('\n', '')
        f.close()
        return np.array(board).flatten(), rows, cols, nb_pacgum

    def find_wrapper_exit(self, w: str, n: int) -> int:
        x = 0
        find = False
        while x < self.rows * self.cols and not find:
            if self.board[x] == w and n != x:
                find = True
            else:
                x += 1
        if find:
            return x
        else:
            return -1
