import copy

import numpy as np
from drl_sample_project_python.do_not_touch.contracts import DeepSingleAgentEnv


class EnvPacManDeepSingleAgent(DeepSingleAgentEnv):
    def __init__(self, max_steps: int, level_file: str):
        self.filename = level_file
        self.max_steps = max_steps
        self.board, self.rows, self.cols, self.pacgum_count, self.ghosts = self.init_board()
        self.initial_board = copy.deepcopy(self.board)
        self.initial_ghosts = copy.deepcopy(self.ghosts)
        self.current_step = 0
        self.current_score = 0.0
        self.current_pacgum = 0
        self.game_over = False
        self.agent_pos = 0
        self.reset()

    def state_description(self) -> np.ndarray:
        r = np.zeros(self.state_description_length())
        i = 0
        for n, b in enumerate(self.board):
            if b.isnumeric():
                r[n] = int(b)
            else:
                r[n] = ord(b)
            i = n
        for m, g in enumerate(self.ghosts):
            r[i + m] = g
        return r

    def state_description_length(self) -> int:
        return len(self.board) + len(self.ghosts)

    def max_actions_count(self) -> int:
        return 4

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.game_over)
        assert (action_id in [0, 1, 2, 3])

        old_pos = self.agent_pos
        new_pos = old_pos

        if action_id == 0:
            new_pos = self.agent_pos - 1
            if not self.board[self.agent_pos - 1].isnumeric():
                new_pos = self.find_wrapper_exit(self.board[self.agent_pos - 1], self.agent_pos - 1) - 1
        elif action_id == 1:
            new_pos = self.agent_pos + 1
            if not self.board[self.agent_pos + 1].isnumeric():
                new_pos = self.find_wrapper_exit(self.board[self.agent_pos + 1], self.agent_pos + 1) + 1
        elif action_id == 2:
            new_pos = self.agent_pos - self.cols
            if not self.board[self.agent_pos - self.cols].isnumeric():
                new_pos = self.find_wrapper_exit(self.board[self.agent_pos - self.cols], self.agent_pos - self.cols) - self.cols
        elif action_id == 3:
            new_pos = self.agent_pos + self.cols
            if not self.board[self.agent_pos + self.cols].isnumeric():
                new_pos = self.find_wrapper_exit(self.board[self.agent_pos + self.cols], self.agent_pos + self.cols) + self.cols

        if new_pos in self.ghosts:
            self.current_score -= 1000
            self.game_over = True
            return

        if self.board[new_pos] == '1':
            self.current_score += 1
            self.current_pacgum += 1

        self.board[old_pos] = '0'
        self.board[new_pos] = '2'
        self.agent_pos = new_pos
        self.current_step += 1

        if self.current_pacgum == self.pacgum_count:
            self.game_over = True
            self.current_score += 1000

        if self.current_step == self.max_steps:
            self.game_over = True

        # Ghosts
        for x in range(len(self.ghosts)):
            g = self.ghosts[x]
            if g == -100:
                continue
            actions = self.get_action_available(g)
            a = np.random.choice(actions, 1, False)
            ghost_pos = 0
            if a == 0:
                ghost_pos = g - 1
                if not self.board[g - 1].isnumeric():
                    ghost_pos = self.find_wrapper_exit(self.board[g - 1], g - 1) - 1
            elif a == 1:
                ghost_pos = g + 1
                if not self.board[g + 1].isnumeric():
                    ghost_pos = self.find_wrapper_exit(self.board[g + 1], g + 1) + 1
            elif a == 2:
                ghost_pos = g - self.cols
                if not self.board[g - self.cols].isnumeric():
                    ghost_pos = self.find_wrapper_exit(self.board[g - self.cols], g - self.cols) - self.cols
            elif a == 3:
                ghost_pos = g + self.cols
                if not self.board[g + self.cols].isnumeric():
                    ghost_pos = self.find_wrapper_exit(self.board[g + self.cols], g + self.cols) + self.cols
            self.ghosts[x] = ghost_pos
            if self.board[ghost_pos] == '2':
                self.game_over = True
                self.current_score -= 1000
                break

    def get_action_available(self, x: int) -> np.ndarray:
        n = np.array([0, 1, 2, 3])  # 0: Left, 1: Right, 2: Up, 3: Down

        if x % self.cols == 0:  # Complètement à Gauche
            n = np.delete(n, np.where(n == 0))
        elif self.board[x - 1] == '7':
            n = np.delete(n, np.where(n == 0))

        if x % self.cols == self.cols - 1:  # Complètement à Droite
            n = np.delete(n, np.where(n == 1))
        elif self.board[x + 1] == '7':
            n = np.delete(n, np.where(n == 1))

        if x // self.rows == 0:  # Complètement en Haut
            n = np.delete(n, np.where(n == 2))
        elif self.board[x - self.cols] == '7':
            n = np.delete(n, np.where(n == 2))

        if x // self.rows == self.cols - 1:  # Complètement en Bas
            n = np.delete(n, np.where(n == 3))
        elif self.board[x + self.cols] == '7':
            n = np.delete(n, np.where(n == 3))

        return n

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        return self.get_action_available(self.agent_pos)

    def reset(self):
        self.board = copy.deepcopy(self.initial_board)
        self.ghosts = copy.deepcopy(self.initial_ghosts)
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

    def move_pac_man(self, x: int):
        self.board[self.agent_pos] = '0'
        self.agent_pos = x
        self.board[x] = '2'

    def get_ghosts(self):
        return self.ghosts

    def reset_random(self):
        self.reset()

    def set_state(self, state):
        self.board = state

    def init_board(self):
        board = []
        ghosts = [-100, -100, -100, -100]
        f = open(self.filename, 'r')
        line = f.readline().replace('\n', '')
        cols = len(line)
        rows = 0
        nb_pacgum = 0
        while line:
            row = []
            for n, l in enumerate(line):
                if l in ('3', '4', '5', '6'):
                    row.append('0')
                    if l == '3':
                        ghosts[0] = n + rows * cols
                    elif l == '4':
                        ghosts[1] = n + rows * cols
                    elif l == '5':
                        ghosts[2] = n + rows * cols
                    elif l == '6':
                        ghosts[3] = n + rows * cols
                else:
                    row.append(l)
                    if l == '1':
                        nb_pacgum += 1
            board.append(row)
            rows += 1
            line = f.readline().replace('\n', '')
        f.close()
        return np.array(board).flatten(), rows, cols, nb_pacgum, ghosts

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
