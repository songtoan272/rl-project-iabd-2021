import random
import numpy as np
from drl_sample_project_python.do_not_touch.contracts import SingleAgentEnv


class EnvTicTacToeSingleAgent(SingleAgentEnv):
    def __init__(self, max_steps: int = 9):
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

    def from_state(self, state_id):
        state_id = str(state_id)
        for b in range(len(state_id)):
            if state_id[b] == '1':
                self.board[b] = 0
            elif state_id[b] == '2':
                self.board[b] = 1
            else:
                self.board[b] = 10
                
    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert(not self.game_over)
        assert(action_id in [0, 1, 2, 3, 4, 5, 6, 7, 8])

        # Agent plays
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

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.game_over = True

        # Concurrent plays randomly
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

    def view(self):
        res = "-" * 10
        board_str = ['X' if b == 1 else ('O' if b == 10 else '_') for b in self.board]
        for idx, b in enumerate(board_str):
            if idx % 3 == 0:
                res += '\n'
            res += b + ' '*3
        res += '\n' + '-' * 10
        print(res)


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
        self.reset()
        c = [0, 1, 10]
        nb_steps = np.random.default_rng().integers(0, self.max_steps, size=1)[0]
        for s in range(0, nb_steps, 2):
            if not self.game_over:
                self.act_with_action_id(random.choice(self.available_actions_ids()))
    
    def gen_episode(self, pi, returns, q):
        S = []
        A = []
        R = []
        while not self.is_game_over():
            s = self.state_id()
            S.append(s)
            available_actions = self.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = 0

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]

            A.append(chosen_action)
            old_score = self.score()
            self.act_with_action_id(chosen_action)
            r = self.score() - old_score
            R.append(r)
        return S, A, R


if __name__ == "__main__":
    env = EnvTicTacToeSingleAgent()
    env.reset_random()
    env.view()