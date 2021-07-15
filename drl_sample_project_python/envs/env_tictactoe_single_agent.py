import operator
import random
import numpy as np
from drl_sample_project_python.do_not_touch.contracts import SingleAgentEnv


def print_samples(nb_samples, pi, q):
    samples = np.random.choice(list(pi.keys()), size=nb_samples, replace=False)
    for n in samples:
        char = ' XO'
        m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        i = 0
        for c in str(n):
            m[i // 3][i % 3] = char[int(c) - 1]
            i += 1

        print(n, ' :\n', pi[n], '\n', q[n])
        for i in m:
            for j in i:
                print(j, end=' ')
            print(end='\n')
        print('\n\n')


class EnvTicTacToeSingleAgent(SingleAgentEnv):
    def __init__(self, max_steps: int, first_player: int = 0, second_pi=None):
        assert(max_steps > 0)
        self.max_steps = max_steps
        self.is_first_player = first_player  # 0: random, 1: always first, 2: always second
        self.second_pi = second_pi
        self.reset()

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

        board_full = True
        for b in self.board:
            if b == 0:
                board_full = False

        if r0 == 3 or r1 == 3 or r2 == 3 or c0 == 3 or c1 == 3 or c2 == 3 or d0 == 3 or d1 == 3:
            self.game_over = True
            self.current_score += 10.0
            return

        """
        if r0 == 21 or r1 == 21 or r2 == 21 or c0 == 21 or c1 == 21 or c2 == 21 or d0 == 21 or d1 == 21:
            self.current_score += 1.0
        """

        if board_full:
            self.current_score = -1.0
            self.game_over = True
            return

        #Tour adverse

        a = random.choice(self.available_actions_ids())
        if self.second_pi is not None:
            if self.state_id() in self.second_pi and random.random() < 0.9 and not self.always_random:
                actions = self.second_pi[self.state_id()]
                a = max(actions.items(), key=operator.itemgetter(1))[0] if type(actions) is dict else actions
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
            self.current_score = -5.0
            return

        board_full = True
        for b in self.board:
            if b == 0:
                board_full = False

        if board_full:
            self.game_over = True
            return

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
            return np.array([], dtype=np.int)

        # Jouer sur une case :
        # 0  1  2
        # 3  4  5
        # 6  7  8
        n = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int)

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
        self.always_random = False
        if self.is_first_player == 2:
            self.board[random.randint(0, len(self.board) - 1)] = 10
        elif self.is_first_player == 0:
            if random.randint(0, 1) == 1:
                self.board[random.randint(0, len(self.board) - 1)] = 10
    
    def is_valid_board(self) -> bool:
        have_places = False
        for i in range(len(self.board)):
            if self.board[i] == 0:
                have_places = True

        if not have_places:
            return False

        r0 = self.board[0] + self.board[1] + self.board[2]
        r1 = self.board[3] + self.board[4] + self.board[5]
        r2 = self.board[6] + self.board[7] + self.board[8]

        c0 = self.board[0] + self.board[3] + self.board[6]
        c1 = self.board[1] + self.board[4] + self.board[7]
        c2 = self.board[2] + self.board[5] + self.board[8]

        d0 = self.board[0] + self.board[4] + self.board[8]
        d1 = self.board[2] + self.board[4] + self.board[6]
        
        if r0 == 3 or r1 == 3 or r2 == 3 or c0 == 3 or c1 == 3 or c2 == 3 or d0 == 3 or d1 == 3:
            return False
        if r0 == 30 or r1 == 30 or r2 == 30 or c0 == 30 or c1 == 30 or c2 == 30 or d0 == 30 or d1 == 30:
            return False
        return True
        
        
    def reset_random(self):
        self.reset()
        self.always_random = True
        """
        playable_board = False
        while not playable_board:
            playable_board = True
            for s in range(0, random.randint(0, 5)):
                if not self.game_over:
                    self.act_with_action_id(random.choice(self.available_actions_ids()))
                else:
                    playable_board = False
        """
        first = True
        while not self.is_valid_board() or first:
            first = False
            for i in range(0, 9):
                c = random.choices((0, 1, 10), (0.5, 0.25, 0.25))[0]
                self.board[i] = c
        self.always_random = False

