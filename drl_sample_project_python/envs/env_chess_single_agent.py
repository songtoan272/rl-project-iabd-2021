import random
import numpy as np
from drl_sample_project_python.do_not_touch.contracts import SingleAgentEnv
from enum import Enum
import copy
from datetime import datetime


class Color(Enum):
    white = 0
    black = 1


class Pieces(Enum):
    empty = 1
    white_pawn = 2
    black_pawn = 8
    white_rook = 3
    black_rook = 9
    white_knight = 4
    black_knight = 10
    white_bishop = 5
    black_bishop = 11
    white_queen = 6
    black_queen = 12
    white_king = 7
    black_king = 13


class CheckOptions(Enum):
    nothing = 0
    check = 1
    checkmate = 2
    pat = 3


def operation_case(case: int, op: str) -> int:
    assert(len(op) == 4)
    if op[0] == '+':
        if row_of_case(case) + int(op[1]) > 7:
            return -1
        case += 8 * int(op[1])
    elif op[0] == '-':
        if row_of_case(case) - int(op[1]) < 0:
            return -1
        case -= 8 * int(op[1])
    if op[2] == '+':
        if col_of_case(case) + int(op[3]) > 7:
            return -1
        case += int(op[3])
    elif op[2] == '-':
        if col_of_case(case) - int(op[3]) < 0:
            return -1
        case -= int(op[3])
    return case


def col_of_case(case: int) -> int:
    return case % 8


def row_of_case(case: int) -> int:
    return case // 8


def valid_case(case: int) -> bool:
    return -1 < case < 64


def all_white_pieces() -> [int]:
    return [Pieces.white_pawn.value, Pieces.white_rook.value, Pieces.white_knight.value,
            Pieces.white_bishop.value, Pieces.white_queen.value, Pieces.white_king.value]


def all_black_pieces() -> [int]:
    return [Pieces.black_pawn.value, Pieces.black_rook.value, Pieces.black_knight.value,
            Pieces.black_bishop.value, Pieces.black_queen.value, Pieces.black_king.value]


def all_color_value() -> [int]:
    return [Color.white.value, Color.black.value]


class ChessBoard():
    def __init__(self, board: [int] = None):
        if board is None:
            self.init_board()
        else:
            self.board = board

    def is_empty(self, case: int) -> bool:
        if not valid_case(case):
            return False
        if self.board[int(case)] != Pieces.empty.value:
            return False
        return True

    def is_same_color(self, case: int, color: int) -> bool:
        if not valid_case(case):
            return False
        if self.board[int(case)] in all_white_pieces() and color == Color.white.value:
            return True
        if self.board[int(case)] in all_black_pieces() and color == Color.black.value:
            return True
        return False

    def is_enemy(self, case: int, color: int) -> bool:
        if not valid_case(case):
            return False
        if not self.is_empty(case) and not self.is_same_color(case, color):
            return True
        return False

    def get_all_cases(self) -> [int]:
        return self.board

    def get_current_state(self) -> int:
        num = ''
        for b in self.get_all_cases():
            num += str(int(b) + 10)
        return int(num)

    def play(self, move: int):
        if move > 1000:
            current_case = move // 100 - 10
            target_case = move % 100 - 10

            self.board[target_case] = self.board[current_case]
            self.board[current_case] = 1

    # 0: No, 1: check, 2: checkmate, 3: pat
    def is_check(self, color: int) -> int:
        enemy_moves = np.array([], dtype=np.int)
        ally_moves = np.array([], dtype=np.int)
        king_case = 0
        for c in range(64):
            if self.board[c] == Pieces.white_king.value and color == Color.white.value:
                king_case = c
            if self.board[c] == Pieces.black_king.value and color == Color.black.value:
                king_case = c
            enemy_moves = np.append(enemy_moves, self.available_moves(c, Color.black.value if color == Color.white.value else Color.white.value))
            ally_moves = np.append(ally_moves, self.available_moves(c, color))

        r = CheckOptions.nothing.value
        if king_case in enemy_moves:
            r = CheckOptions.check.value
            if len(ally_moves) == 0:
                r = CheckOptions.checkmate.value
        elif len(ally_moves) == 0:
            r = CheckOptions.pat.value

        return r

    def white_pawn_moves(self, case: int, color: int) -> np.ndarray:
        r = np.array([], dtype=np.int)

        if row_of_case(case) == 6 and self.is_empty(operation_case(case, '-2__')) and self.is_empty(operation_case(case, '-1__')):
            r = np.append(r, operation_case(case, '-2__'))
        if self.is_empty(operation_case(case, '-1__')):
            r = np.append(r, operation_case(case, '-1__'))
            if valid_case(operation_case(case, '-1-1')) and self.is_enemy(operation_case(case, '-1-1'), color):
                r = np.append(r, operation_case(case, '-1-1'))
            if valid_case(operation_case(case, '-1+1')) and self.is_enemy(operation_case(case, '-1+1'), color):
                r = np.append(r, operation_case(case, '-1+1'))
        return r

    def black_pawn_moves(self, case: int, color: int) -> np.ndarray:
        r = np.array([], dtype=np.int)

        if row_of_case(case) == 1 and self.is_empty(operation_case(case, '+2__')) and self.is_empty(operation_case(case, '+1__')):
            r = np.append(r, operation_case(case, '+2__'))
        if self.is_empty(operation_case(case, '+1__')):
            r = np.append(r, operation_case(case, '+1__'))
            if valid_case(operation_case(case, '+1-1')) and self.is_enemy(operation_case(case, '+1-1'), color):
                r = np.append(r, operation_case(case, '-1-1'))
            if valid_case(operation_case(case, '+1+1')) and self.is_enemy(operation_case(case, '+1+1'), color):
                r = np.append(r, operation_case(case, '+1+1'))
        return r

    def rook_moves(self, case: int, color: int) -> np.ndarray:
        r = np.array([], dtype=np.int)

        minus_col = False
        minus_row = False
        plus_col = False
        plus_row = False
        for a in range(1, 8):
            if not minus_col and col_of_case(case) - a >= 0:
                minus_col = True
                if self.is_empty(operation_case(case, '__-' + str(a))):
                    r = np.append(r, operation_case(case, '__-' + str(a)))
                    minus_col = False
                elif self.is_enemy(operation_case(case, '__-' + str(a)), color):
                    r = np.append(r, operation_case(case, '__-' + str(a)))

            if not minus_row and row_of_case(case) - a >= 0:
                minus_row = True
                if self.is_empty(operation_case(case, '-' + str(a) + '__')):
                    r = np.append(r, operation_case(case, '-' + str(a) + '__'))
                    minus_row = False
                elif self.is_enemy(operation_case(case, '-' + str(a) + '__'), color):
                    r = np.append(r, operation_case(case, '-' + str(a) + '__'))

            if not plus_col and col_of_case(case) + a < 8:
                plus_col = True
                if self.is_empty(operation_case(case, '__+' + str(a))):
                    r = np.append(r, operation_case(case, '__+' + str(a)))
                    plus_col = False
                elif self.is_enemy(operation_case(case, '__+' + str(a)), color):
                    r = np.append(r, operation_case(case, '__+' + str(a)))

            if not plus_row and row_of_case(case) + a < 8:
                plus_row = True
                if self.is_empty(operation_case(case, '+' + str(a) + '__')):
                    r = np.append(r, operation_case(case, '+' + str(a) + '__'))
                    plus_row = False
                elif self.is_enemy(operation_case(case, '+' + str(a) + '__'), color):
                    r = np.append(r, operation_case(case, '+' + str(a) + '__'))
        return r

    def knight_moves(self, case: int, color: int) -> np.ndarray:
        r = np.array([], dtype=np.int)
        if row_of_case(case) - 2 >= 0:
            if col_of_case(case) - 1 >= 0 and not self.is_same_color(operation_case(case, '-2-1'), color):
                r = np.append(r, operation_case(case, '-2-1'))
            if col_of_case(case) + 1 < 8 and not self.is_same_color(operation_case(case, '-2+1'), color):
                r = np.append(r, operation_case(case, '-2+1'))
        if row_of_case(case) - 1 >= 0:
            if col_of_case(case) - 2 >= 0 and not self.is_same_color(operation_case(case, '-1-2'), color):
                r = np.append(r, operation_case(case, '-1-2'))
            if col_of_case(case) + 2 < 8 and not self.is_same_color(operation_case(case, '-1+2'), color):
                r = np.append(r, operation_case(case, '-1+2'))
        if row_of_case(case) + 1 < 8:
            if col_of_case(case) - 2 >= 0 and not self.is_same_color(operation_case(case, '+1-2'), color):
                r = np.append(r, operation_case(case, '+1-2'))
            if col_of_case(case) + 2 < 8 and not self.is_same_color(operation_case(case, '+1+2'), color):
                r = np.append(r, operation_case(case, '+1+2'))
        if row_of_case(case) + 2 < 8:
            if col_of_case(case) - 1 >= 0 and not self.is_same_color(operation_case(case, '+2-1'), color):
                r = np.append(r, operation_case(case, '+2-1'))
            if col_of_case(case) + 1 < 8 and not self.is_same_color(operation_case(case, '+2+1'), color):
                r = np.append(r, operation_case(case, '+2+1'))
        return r

    def bishop_moves(self, case: int, color: int) -> np.ndarray:
        r = np.array([], dtype=np.int)
        d1 = False  # Diag Up Left
        d2 = False  # Diag Up Right
        d3 = False  # Diag Down Right
        d4 = False  # Diag Down Left
        for a in range(1, 8):
            if not d1 and row_of_case(case) - a >= 0 and col_of_case(case) - a >= 0:
                d1 = True
                if self.is_empty(operation_case(case, '-' + str(a) + '-' + str(a))):
                    r = np.append(r, operation_case(case, '-' + str(a) + '-' + str(a)))
                    d1 = False
                elif self.is_enemy(operation_case(case, '-' + str(a) + '-' + str(a)), color):
                    r = np.append(r, operation_case(case, '-' + str(a) + '-' + str(a)))
            if not d2 and row_of_case(case) - a >= 0 and col_of_case(case) + a < 8:
                d2 = True
                if self.is_empty(operation_case(case, '-' + str(a) + '+' + str(a))):
                    r = np.append(r, operation_case(case, '-' + str(a) + '+' + str(a)))
                    d2 = False
                elif self.is_enemy(operation_case(case, '-' + str(a) + '+' + str(a)), color):
                    r = np.append(r, operation_case(case, '-' + str(a) + '+' + str(a)))
            if not d3 and row_of_case(case) + a < 8 and col_of_case(case) + a < 8:
                d3 = True
                if self.is_empty(operation_case(case, '+' + str(a) + '+' + str(a))):
                    r = np.append(r, operation_case(case, '+' + str(a) + '+' + str(a)))
                    d3 = False
                elif self.is_enemy(operation_case(case, '+' + str(a) + '+' + str(a)), color):
                    r = np.append(r, operation_case(case, '+' + str(a) + '+' + str(a)))
            if not d4 and row_of_case(case) + a < 8 and col_of_case(case) - a >= 0:
                d4 = True
                if self.is_empty(operation_case(case, '+' + str(a) + '-' + str(a))):
                    r = np.append(r, operation_case(case, '+' + str(a) + '-' + str(a)))
                    d4 = False
                elif self.is_enemy(operation_case(case, '+' + str(a) + '-' + str(a)), color):
                    r = np.append(r, operation_case(case, '+' + str(a) + '-' + str(a)))
        return r

    def king_moves(self, case: int, color: int) -> np.ndarray:
        r = np.array([], dtype=np.int)
        if self.is_empty(operation_case(case, '-1-1')) or self.is_enemy(operation_case(case, '-1-1'), color):
            r = np.append(r, operation_case(case, '-1-1'))
        if self.is_empty(operation_case(case, '-1+1')) or self.is_enemy(operation_case(case, '-1+1'), color):
            r = np.append(r, operation_case(case, '-1+1'))
        if self.is_empty(operation_case(case, '+1+1')) or self.is_enemy(operation_case(case, '+1+1'), color):
            r = np.append(r, operation_case(case, '+1+1'))
        if self.is_empty(operation_case(case, '+1-1')) or self.is_enemy(operation_case(case, '+1-1'), color):
            r = np.append(r, operation_case(case, '+1-1'))
        if self.is_empty(operation_case(case, '-1__')) or self.is_enemy(operation_case(case, '-1__'), color):
            r = np.append(r, operation_case(case, '-1__'))
        if self.is_empty(operation_case(case, '+1__')) or self.is_enemy(operation_case(case, '+1__'), color):
            r = np.append(r, operation_case(case, '+1__'))
        if self.is_empty(operation_case(case, '__-1')) or self.is_enemy(operation_case(case, '__-1'), color):
            r = np.append(r, operation_case(case, '__-1'))
        if self.is_empty(operation_case(case, '__+1')) or self.is_enemy(operation_case(case, '__+1'), color):
            r = np.append(r, operation_case(case, '__+1'))
        return r

    def available_moves(self, case: int, color: int) -> np.ndarray:
        result = np.array([], dtype=np.int)
        if self.is_empty(case):
            return result
        if self.board[case] not in all_white_pieces() and color == Color.white.value:
            return result
        if self.board[case] not in all_black_pieces() and color == Color.black.value:
            return result
        elif self.board[case] == Pieces.white_pawn.value:
            return self.white_pawn_moves(case, Color.white.value)
        elif self.board[case] == Pieces.black_pawn.value:
            return self.black_pawn_moves(case, Color.black.value)
        elif self.board[case] in (Pieces.white_rook.value, Pieces.black_rook.value):
            return self.rook_moves(case, color)
        elif self.board[case] in (Pieces.white_knight.value, Pieces.black_knight.value):
            return self.knight_moves(case, color)
        elif self.board[case] in (Pieces.white_bishop.value, Pieces.black_bishop.value):
            return self.bishop_moves(case, color)
        elif self.board[case] in (Pieces.white_king.value, Pieces.black_king.value):
            return self.king_moves(case, color)
        elif self.board[case] in (Pieces.white_queen.value, Pieces.black_queen.value):
            result = np.append(result, self.rook_moves(case, color))
            result = np.append(result, self.bishop_moves(case, color))
        return result

    def init_board(self):
        self.board = np.ones(64)

        for c in range(64):
            if row_of_case(c) == 1:
                self.board[c] = Pieces.black_pawn.value
            if row_of_case(c) == 6:
                self.board[c] = Pieces.white_pawn.value

        self.board[0] = Pieces.black_rook.value
        self.board[1] = Pieces.black_knight.value
        self.board[2] = Pieces.black_bishop.value
        self.board[3] = Pieces.black_queen.value
        self.board[4] = Pieces.black_king.value
        self.board[5] = Pieces.black_bishop.value
        self.board[6] = Pieces.black_knight.value
        self.board[7] = Pieces.black_rook.value

        self.board[56] = Pieces.white_rook.value
        self.board[57] = Pieces.white_knight.value
        self.board[58] = Pieces.white_bishop.value
        self.board[59] = Pieces.white_queen.value
        self.board[60] = Pieces.white_king.value
        self.board[61] = Pieces.white_bishop.value
        self.board[62] = Pieces.white_knight.value
        self.board[63] = Pieces.white_rook.value


class EnvChessSingleAgent(SingleAgentEnv):
    def __init__(self, max_steps: int, color: int = Color.white.value):
        assert(max_steps > 0)
        assert(color in all_color_value())
        self.max_steps = max_steps
        self.color = color
        self.reset()

    def state_id(self) -> int:
        return self.board.get_current_state()

    def score(self) -> float:
        return self.current_score

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        self.board.play(action_id)

        r = self.board.is_check(Color.black.value if self.color == Color.white.value else Color.white.value)
        if r == CheckOptions.check.value:
            self.current_score += 1.0
        elif r == CheckOptions.pat.value:
            self.current_score = -1.0
            self.game_over = True
        elif r == CheckOptions.checkmate.value:
            self.current_score += 100.0
            self.game_over = True

        if self.game_over:
            return

        moves = np.array([], dtype=np.int)
        for c in range(64):
            move = self.board.available_moves(c, Color.black.value if self.color == Color.white.value else Color.white.value)
            moves = np.append(moves, move)

        m = random.choice(moves)
        self.board.play(m)

        r = self.board.is_check(self.color)
        if r == CheckOptions.check.value:
            self.current_score -= 1.0
        elif r == CheckOptions.pat.value:
            self.current_score = -1.0
            self.game_over = True
        elif r == CheckOptions.checkmate.value:
            self.current_score = -100.0
            self.game_over = True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.game_over = True

    def available_actions_ids(self) -> np.ndarray:
        print(datetime.now().strftime("%H:%M:%S"))
        result = np.array([], dtype=np.int)
        for c in range(64):
            for m in self.board.available_moves(c, self.color):
                tmp_board = copy.deepcopy(self.board)
                action = (c + 10) * 100 + m + 10
                tmp_board.play(action)
                if tmp_board.is_check(self.color) == CheckOptions.nothing.value:
                    result = np.append(result, action)

        return result

    def get_current_step(self) -> int:
        return self.current_step

    def reset(self):
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0
        self.queen_castling_available = True
        self.king_castling_available = True
        self.board = ChessBoard()
