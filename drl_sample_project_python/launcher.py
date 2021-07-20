import copy
import time
import tkinter as tk
import json
import random
from tkinter import messagebox
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

import drl_sample_project_python.envs.env_line_world_deep_single_agent as lw_deep
import drl_sample_project_python.envs.env_grid_world_deep_single_agent as gw_deep
import drl_sample_project_python.envs.env_tictactoe_deep_single_agent as ttt_deep
import drl_sample_project_python.envs.env_pac_man_deep_single_agent as pm_deep

FONT = 'Comic Sans MS'

GAMES = [('Line World', 1), ('Grid World', 2), ('Tic Tac Toe', 3), ('Pac-Man (Small custom)', 4),
         ('Pac-Man (Mediul custom)', 5), ('Pac-Man (Huge custom)', 6), ('Pac-Man (True level 1)', 7)]
ALGOS = [('Policy iteration', 1), ('Value iteration', 2), ('Monte Carlo ES', 3),
         ('On policy first visit Monte Carlo', 4), ('Off policy Monte Carlo Control', 5), ('Sarsa', 6),
         ('Q-learning', 7), ('Expected Sarsa', 8), ('Periodic semi-gradient sarsa', 9), ('DQN', 10),
         ('REINFORCE Monte-Carlo Policy-Gradient Control', 11), ('REINFORCE with Baseline', 12)]

env_line_world = lw_deep.EnvLineWorldDeepSingleAgent(7, 100)
env_grid_world = gw_deep.EnvGridWorldDeepSingleAgent(5, 5, 100, (4, 4), (0, 0))
env_tic_tact_toe = ttt_deep.EnvTicTacToeDeepSingleAgent(100)
env_pac_man = [pm_deep.EnvPacManDeepSingleAgent(100, './models/pac_man_level_custom_2.txt')]

SIMULATE_BTN_POS = (650, 860)
CANVAS = []
BUTTONS = []
LABELS = []
TTT_LABELS = []


def random_color():
    return list(np.random.choice(range(256), size=3))


def load_neuralnet(game: int, algo: int):
    file_name = ''
    if game == 1:
        if algo == 9:
            file_name = 'episodic_semi_gradient_sarsa_line_world'
        elif algo == 10:
            file_name = 'deep_q_learning_line_world'
        elif algo == 11:
            file_name = ''
        elif algo == 12:
            file_name = ''
    elif game == 2:
        if algo == 9:
            file_name = 'episodic_semi_gradient_sarsa_grid_world'
        elif algo == 10:
            file_name = 'deep_q_learning_grid_world'
        elif algo == 11:
            file_name = ''
        elif algo == 12:
            file_name = ''
    elif game == 3:
        if algo == 9:
            file_name = 'episodic_semi_gradient_sarsa_tic_tac_toe'
        elif algo == 10:
            file_name = 'deep_q_learning_tic_tac_toe'
        elif algo == 11:
            file_name = ''
        elif algo == 12:
            file_name = ''
    elif game in (4, 5, 6, 7):
        if algo == 9:
            file_name = 'episodic_semi_gradient_sarsa_pac_man'
        elif algo == 10:
            file_name = 'deep_q_learning_pac_man'
        elif algo == 11:
            file_name = ''
        elif algo == 12:
            file_name = ''

    if game == 4:
        file_name += '_small_custom'
    elif game == 5:
        file_name += '_medium_custom'
    elif game == 6:
        file_name += '_huge_custom'
    elif game == 7:
        file_name += '_true_level_1'

    path = './models/' + file_name + '.h5'
    nerual_net = tf.keras.models.load_model(path, compile=False)
    return nerual_net


def predict_model(neural_net, env) -> int:
    all_q_value = np.squeeze(predict_model_all_q_value(neural_net, env))
    return env.available_actions_ids()[np.argmax(all_q_value)]


def predict_model_all_q_value(neural_net, env):
    all_q_inputs = np.zeros((len(env.available_actions_ids()),
                             env.state_description_length() + env.max_actions_count()))
    for i, a in enumerate(env.available_actions_ids()):
        all_q_inputs[i] = np.hstack(
            [env.state_description(), tf.keras.utils.to_categorical(a, env.max_actions_count())])
    all_q_value = neural_net.predict(all_q_inputs)
    return all_q_value


def import_json(game: int, algo: int):
    file_name = ''
    if game == 1:
        if algo == 1:
            file_name = 'policy_iteration_line_world'
        elif algo == 2:
            file_name = 'value_iteration_line_world'
        elif algo == 3:
            file_name = 'monte_carlo_es_line_world'
        elif algo == 4:
            file_name = 'on_policy_monte_carlo_line_world'
        elif algo == 5:
            file_name = 'off_policy_monte_carlo_line_world'
        elif algo == 6:
            file_name = 'sarsa_line_world'
        elif algo == 7:
            file_name = 'q_learning_line_world'
        elif algo == 8:
            file_name = 'expected_sarsa_line_world'
    elif game == 2:
        if algo == 1:
            file_name = 'policy_iteration_grid_world'
        elif algo == 2:
            file_name = 'value_iteration_grid_world'
        elif algo == 3:
            file_name = 'monte_carlo_es_grid_world'
        elif algo == 4:
            file_name = 'on_policy_monte_carlo_grid_world'
        elif algo == 5:
            file_name = 'off_policy_monte_carlo_grid_world'
        elif algo == 6:
            file_name = 'sarsa_grid_world'
        elif algo == 7:
            file_name = 'q_learning_grid_world'
        elif algo == 8:
            file_name = 'expected_sarsa_grid_world'
    elif game == 3:
        if algo == 3:
            file_name = 'monte_carlo_es_tic_tac_toe'
        elif algo == 4:
            file_name = 'on_policy_monte_carlo_tic_tac_toe'
        elif algo == 5:
            file_name = 'off_policy_monte_carlo_tic_tac_toe'
        elif algo == 6:
            file_name = 'sarsa_tic_tac_toe'
        elif algo == 7:
            file_name = 'q_learning_tic_tac_toe'
        elif algo == 8:
            file_name = 'expected_sarsa_tic_tac_toe'
    f = open('./models/' + file_name + '.json', 'r')
    data = json.load(f)
    f.close()

    return data


def import_json_q(game: int, algo: int):
    file_name = ''
    if game == 1:
        if algo == 3:
            file_name = 'monte_carlo_es_line_world_q'
        elif algo == 4:
            file_name = 'on_policy_monte_carlo_line_world_q'
        elif algo == 5:
            file_name = 'off_policy_monte_carlo_line_world_q'
        elif algo == 6:
            file_name = 'sarsa_line_world_q'
        elif algo == 7:
            file_name = 'q_learning_line_world_q'
        elif algo == 8:
            file_name = 'expected_sarsa_line_world_q'
    elif game == 2:
        if algo == 3:
            file_name = 'monte_carlo_es_grid_world_q'
        elif algo == 4:
            file_name = 'on_policy_monte_carlo_grid_world_q'
        elif algo == 5:
            file_name = 'off_policy_monte_carlo_grid_world_q'
        elif algo == 6:
            file_name = 'sarsa_grid_world_q'
        elif algo == 7:
            file_name = 'q_learning_grid_world_q'
        elif algo == 8:
            file_name = 'expected_sarsa_grid_world_q'
    elif game == 3:
        if algo == 3:
            file_name = 'monte_carlo_es_tic_tac_toe_q'
        elif algo == 4:
            file_name = 'on_policy_monte_carlo_tic_tac_toe_q'
        elif algo == 5:
            file_name = 'off_policy_monte_carlo_tic_tac_toe_q'
        elif algo == 6:
            file_name = 'sarsa_tic_tac_toe_q'
        elif algo == 7:
            file_name = 'q_learning_tic_tac_toe_q'
        elif algo == 8:
            file_name = 'expected_sarsa_tic_tac_toe_q'

    if file_name == '':
        return {}
    f = open('./models/' + file_name + '.json', 'r')
    data = json.load(f)
    f.close()

    return data


def pac_man(algo: int, canvas, game):

    if game == 4:
        env_pac_man[0] = pm_deep.EnvPacManDeepSingleAgent(100, './models/pac_man_level_custom_2.txt')
    elif game == 5:
        env_pac_man[0] = pm_deep.EnvPacManDeepSingleAgent(100, './models/pac_man_level_custom.txt')
    elif game == 6:
        env_pac_man[0] = pm_deep.EnvPacManDeepSingleAgent(100, './models/pac_man_level_custom_3.txt')
    elif game == 7:
        env_pac_man[0] = pm_deep.EnvPacManDeepSingleAgent(100, './models/pac_man_level_1.txt')

    board = []
    ghosts = []
    env_pac_man[0].reset()
    board, rows, cols, pacgum_count, ghosts = env_pac_man[0].init_board()
    initial_board = copy.deepcopy(board)

    wrapper = {}

    square_size = round(min(750 / rows, 900 / cols), 0)

    pac = Image.open('./models/pac_man.png')
    pac.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
    tkinter_pac = ImageTk.PhotoImage(pac)
    pac_canvas = tk.Canvas(width=square_size-2, height=square_size-2, bg='black', highlightthickness=0)
    pac_canvas.pack()
    pac_canvas.place(x=-100, y=-100)
    pac_canvas.create_image(1, 1, image=tkinter_pac, anchor=tk.NW)
    pac_label = tk.Label(width=square_size, height=square_size, image=tkinter_pac)
    pac_label.image = tkinter_pac
    CANVAS.append(pac_canvas)

    if ghosts[0] != -100:
        red = Image.open('./models/red.png')
        red.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_red = ImageTk.PhotoImage(red)
        red_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        red_canvas.pack()
        red_canvas.place(x=-100, y=-100)
        red_canvas.create_image(1, 1, image=tkinter_red, anchor=tk.NW)
        red_label = tk.Label(width=square_size, height=square_size, image=tkinter_red)
        red_label.image = tkinter_red
        CANVAS.append(red_canvas)

    if ghosts[1] != -100:
        blue = Image.open('./models/blue.png')
        blue.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_blue = ImageTk.PhotoImage(blue)
        blue_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        blue_canvas.pack()
        blue_canvas.place(x=-100, y=-100)
        blue_canvas.create_image(1, 1, image=tkinter_blue, anchor=tk.NW)
        blue_label = tk.Label(width=square_size, height=square_size, image=tkinter_blue)
        blue_label.image = tkinter_blue
        CANVAS.append(blue_canvas)

    if ghosts[2] != -100:
        pink = Image.open('./models/pink.png')
        pink.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_pink = ImageTk.PhotoImage(pink)
        pink_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        pink_canvas.pack()
        pink_canvas.place(x=-100, y=-100)
        pink_canvas.create_image(1, 1, image=tkinter_pink, anchor=tk.NW)
        pink_label = tk.Label(width=square_size, height=square_size, image=tkinter_pink)
        pink_label.image = tkinter_pink
        CANVAS.append(pink_canvas)

    if ghosts[3] != -100:
        orange = Image.open('./models/orange.png')
        orange.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_orange = ImageTk.PhotoImage(orange)
        orange_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        orange_canvas.pack()
        orange_canvas.place(x=-100, y=-100)
        orange_canvas.create_image(1, 1, image=tkinter_orange, anchor=tk.NW)
        orange_label = tk.Label(width=square_size, height=square_size, image=tkinter_orange)
        orange_label.image = tkinter_orange
        CANVAS.append(orange_canvas)

    def get_q_value():
        LABELS[0].config(text='')
        LABELS[1].config(text='')
        LABELS[2].config(text='')
        LABELS[3].config(text='')
        neural_net = load_neuralnet(4, algo)
        all_q_value = predict_model_all_q_value(neural_net, env_pac_man[0]).flatten()
        for i, a in enumerate(env_pac_man[0].available_actions_ids()):
            LABELS[a].config(text=round(all_q_value[i], 4))

    def create_line():
        canvas.delete("all")
        canvas.place(x=420, y=80)
        for i in range(rows):
            for j in range(cols):
                fill = 'black'
                outline = 'black'

                if not board[i * cols + j].isnumeric():
                    if board[i * cols + j] not in list(wrapper.keys()):
                        wrapper[board[i * cols + j]] = random_color()
                    fill = "#%02x%02x%02x" % (wrapper[board[i * cols + j]][0], wrapper[board[i * cols + j]][1], wrapper[board[i * cols + j]][2])
                if board[i * cols + j] == '7':
                    fill = "#%02x%02x%02x" % (25, 25, 166)
                    outline = fill
                canvas.create_rectangle(5 + j * square_size, 5 + i * square_size, 5 + square_size + j * square_size, 5 + square_size + i * square_size,
                                        outline=outline, fill=fill)

                if board[i * cols + j] == '1':
                    canvas.create_oval(5 + square_size / 3 + j * square_size, 5 + square_size / 3 + i * square_size,
                                       5 + square_size - square_size / 3 + j * square_size, 5 + square_size - square_size / 3 + i * square_size,
                                       outline="white", fill='white')

                if board[i * cols + j] == '2':
                    pac_canvas.place(x=420 + 6 + j * square_size, y=80 + 6 + i * square_size)
                if ghosts[0] == i * cols + j:
                    red_canvas.place(x=420 + 6 + j * square_size, y=80 + 6 + i * square_size)
                if ghosts[1] == i * cols + j:
                    blue_canvas.place(x=420 + 6 + j * square_size, y=80 + 6 + i * square_size)
                if ghosts[2] == i * cols + j:
                    pink_canvas.place(x=420 + 6 + j * square_size, y=80 + 6 + i * square_size)
                if ghosts[3] == i * cols + j:
                    orange_canvas.place(x=420 + 6 + j * square_size, y=80 + 6 + i * square_size)
        get_q_value()

    create_line()

    def clear_pac():
        for i in range(len(board)):
            if board[i] == '2':
                if initial_board[i] == '2':
                    board[i] = '0'
                else:
                    board[i] = initial_board[i]

    def click_square(event=None):
        if 5 <= event.y <= rows * square_size and 5 <= event.x <= cols * square_size:
            i = 0
            j = 0
            while event.x > 5 + square_size * i and i <= cols:
                i += 1
            i -= 1

            while event.y > 5 + square_size * j and j <= rows:
                j += 1
            j -= 1

            clear_pac()
            if board[j * cols + i] in ('0', '1'):
                board[j * cols + i] = '2'
                env_pac_man[0].move_pac_man(j * cols + i)
            create_line()

    def simulate():
        neural_net = load_neuralnet(4, algo)
        env_pac_man[0].set_state(np.array(board).flatten())
        action = predict_model(neural_net, env_pac_man[0])
        env_pac_man[0].act_with_action_id(action)
        b = env_pac_man[0].state_description_ui()[:-4]
        for n, s in enumerate(b):
            board[n] = str(int(s))
            if s > 10:
                board[n] = chr(int(s))
        for n, g in enumerate(env_pac_man[0].get_ghosts()):
            ghosts[n] = g
        create_line()

    def loop_simulate(x):
        simulate()
        canvas.update()
        canvas.update_idletasks()
        if x > 0 and not env_pac_man[0].is_game_over():
            canvas.after(100, loop_simulate(x-1))

    canvas.bind('<Button-1>', click_square)
    create_line()

    vld_btn = tk.Button(text='Simuler', command=lambda: simulate())
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=SIMULATE_BTN_POS[0], y=SIMULATE_BTN_POS[1])

    vld_btn_loop = tk.Button(text='Simuler Loop', command=lambda: loop_simulate(500))
    vld_btn_loop.config(font=(FONT, '18'), width=13)
    vld_btn_loop.place(x=SIMULATE_BTN_POS[0] + 220, y=SIMULATE_BTN_POS[1])

    BUTTONS.append(vld_btn)
    BUTTONS.append(vld_btn_loop)


def tic_tac_toe(algo: int, canvas):
    grid = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def is_terminated() -> bool:
        one_empty = True
        for g in grid:
            if g == 0:
                one_empty = False

        r1 = grid[0] + grid[1] + grid[2]
        r2 = grid[3] + grid[4] + grid[5]
        r3 = grid[6] + grid[7] + grid[8]

        c1 = grid[0] + grid[3] + grid[6]
        c2 = grid[1] + grid[4] + grid[7]
        c3 = grid[2] + grid[5] + grid[8]

        d1 = grid[0] + grid[4] + grid[8]
        d2 = grid[2] + grid[4] + grid[6]

        if r1 == 3 or r2 == 3 or r3 == 3 or c1 == 3 or c2 == 3 or c3 == 3 or d1 == 3 or d2 == 3:
            return True
        if r1 == 30 or r2 == 30 or r3 == 30 or c1 == 30 or c2 == 30 or c3 == 30 or d1 == 30 or d2 == 30:
            return True
        return one_empty

    def place_circle(x: int, y: int):
        if grid[y * 3 + x] == 0:
            grid[y * 3 + x] = 5

    def get_q_value():
        if algo > 2:
            TTT_LABELS[0].config(text="")
            TTT_LABELS[1].config(text="")
            TTT_LABELS[2].config(text="")
            TTT_LABELS[3].config(text="")
            TTT_LABELS[4].config(text="")
            TTT_LABELS[5].config(text="")
            TTT_LABELS[6].config(text="")
            TTT_LABELS[7].config(text="")
            TTT_LABELS[8].config(text="")
        if 2 < algo <= 8:
            q = import_json_q(3, algo)
            n = ''
            for g in grid:
                if g == 0:
                    n += '1'
                elif g == 1:
                    n += '2'
                else:
                    n += '3'
            if n in q.keys():
                q_value = q[n]
                if '0' in q_value.keys():
                    TTT_LABELS[0].config(text=round(q_value['0'], 4))
                if '1' in q_value.keys():
                    TTT_LABELS[1].config(text=round(q_value['1'], 4))
                if '2' in q_value.keys():
                    TTT_LABELS[2].config(text=round(q_value['2'], 4))
                if '3' in q_value.keys():
                    TTT_LABELS[3].config(text=round(q_value['3'], 4))
                if '4' in q_value.keys():
                    TTT_LABELS[4].config(text=round(q_value['4'], 4))
                if '5' in q_value.keys():
                    TTT_LABELS[5].config(text=round(q_value['5'], 4))
                if '6' in q_value.keys():
                    TTT_LABELS[6].config(text=round(q_value['6'], 4))
                if '7' in q_value.keys():
                    TTT_LABELS[7].config(text=round(q_value['7'], 4))
                if '8' in q_value.keys():
                    TTT_LABELS[8].config(text=round(q_value['8'], 4))
        elif algo > 8:
            neural_net = load_neuralnet(3, algo)
            all_q_value = predict_model_all_q_value(neural_net, env_tic_tact_toe).flatten()
            for i, a in enumerate(env_tic_tact_toe.available_actions_ids()):
                TTT_LABELS[a].config(text=round(all_q_value[i], 4))

    def create_line():
        canvas.delete("all")
        canvas.create_line(5, 105, 305, 105)
        canvas.create_line(5, 205, 305, 205)
        canvas.create_line(105, 5, 105, 305)
        canvas.create_line(205, 5, 205, 305)

        canvas.create_line(505, 105, 805, 105)
        canvas.create_line(505, 205, 805, 205)
        canvas.create_line(605, 5, 605, 305)
        canvas.create_line(705, 5, 705, 305)

        canvas_places = (620, 310)

        for i in range(0, 3):
            for j in range(0, 3):
                if grid[j * 3 + i] == 1:
                    canvas.create_line(10 + i * 100, 10 + j * 100, 100 + i * 100, 100 + j * 100, width=3)
                    canvas.create_line(100 + i * 100, 10 + j * 100, 10 + i * 100, 100 + j * 100, width=3)
                elif grid[j * 3 + i] == 10 or grid[j * 3 + i] == 5:
                    canvas.create_oval(10 + i * 100, 10 + j * 100, 100 + i * 100, 100 + j * 100, outline="black",
                                       width=3)
        canvas.place(x=canvas_places[0], y=canvas_places[1])
        get_q_value()

    def click_square(event=None):
        if 5 <= event.y <= 305 and 5 <= event.x <= 305 and not is_terminated():
            for g in range(len(grid)):
                if grid[g] == 5:
                    grid[g] = 0
            i = 0
            j = 0

            while event.x > 5 + 100 * i and i <= 3:
                i += 1
            i -= 1

            while event.y > 5 + 100 * j and j <= 3:
                j += 1
            j -= 1

            place_circle(i, j)
            b = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for a, g in enumerate(grid):
                if g == 5:
                    b[a] = 10
                else:
                    b[a] = g
            env_tic_tact_toe.set_state(b)
            create_line()

    def simulate():
        n = ''
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for g in range(len(grid)):
            if grid[g] == 5:
                grid[g] = 10
            board[g] = grid[g]
            if grid[g] == 0:
                n += '1'
            elif grid[g] == 1:
                n += '2'
            else:
                n += '3'
        if not is_terminated():
            if algo > 8:
                neural_net = load_neuralnet(3, algo)
                env_tic_tact_toe.set_state(np.array(board).flatten())
                action = predict_model(neural_net, env_tic_tact_toe)
                env_tic_tact_toe.act_with_action_id(action)
            else:
                json_data = import_json(3, algo)
                action = json_data[n]

            x = action % 3
            y = action // 3
            grid[y * 3 + x] = 1

            create_line()

    simulate()
    canvas.bind('<Button-1>', click_square)
    create_line()

    vld_btn = tk.Button(text='Simuler', command=lambda: simulate())
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=SIMULATE_BTN_POS[0], y=SIMULATE_BTN_POS[1])

    BUTTONS.append(vld_btn)


def grid_world(algo: int, canvas):
    is_started = False
    grid = [[-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 100]]

    def set_start_position(x: int, y: int):
        for i in range(0, len(grid)):
            for j in range(0, len(grid[i])):
                grid[i][j] = 0 if grid[i][j] == 1 else grid[i][j]
        grid[x][y] = 1

    def get_q_value():
        if algo > 2:
            LABELS[0].config(text='')
            LABELS[1].config(text='')
            LABELS[2].config(text='')
            LABELS[3].config(text='')

        if 2 < algo <= 8:
            q = import_json_q(2, algo)
            n = None
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == 1:
                        n = j * 5 + i
            if n is None:
                return

            if str(n) in q.keys():
                q_value = q[str(n)]
                if '0' in q_value.keys():
                    LABELS[0].config(text=round(q_value['0'], 4))
                if '1' in q_value.keys():
                    LABELS[1].config(text=round(q_value['1'], 4))
                if '2' in q_value.keys():
                    LABELS[2].config(text=round(q_value['2'], 4))
                if '3' in q_value.keys():
                    LABELS[3].config(text=round(q_value['3'], 4))
        elif algo > 8:
            neural_net = load_neuralnet(2, algo)
            all_q_value = predict_model_all_q_value(neural_net, env_grid_world).flatten()
            for i, a in enumerate(env_grid_world.available_actions_ids()):
                LABELS[a].config(text=round(all_q_value[i], 4))

    def create_line():
        canvas.delete("all")
        for i in range(0, 5):
            for j in range(0, 5):
                fill = 'white'
                if grid[i][j] == -1:
                    fill = 'red'
                elif grid[i][j] == 1:
                    fill = '#47d147'
                elif grid[i][j] == 2:
                    fill = '#70db70'
                elif grid[i][j] == 100:
                    fill = 'blue'
                canvas.create_rectangle(5 + i * 100, 5 + j * 100, 105 + i * 100, 105 + j * 100, outline='black',
                                        fill=fill)
        canvas.place(x=500, y=210)
        get_q_value()

    def click_square(event=None):
        if 5 <= event.y <= 505 and 5 <= event.x <= 505 and not is_started:
            i = 0
            j = 0

            while event.x > 5 + 100 * i and i <= 5:
                i += 1
            i -= 1

            while event.y > 5 + 100 * j and j <= 5:
                j += 1
            j -= 1

            if (i == 0 and j == 0) or (i == 4 and j == 4):
                return
            set_start_position(i, j)
            env_grid_world.set_state(i * 5 + j)
            create_line()

    def simulate():
        x = None
        y = None
        n = None
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    n = j * 5 + i
                    x = i
                    y = j
        if algo > 8:
            neural_net = load_neuralnet(2, algo)
            env_grid_world.set_state(n)
            action = predict_model(neural_net, env_grid_world)
            env_grid_world.act_with_action_id(action)
        else:
            json_data = import_json(2, algo)
            action = json_data[str(n)]

        grid[x][y] = 2
        if action == 0:
            if x > 0:
                grid[x - 1][y] = 1
        elif action == 1:
            if x < 4:
                grid[x + 1][y] = 1
        elif action == 2:
            if y > 0:
                grid[x][y - 1] = 1
        elif action == 3:
            if y < 4:
                grid[x][y + 1] = 1

        create_line()

    canvas.bind('<Button-1>', click_square)
    create_line()

    vld_btn = tk.Button(text='Simuler', command=lambda: simulate())
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=SIMULATE_BTN_POS[0], y=SIMULATE_BTN_POS[1])

    BUTTONS.append(vld_btn)


def line_world(algo: int, canvas):
    is_started = False
    grid = [(5, -1), (105, 0), (205, 0), (305, 0), (405, 0), (505, 0), (605, 100)]

    def set_start_position(x: int):
        for i in range(0, len(grid)):
            grid[i] = (grid[i][0], 0 if grid[i][1] == 1 else grid[i][1])
        grid[x] = (grid[x][0], 1)

    def get_q_value():
        if algo > 2:
            LABELS[0].config(text='')
            LABELS[1].config(text='')

        if 2 < algo <= 8:
            q = import_json_q(1, algo)
            n = None
            for i, (x, state) in enumerate(grid):
                if state == 1:
                    n = i
            if n is None:
                return
            if str(n) in q.keys():
                q_value = q[str(n)]
                if '0' in q_value.keys():
                    LABELS[0].config(text=round(q_value['0'], 4))
                if '1' in q_value.keys():
                    LABELS[1].config(text=round(q_value['1'], 4))
        elif algo > 8:
            neural_net = load_neuralnet(1, algo)
            all_q_value = predict_model_all_q_value(neural_net, env_line_world).flatten()
            for i, a in enumerate(env_line_world.available_actions_ids()):
                LABELS[a].config(text=round(all_q_value[i], 4))

    def create_line():
        canvas.delete("all")
        for i, (x, state) in enumerate(grid):
            fill = 'white'
            if state == -1:
                fill = 'red'
            elif state == 100:
                fill = 'blue'
            elif state == 1:
                fill = '#47d147'
            elif state == 2:
                fill = '#70db70'
            canvas.create_rectangle(x, 5, 100 + x, 100, outline='black', fill=fill)
        canvas.place(x=400, y=410)
        get_q_value()

    def click_square(event=None):
        if 5 <= event.y <= 100 and 5 <= event.x <= 705 and not is_started:
            i = 0
            find = False
            while i < len(grid) and not find:
                if event.x < grid[i][0]:
                    find = True
                else:
                    i += 1
            i -= 1
            if not is_started and i not in (0, 6):
                set_start_position(i)
                env_line_world.set_state(i)
                create_line()

    def simulate():
        n = None
        for i, (x, state) in enumerate(grid):
            if state == 1:
                n = i
        if n is None:
            return

        if algo > 8:
            neural_net = load_neuralnet(1, algo)
            env_line_world.set_state(n)
            action = predict_model(neural_net, env_line_world)
            env_line_world.act_with_action_id(action)
        else:
            json_data = import_json(1, algo)
            action = json_data[str(n)]

        n_s = 0
        if action == 0:
            n_s = n - 1 if n > 0 else n
        elif action == 1:
            n_s = n + 1 if n < 6 else n
        grid[n] = (grid[n][0], 2)
        grid[n_s] = (grid[n_s][0], 1)
        create_line()

    canvas.bind('<Button-1>', click_square)
    create_line()

    vld_btn = tk.Button(text='Simuler', command=lambda: simulate())
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=SIMULATE_BTN_POS[0], y=SIMULATE_BTN_POS[1])

    BUTTONS.append(vld_btn)


def validate(game: int, algo: int, canvas):
    env_line_world.reset()
    env_grid_world.reset()
    env_tic_tact_toe.reset()
    env_pac_man[0].reset()
    while len(CANVAS) > 0:
        CANVAS[0].delete('all')
        CANVAS[0].destroy()
        CANVAS.pop(0)

    while len(BUTTONS) > 0:
        BUTTONS[0].destroy()
        BUTTONS.pop(0)

    while len(LABELS) > 0:
        LABELS[0].destroy()
        LABELS.pop(0)

    while len(TTT_LABELS) > 0:
        TTT_LABELS[0].destroy()
        TTT_LABELS.pop(0)

    global nerual_net
    nerual_net = None
    if game == 0 or algo == 0:
        messagebox.showinfo('Erreur', 'Veuillez choisir un jeu et un algorithme')
    if (game == 3 and algo in (1, 2)) or (game in (4, 5, 6, 7) and algo not in (9, 10, 11)):
        messagebox.showinfo("Erreur", "Ce couple jeu/algorithme n'est pas disponible")
        return

    if algo > 2:
        if game != 3:
            gamepad = Image.open('./models/gamepad_control.png')
            gamepad.thumbnail((100, 100), Image.ANTIALIAS)
            tkinter_gamepad = ImageTk.PhotoImage(gamepad)
            gamepad_canvas = tk.Canvas(width=100, height=100, highlightthickness=0)
            gamepad_canvas.pack()
            gamepad_canvas.place(x=1300, y=410)
            gamepad_canvas.create_image(0, 0, image=tkinter_gamepad, anchor=tk.NW)
            gamepad_label = tk.Label(width=100, height=100, image=tkinter_gamepad)
            gamepad_label.image = tkinter_gamepad
            CANVAS.append(gamepad_canvas)

            lb_left = tk.Label(text="")
            lb_left.config(font=(FONT, '12'))
            lb_left.place(x=1220, y=442)
            LABELS.append(lb_left)

            lb_right = tk.Label(text="")
            lb_right.config(font=(FONT, '12'))
            lb_right.place(x=1420, y=442)
            LABELS.append(lb_right)

            lb_up = tk.Label(text="")
            lb_up.config(font=(FONT, '12'))
            lb_up.place(x=1325, y=370)
            LABELS.append(lb_up)

            lb_down = tk.Label(text="")
            lb_down.config(font=(FONT, '12'))
            lb_down.place(x=1325, y=520)
            LABELS.append(lb_down)
        else:
            lb_1 = tk.Label(text="")
            lb_1.config(font=(FONT, '12'))
            lb_1.place(x=1140, y=350)
            TTT_LABELS.append(lb_1)

            lb_2 = tk.Label(text="")
            lb_2.config(font=(FONT, '12'))
            lb_2.place(x=1240, y=350)
            TTT_LABELS.append(lb_2)

            lb_3 = tk.Label(text="")
            lb_3.config(font=(FONT, '12'))
            lb_3.place(x=1340, y=350)
            TTT_LABELS.append(lb_3)

            lb_4 = tk.Label(text="")
            lb_4.config(font=(FONT, '12'))
            lb_4.place(x=1140, y=450)
            TTT_LABELS.append(lb_4)

            lb_5 = tk.Label(text="")
            lb_5.config(font=(FONT, '12'))
            lb_5.place(x=1240, y=450)
            TTT_LABELS.append(lb_5)

            lb_6 = tk.Label(text="")
            lb_6.config(font=(FONT, '12'))
            lb_6.place(x=1340, y=450)
            TTT_LABELS.append(lb_6)

            lb_7 = tk.Label(text="")
            lb_7.config(font=(FONT, '12'))
            lb_7.place(x=1140, y=550)
            TTT_LABELS.append(lb_7)

            lb_8 = tk.Label(text="")
            lb_8.config(font=(FONT, '12'))
            lb_8.place(x=1240, y=550)
            TTT_LABELS.append(lb_8)

            lb_9 = tk.Label(text="")
            lb_9.config(font=(FONT, '12'))
            lb_9.place(x=1340, y=550)
            TTT_LABELS.append(lb_9)

    if game == 1:
        line_world(algo, canvas)
    elif game == 2:
        grid_world(algo, canvas)
    elif game == 3:
        tic_tac_toe(algo, canvas)
    elif game in (4, 5, 6, 7):
        pac_man(algo, canvas, game)


if __name__ == "__main__":
    window = tk.Tk()
    window.title('Best projet <3')
    window.geometry('1560x920')

    main_label = tk.Label(text='Bienvenu sur le meilleur Projet DRL !')
    main_label.config(font=(FONT, '26'))
    main_label.pack()

    game_label = tk.Label(text='Choisissez le jeu :')
    game_label.config(font=(FONT, '16'))
    game_label.place(x=0, y=5)

    game = tk.IntVar()

    for x, (g, v) in enumerate(GAMES):
        rd = tk.Radiobutton(text=g, padx=20, variable=game, value=v)
        rd.config(font=(FONT, '12'))
        rd.place(x=0, y=5 + (x + 1) * 40)

    algo_label = tk.Label(text="Choisissez l'algorithme :")
    algo_label.config(font=(FONT, '16'))
    algo_label.place(x=0, y=320)

    algo = tk.IntVar()

    for x, (g, v) in enumerate(ALGOS):
        rd = tk.Radiobutton(text=g, padx=20, variable=algo, value=v)
        rd.config(font=(FONT, '12'))
        rd.place(x=0, y=320 + (x + 1) * 40)

    canvas = tk.Canvas(width=1000, height=1000)
    vld_btn = tk.Button(text='Valider', command=lambda: validate(game.get(), algo.get(), canvas))
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=10, y=860)

    window.mainloop()
