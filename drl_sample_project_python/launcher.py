import copy
import time
import tkinter as tk
import json
import random
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

import drl_sample_project_python.envs.env_line_world_deep_single_agent as lw_deep
import drl_sample_project_python.envs.env_grid_world_deep_single_agent as gw_deep
import drl_sample_project_python.envs.env_tictactoe_deep_single_agent as ttt_deep
import drl_sample_project_python.envs.env_pac_man_deep_single_agent as pm_deep

FONT = 'Comic Sans MS'

GAMES = [('Line World', 1), ('Grid World', 2), ('Tic Tac Toe', 3), ('Pac-Man', 4)]
ALGOS = [('Policy iteration', 1), ('Value iteration', 2), ('Monte Carlo ES', 3),
         ('On policy first visit Monte Carlo', 4), ('Off policy Monte Carlo Control', 5), ('Sarsa', 6),
         ('Q-learning', 7), ('Expected Sarsa', 8), ('Periodic semi-gradient sarsa', 9), ('DQN', 10)]

env_line_world = lw_deep.EnvLineWorldDeepSingleAgent(7, 100)
env_grid_world = gw_deep.EnvGridWorldDeepSingleAgent(5, 5, 100, (4, 4), (0, 0))
env_tic_tact_toe = ttt_deep.EnvTicTacToeDeepSingleAgent(100)
env_pac_man = pm_deep.EnvPacManDeepSingleAgent(100, './models/pac_man_level_custom_2.txt')

SIMULATE_BTN_POS = (650, 845)
CANVAS = []


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
            file_name = ''
        elif algo == 11:
            file_name = ''
        elif algo == 12:
            file_name = ''
    elif game == 4:
        if algo == 9:
            file_name = 'episodic_semi_gradient_sarsa_pac_man'
        elif algo == 10:
            file_name = ''
        elif algo == 11:
            file_name = ''
        elif algo == 12:
            file_name = ''

    path = './models/' + file_name + '.h5'
    nerual_net = tf.keras.models.load_model(path, compile=False)
    return nerual_net


def predict_model(neural_net, env) -> int:
    all_q_inputs = np.zeros((len(env.available_actions_ids()),
                             env.state_description_length() + env.max_actions_count()))
    for i, a in enumerate(env.available_actions_ids()):
        all_q_inputs[i] = np.hstack(
            [env.state_description(), tf.keras.utils.to_categorical(a, env.max_actions_count())])
    all_q_value = np.squeeze(neural_net.predict(all_q_inputs))
    print(all_q_value)
    return env.available_actions_ids()[np.argmax(all_q_value)]


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
            file_name = 'expected_sarsa_line_word'
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


def pac_man(algo: int, canvas):
    board = []
    ghosts = []
    env_pac_man.reset()
    board, rows, cols, pacgum_count, ghosts = env_pac_man.init_board()
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

    if ghosts[0] != -100:
        red = Image.open('./models/red.png')
        red.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_red = ImageTk.PhotoImage(red)
        red_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        red_canvas.pack()
        red_canvas.place(x=-100, y=-100)
        red_canvas.create_image(1, 1, image=tkinter_red, anchor=tk.NW)

    if ghosts[1] != -100:
        blue = Image.open('./models/blue.png')
        blue.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_blue = ImageTk.PhotoImage(blue)
        blue_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        blue_canvas.pack()
        blue_canvas.place(x=-100, y=-100)
        blue_canvas.create_image(1, 1, image=tkinter_blue, anchor=tk.NW)

    if ghosts[2] != -100:
        pink = Image.open('./models/pink.png')
        pink.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_pink = ImageTk.PhotoImage(pink)
        pink_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        pink_canvas.pack()
        pink_canvas.place(x=-100, y=-100)
        pink_canvas.create_image(1, 1, image=tkinter_pink, anchor=tk.NW)

    if ghosts[3] != -100:
        orange = Image.open('./models/orange.png')
        orange.thumbnail((square_size - 4, square_size - 4), Image.ANTIALIAS)
        tkinter_orange = ImageTk.PhotoImage(orange)
        orange_canvas = tk.Canvas(width=square_size - 2, height=square_size - 2, bg='black', highlightthickness=0)
        orange_canvas.pack()
        orange_canvas.place(x=-100, y=-100)
        orange_canvas.create_image(1, 1, image=tkinter_orange, anchor=tk.NW)

    pac_label = tk.Label(width=square_size, height=square_size, image=tkinter_pac)
    pac_label.image = tkinter_pac
    red_label = tk.Label(width=square_size, height=square_size, image=tkinter_red)
    red_label.image = tkinter_red
    pink_label = tk.Label(width=square_size, height=square_size, image=tkinter_pink)
    pink_label.image = tkinter_pink
    orange_label = tk.Label(width=square_size, height=square_size, image=tkinter_orange)
    orange_label.image = tkinter_orange
    blue_label = tk.Label(width=square_size, height=square_size, image=tkinter_blue)
    blue_label.image = tkinter_blue

    CANVAS.append(pac_canvas)
    CANVAS.append(red_canvas)
    CANVAS.append(pink_canvas)
    CANVAS.append(blue_canvas)
    CANVAS.append(orange_canvas)

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
                env_pac_man.move_pac_man(j * cols + i)
            create_line()

    def simulate():
        neural_net = load_neuralnet(4, algo)
        env_pac_man.set_state(np.array(board).flatten())
        action = predict_model(neural_net, env_pac_man)
        env_pac_man.act_with_action_id(action)
        b = env_pac_man.state_description()[:-8]
        for n, s in enumerate(b):
            board[n] = str(int(s))
            if s > 10:
                board[n] = chr(int(s))
        for n, g in enumerate(env_pac_man.get_ghosts()):
            ghosts[n] = g
        create_line()

    def loop_simulate(x):
        simulate()
        canvas.update()
        canvas.update_idletasks()
        if x > 0:
            canvas.after(100, loop_simulate(x-1))

    canvas.bind('<Button-1>', click_square)
    create_line()

    vld_btn = tk.Button(text='Simuler', command=lambda: simulate())
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=SIMULATE_BTN_POS[0], y=SIMULATE_BTN_POS[1])

    vld_btn_loop = tk.Button(text='Simuler Loop', command=lambda: loop_simulate(500))
    vld_btn_loop.config(font=(FONT, '18'), width=13)
    vld_btn_loop.place(x=SIMULATE_BTN_POS[0] + 300, y=SIMULATE_BTN_POS[1])


def tic_tac_toe(algo: int, canvas):
    is_started = False
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]

    def is_terminated() -> bool:
        t_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        one_empty = False
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 2:
                    t_grid[i][j] = 1
                elif grid[i][j] == 3:
                    t_grid[i][j] = 10
                elif grid[i][j] == 1:
                    one_empty = True

        r1 = t_grid[0][0] + t_grid[0][1] + t_grid[0][2]
        r2 = t_grid[1][0] + t_grid[1][1] + t_grid[1][2]
        r3 = t_grid[2][0] + t_grid[2][1] + t_grid[2][2]

        c1 = t_grid[0][0] + t_grid[1][0] + t_grid[2][0]
        c2 = t_grid[0][1] + t_grid[1][1] + t_grid[2][1]
        c3 = t_grid[0][2] + t_grid[1][2] + t_grid[2][2]

        d1 = t_grid[0][0] + t_grid[1][1] + t_grid[2][2]
        d2 = t_grid[0][2] + t_grid[1][1] + t_grid[2][0]

        if r1 == 3 or r2 == 3 or r3 == 3 or c1 == 3 or c2 == 3 or c3 == 3 or d1 == 3 or d2 == 3:
            return True
        if r1 == 30 or r2 == 30 or r3 == 30 or c1 == 30 or c2 == 30 or c3 == 30 or d1 == 30 or d2 == 30:
            return True
        return not one_empty

    def place_circle(x: int, y: int):
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 5:
                    grid[i][j] = 1
        if grid[x][y] == 1:
            grid[x][y] = 5

    def create_line():
        canvas.delete("all")
        canvas.create_line(5, 105, 305, 105)
        canvas.create_line(5, 205, 305, 205)
        canvas.create_line(105, 5, 105, 305)
        canvas.create_line(205, 5, 205, 305)

        canvas_places = (620, 210)

        for i in range(0, 3):
            for j in range(0, 3):
                if grid[i][j] == 2 or grid[i][j] == 5:
                    canvas.create_line(10 + i * 100, 10 + j * 100, 100 + i * 100, 100 + j * 100, width=3)
                    canvas.create_line(100 + i * 100, 10 + j * 100, 10 + i * 100, 100 + j * 100, width=3)
                elif grid[i][j] == 3:
                    canvas.create_oval(10 + i * 100, 10 + j * 100, 100 + i * 100, 100 + j * 100, outline="black",
                                       width=3)
        canvas.place(x=canvas_places[0], y=canvas_places[1])

    def click_square(event=None):
        if 5 <= event.y <= 305 and 5 <= event.x <= 305 and not is_terminated():
            i = 0
            j = 0

            while event.x > 5 + 100 * i and i <= 3:
                i += 1
            i -= 1

            while event.y > 5 + 100 * j and j <= 3:
                j += 1
            j -= 1

            place_circle(i, j)
            create_line()

    def simulate():
        n = ''
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[j][i] == 5:
                    grid[j][i] = 2

                if grid[j][i] == 2 or grid[j][i] == 5:
                    board[i][j] = 1
                elif grid[j][i] == 3:
                    board[i][j] = 10
                n += str(grid[j][i])

        if not is_terminated():
            if algo > 8:
                neural_net = load_neuralnet(3, algo)
                print(board)
                print(np.array(board).flatten())
                env_tic_tact_toe.set_state(np.array(board).flatten())
                action = predict_model(neural_net, env_tic_tact_toe)
            else:
                json_data = import_json(3, algo)
                action = json_data[n]

            x = action % 3
            y = action // 3
            grid[x][y] = 3

            create_line()

    simulate()
    canvas.bind('<Button-1>', click_square)
    create_line()

    vld_btn = tk.Button(text='Simuler', command=lambda: simulate())
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=SIMULATE_BTN_POS[0], y=SIMULATE_BTN_POS[1])


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

    def create_line():
        canvas.delete("all")
        for i in range(0, 5):
            for j in range(0, 5):
                fill = 'white'
                if grid[i][j] == -1:
                    fill = 'red'
                elif grid[i][j] == 100:
                    fill = 'blue'
                elif grid[i][j] == 1:
                    fill = '#47d147'
                elif grid[i][j] == 2:
                    fill = '#70db70'
                canvas.create_rectangle(5 + i * 100, 5 + j * 100, 105 + i * 100, 105 + j * 100, outline='black',
                                        fill=fill)
        canvas.place(x=450, y=100)

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


def line_world(algo: int, canvas):
    is_started = False
    grid = [(5, -1), (105, 0), (205, 0), (305, 0), (405, 0), (505, 0), (605, 100)]

    def set_start_position(x: int):
        for i in range(0, len(grid)):
            grid[i] = (grid[i][0], 0 if grid[i][1] == 1 else grid[i][1])
        grid[x] = (grid[x][0], 1)

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
        canvas.place(x=350, y=310)

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


def validate(game: int, algo: int, canvas):
    for c in CANVAS:
        c.delete('all')
    global nerual_net
    nerual_net = None
    if game == 0 or algo == 0:
        messagebox.showinfo('Erreur', 'Veuillez choisir un jeu et un algorithme')

    if (game == 3 and algo in (1, 2)) or (game == 4 and algo not in (9, 10, 11)):
        messagebox.showinfo("Erreur", "Ce couple jeu/algorithme n'est pas disponible")
        return

    if game == 1:
        line_world(algo, canvas)
    elif game == 2:
        grid_world(algo, canvas)
    elif game == 3:
        tic_tac_toe(algo, canvas)
    elif game == 4:
        pac_man(algo, canvas)


if __name__ == "__main__":
    window = tk.Tk()
    window.title('Best projet <3')
    window.geometry('1280x920')

    main_label = tk.Label(text='Bienvenu sur le meilleur Projet DRL !')
    main_label.config(font=(FONT, '26'))
    main_label.pack()

    game_label = tk.Label(text='Choisissez le jeu :')
    game_label.config(font=(FONT, '16'))
    game_label.place(x=0, y=75)

    game = tk.IntVar()

    for x, (g, v) in enumerate(GAMES):
        rd = tk.Radiobutton(text=g, padx=20, variable=game, value=v)
        rd.config(font=(FONT, '12'))
        rd.place(x=0, y=75 + (x + 1) * 40)

    algo_label = tk.Label(text="Choisissez l'algorithme :")
    algo_label.config(font=(FONT, '16'))
    algo_label.place(x=0, y=280)

    algo = tk.IntVar()

    for x, (g, v) in enumerate(ALGOS):
        rd = tk.Radiobutton(text=g, padx=20, variable=algo, value=v)
        rd.config(font=(FONT, '12'))
        rd.place(x=0, y=280 + (x + 1) * 40)

    canvas = tk.Canvas(width=1000, height=1000)
    vld_btn = tk.Button(text='Valider', command=lambda: validate(game.get(), algo.get(), canvas))
    vld_btn.config(font=(FONT, '18'), width=13)
    vld_btn.place(x=10, y=800)

    window.mainloop()
