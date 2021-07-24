import drl_sample_project_python.main
from drl_sample_project_python.envs.env_line_world_deep_single_agent import EnvLineWorldDeepSingleAgent
from drl_sample_project_python.envs.env_grid_world_deep_single_agent import EnvGridWorldDeepSingleAgent
from drl_sample_project_python.envs.env_tictactoe_deep_single_agent import EnvTicTacToeDeepSingleAgent
from drl_sample_project_python.envs.env_pac_man_deep_single_agent import EnvPacManDeepSingleAgent
from drl_sample_project_python.algos.episodic_semi_gradient_sarsa import get_episodic_semi_gradient_sarsa
from drl_sample_project_python.algos.deep_q_learning import get_deep_q_learning
from drl_sample_project_python.algos.reinforce_monte_carlo_policy_gradient_control import get_reinforce_monte_carlo_policy_gradient_control
from drl_sample_project_python.algos.reinforce_with_baseline import get_reinforce_with_baseline

import tensorflow as tf


tf.config.set_visible_devices([], 'GPU')
max_iter = 100000
max_steps = 30
line_world = EnvLineWorldDeepSingleAgent(7, max_steps)
grid_world = EnvGridWorldDeepSingleAgent(5, 5, max_steps, (4, 4), (0, 0))
tic_tac_toe = EnvTicTacToeDeepSingleAgent(max_steps)
pac_man = EnvPacManDeepSingleAgent(max_steps, '././models/pac_man_level_custom_2.txt')


def episodic_semi_gradient_sarsa_on_line_world():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(line_world.state_description_length() + line_world.max_actions_count())),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.9
    epsilon = 0.2
    model_name = 'episodic_semi_gradient_sarsa_line_world'
    get_episodic_semi_gradient_sarsa(line_world, gamma, epsilon, max_iter, q, model_name)


def episodic_semi_gradient_sarsa_on_grid_world():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                              input_dim=(grid_world.state_description_length() + grid_world.max_actions_count())),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.9
    epsilon = 0.2
    model_name = 'episodic_semi_gradient_sarsa_grid_world'
    get_episodic_semi_gradient_sarsa(grid_world, gamma, epsilon, max_iter, q, model_name)


def episodic_semi_gradient_sarsa_on_tic_tac_toe():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh,
                              input_dim=(tic_tac_toe.state_description_length() + tic_tac_toe.max_actions_count())),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.9
    epsilon = 1.0
    model_name = 'episodic_semi_gradient_sarsa_tic_tac_toe'
    get_episodic_semi_gradient_sarsa(tic_tac_toe, gamma, epsilon, max_iter, q, model_name)


def episodic_semi_gradient_sarsa_on_pac_man():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh,
                              input_dim=(pac_man.state_description_length() + pac_man.max_actions_count())),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.999
    epsilon = 1.0
    model_name = 'episodic_semi_gradient_sarsa_pac_man'
    get_episodic_semi_gradient_sarsa(pac_man, gamma, epsilon, max_iter, q, model_name)


def deep_q_learning_line_world():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(line_world.state_description_length() + line_world.max_actions_count())),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.9
    epsilon = 1.0
    model_name = 'deep_q_learning_line_world'
    get_deep_q_learning(line_world, gamma, epsilon, max_iter, q, model_name)


def deep_q_learning_grid_world():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(grid_world.state_description_length() + grid_world.max_actions_count())),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.9
    epsilon = 0.3
    model_name = 'deep_q_learning_grid_world'
    get_deep_q_learning(grid_world, gamma, epsilon, max_iter, q, model_name)


def deep_q_learning_tic_tact_toe():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh,
                              input_dim=(tic_tac_toe.state_description_length() + tic_tac_toe.max_actions_count())),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.9
    epsilon = 0.1
    model_name = 'deep_q_learning_tic_tac_toe'
    get_deep_q_learning(tic_tac_toe, gamma, epsilon, max_iter, q, model_name)


def deep_q_learning_on_pac_man():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=(pac_man.state_description_length() + pac_man.max_actions_count())),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh, kernel_initializer='random_normal'),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh, kernel_initializer='random_normal'),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh, kernel_initializer='random_normal'),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.8
    epsilon = 1.0
    model_name = 'deep_q_learning_pac_man'
    get_deep_q_learning(pac_man, gamma,epsilon, max_iter, q, model_name)


def reinforce_monte_carlo_line_world():
    pi = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=line_world.state_description_length()),
        tf.keras.layers.Dense(line_world.max_actions_count(), activation=tf.keras.activations.softmax),
    ])
    pi.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy)

    gamma = 0.7
    model_name = 'reinforce_monte_carlo_line_world'
    get_reinforce_monte_carlo_policy_gradient_control(line_world, gamma, max_iter, pi, model_name)


def reinforce_monte_carlo_grid_world():
    pi = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=grid_world.state_description_length()),
        tf.keras.layers.Dense(grid_world.max_actions_count(), activation=tf.keras.activations.softmax),
    ])
    pi.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.categorical_crossentropy)

    gamma = 0.7
    model_name = 'reinforce_monte_carlo_grid_world'
    get_reinforce_monte_carlo_policy_gradient_control(grid_world, gamma, max_iter, pi, model_name)


def reinforce_with_baseline_line_world():
    pi = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=line_world.state_description_length()),
        tf.keras.layers.Dense(line_world.max_actions_count(), activation=tf.keras.activations.softmax),
    ])
    pi.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    v = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=line_world.state_description_length()),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    v.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.7
    model_name = 'reinforce_with_baseline_line_world'
    get_reinforce_with_baseline(line_world, gamma, max_iter, pi, v, model_name)


def reinforce_with_baseline_grid_world():
    pi = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=grid_world.state_description_length()),
        tf.keras.layers.Dense(grid_world.max_actions_count(), activation=tf.keras.activations.softmax),
    ])
    pi.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    v = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=grid_world.state_description_length()),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    v.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.7
    model_name = 'reinforce_with_baseline_grid_world'
    get_reinforce_with_baseline(grid_world, gamma, max_iter, pi, v, model_name)


def reinforce_with_baseline_tic_tac_toe():
    pi = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=tic_tac_toe.state_description_length()),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(tic_tac_toe.max_actions_count(), activation=tf.keras.activations.softmax),
    ])
    pi.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    v = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=tic_tac_toe.state_description_length()),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    v.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 0.7
    model_name = 'reinforce_with_baseline_tic_tac_toe'
    get_reinforce_with_baseline(tic_tac_toe, gamma, max_iter, pi, v, model_name)


def reinforce_with_baseline_pac_man():
    pi = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=pac_man.state_description_length()),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(pac_man.max_actions_count(), activation=tf.keras.activations.softmax),
    ])
    pi.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    v = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh, kernel_initializer='random_normal',
                              input_dim=pac_man.state_description_length()),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    v.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    gamma = 1.0
    model_name = 'reinforce_with_baseline_pac_man'
    get_reinforce_with_baseline(pac_man, gamma, max_iter, pi, v, model_name)


def demo():
    #print('\n\nEpisodic semi gradient sarsa : Line World\n')
    #episodic_semi_gradient_sarsa_on_line_world()
    #print('\n\nEpisodic semi gradient sarsa : Grid World\n')
    #episodic_semi_gradient_sarsa_on_grid_world()
    #print('\n\nEpisodic semi gradient sarsa : Tic Tac Toe\n')
    #episodic_semi_gradient_sarsa_on_tic_tac_toe()
    #print('\n\nEpisodic semi gradient sarsa : Pac Man\n')
    #episodic_semi_gradient_sarsa_on_pac_man()

    #print('\n\nDeep Q-Learning : Line World\n')
    #deep_q_learning_line_world()
    #print('\n\nDeep Q-Learning : Grid World\n')
    #deep_q_learning_grid_world()
    #print('\n\nDeep Q-Learning : Tic Tac Toe\n')
    #deep_q_learning_tic_tact_toe()
    #print('\n\nDeep Q-Learning : Pax Man\n')
    #deep_q_learning_on_pac_man()

    #print('\n\nREINFORCE Monte-Carlo : Line World\n')
    #reinforce_monte_carlo_line_world()
    #print('\n\nREINFORCE Monte-Carlo : Grid World\n')
    #reinforce_monte_carlo_grid_world()

    #print('\n\nREINFORCE with Baseline : Line World\n')
    #reinforce_with_baseline_line_world()
    #print('\n\nREINFORCE with Baseline : Grid World\n')
    #reinforce_with_baseline_grid_world()
    #print('\n\nREINFORCE with Baseline : TicTacToe\n')
    #reinforce_with_baseline_tic_tac_toe()
    print('\n\nREINFORCE with Baseline : PacMan\n')
    reinforce_with_baseline_pac_man()
