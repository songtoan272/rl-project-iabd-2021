import drl_sample_project_python.main
from drl_sample_project_python.do_not_touch.single_agent_env_wrapper import Env3
from drl_sample_project_python.envs.env_line_world_deep_single_agent import EnvLineWorldDeepSingleAgent
from drl_sample_project_python.envs.env_grid_world_deep_single_agent import EnvGridWorldDeepSingleAgent
from drl_sample_project_python.algos.episodic_semi_gradient_sarsa import get_episodic_semi_gradient_sarsa

import tensorflow as tf
import numpy as np


max_iter = 100
max_steps = 100
line_world = EnvLineWorldDeepSingleAgent(7, max_steps)
grid_world = EnvGridWorldDeepSingleAgent(5, 5, max_steps, (4, 4), (0, 0))


def episodic_semi_gradient_sarsa_on_line_world():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(line_world.state_description_length() + line_world.max_actions_count())),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    pre_warm = 10
    epsilon = 0.1
    gamma = 0.9
    nn = get_episodic_semi_gradient_sarsa(line_world, pre_warm, epsilon, gamma, max_iter, q)
    drl_sample_project_python.main.save_neural_net(nn, 'episodic_semi_gradient_sarsa_line_world')
    return nn


def episodic_semi_gradient_sarsa_on_grid_world():
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(grid_world.state_description_length() + grid_world.max_actions_count())),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    pre_warm = 10
    epsilon = 0.1
    gamma = 0.9
    nn = get_episodic_semi_gradient_sarsa(grid_world, pre_warm, epsilon, gamma, max_iter, q)
    drl_sample_project_python.main.save_neural_net(nn, 'episodic_semi_gradient_sarsa_grid_world')
    return nn


def demo():
    #print('\n\nEpisodic semi gradient sarsa : Line World\n')
    #episodic_semi_gradient_sarsa_on_line_world()
    print('\n\nEpisodic semi gradient sarsa : Grid World\n')
    episodic_semi_gradient_sarsa_on_grid_world()

