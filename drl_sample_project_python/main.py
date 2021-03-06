import random
import json
import operator

import numpy as np
import tensorflow as tf

import to_do.dynamic_programming as dynamic_programming
import to_do.monte_carlo_methods as monte_carlo_methods
import to_do.temporal_difference_learning as temporal_difference_learning
import to_do.deep_rl as deep_rl
import to_do.chess as chess
from drl_sample_project_python.envs.env_grid_world_deep_single_agent import EnvGridWorldDeepSingleAgent


def export_to_json(pi, file_name: str):
    action_by_state = {int(str(k)): int(max(v.items(), key=operator.itemgetter(1))[0]) if type(v) is dict else int(str(v)) for k, v in pi.items()}

    pi_json = json.dumps(action_by_state)
    f = open('./models/' + file_name + '.json', 'w')
    f.write(pi_json)
    f.close()


def save_neural_net(neural_net, name: str):
    neural_net.save('./models/' + name + '.h5')


if __name__ == "__main__":
    #print('\n\n\nDynamic\n\n\n')
    #dynamic_programming.demo()
    #print('\n\n\nmonte_carlo\n\n\n')
    #monte_carlo_methods.demo()
    #print('\n\n\ntemporal\n\n\n')
    #temporal_difference_learning.demo()
    print('\n\n\nDeep RL\n\n\n')
    deep_rl.demo()
