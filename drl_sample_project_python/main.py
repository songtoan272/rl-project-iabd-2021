import json
import operator
import random

import matplotlib.pyplot as plt

import numpy as np
import to_do.dynamic_programming as dynamic_programming
import to_do.temporal_difference_learning as temporal_difference_learning
import to_do.monte_carlo_methods as monte_carlo_methods
import to_do.deep_rl as deep_rl


def export_to_json(pi, file_name: str, q=None):
    action_by_state = {int(str(k)): int(max(v.items(), key=operator.itemgetter(1))[0]) if type(v) is dict else int(str(v)) for k, v in pi.items()}
    pi_json = json.dumps(action_by_state)
    f = open('./models/' + file_name + '.json', 'w')
    f.write(pi_json)
    f.close()

    print(q)
    if q is not None:
        decrypt_q = {int(str(k)): {int(str(k_p)): float(str(v_p)) for k_p, v_p in v.items()} for k, v in q.items()}
        q_json = json.dumps(decrypt_q)
        f = open('./models/' + file_name + '_q.json', 'w')
        f.write(q_json)
        f.close()


def save_neural_net(neural_net, name: str):
    neural_net.save('./models/' + name + '.h5')


def plot_scores(name: str, scores, scale, save=True):
    plt.title(name)
    plt.xlabel("Nombre de parties")
    plt.ylabel("Score moyen pour " + str(scale) + " parties")
    plt.plot(np.arange(1, len(scores) + 1) * scale, scores)
    if save:
        plt.savefig('./models/' + name + '_curve.png')
    plt.show()


if __name__ == "__main__":
    #print('\n\n\nDynamic\n\n\n')
    #dynamic_programming.demo()
    #print('\n\n\nMonte_carlo\n\n\n')
    #monte_carlo_methods.demo()
    #print('\n\n\nTemporal\n\n\n')
    #temporal_difference_learning.demo()
    print('\n\n\nDeep RL\n\n\n')
    deep_rl.demo()

