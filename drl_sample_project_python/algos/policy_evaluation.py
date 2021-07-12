import numpy as np

from drl_sample_project_python.do_not_touch.result_structures import ValueFunction
from drl_sample_project_python.do_not_touch.contracts import MDPEnv


def get_policy_evaluation(env: MDPEnv, gamma: float, theta: float) -> ValueFunction:
    S = env.states()
    A = env.actions()
    R = env.rewards()
    V = {}

    pi = np.zeros((len(env.states()), len(env.actions())))
    pi[:, :] = 1 / len(env.actions())

    for s in S:
        V[s] = 0.0

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0.0
            for a in A:
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V