import random
import numpy as np

from drl_sample_project_python.do_not_touch.contracts import MDPEnv
from drl_sample_project_python.do_not_touch.result_structures import PolicyAndValueFunction


def get_policy_iteration(env: MDPEnv, gamma: float, theta: float) -> PolicyAndValueFunction:
    S = env.states()
    A = env.actions()
    R = env.rewards()
    V = {}
    pi = {}

    for s in S:
        pi[s] = {}
        V[s] = 0.0
        for a in A:
            pi[s][a] = 0.0
        pi[s][random.randint(0, len(A) - 1)] = 1.0

    while True:
        while True:
            delta = 0.0
            for s in S:
                v = V[s]
                V[s] = 0.0
                for a in A:
                    for s_p in S:
                        for r_idx, r in enumerate(R):
                            V[s] += pi[s][a] * env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        policy_stable = True
        for s in S:
            old_state_policy = {}
            for a in list(pi[s].keys()):
                old_state_policy[a] = pi[s][a]

            best_a = -1
            best_a_score = None

            for a in A:
                a_score = 0.0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score
            for a in A:
                pi[s][a] = 0.0
            pi[s][best_a] = 1.0
            if not np.array_equal(old_state_policy, pi[s]):
                policy_stable = False
        if policy_stable:
            break
    return PolicyAndValueFunction(pi=pi, v=V)
