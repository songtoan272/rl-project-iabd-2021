import random

from drl_sample_project_python.do_not_touch.contracts import MDPEnv
from drl_sample_project_python.do_not_touch.result_structures import PolicyAndValueFunction


def get_value_iteration(env: MDPEnv, gamma: float, theta: float) -> PolicyAndValueFunction:
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
        delta = 0.0
        for s in S:
            v = V[s]
            max_a = -1.0

            best_a = -1
            best_a_score = None

            for a in A:
                a_score = 0.0
                for s_p in S:
                    for i, r in enumerate(R):
                        t = env.transition_probability(s, a, s_p, i) * (r + gamma * V[s_p])
                        a_score += t
                        if t > max_a:
                            max_a = t
                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score

            for a in A:
                pi[s][a] = 0.0
            pi[s][best_a] = 1.0
            V[s] = max_a
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return PolicyAndValueFunction(pi=pi, v=V)
