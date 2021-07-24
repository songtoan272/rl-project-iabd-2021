import numpy as np
from tqdm import *

import drl_sample_project_python.main
from drl_sample_project_python.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.do_not_touch.result_structures import PolicyAndActionValueFunction


def init_state(env, s, pi, q):
    available_actions = env.available_actions_ids()
    if s not in pi:
        pi[s] = {}
        q[s] = {}
        for a in available_actions:
            pi[s][a] = 1.0 / len(available_actions)
            q[s][a] = 0.0
    return pi, q


def get_sarsa(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_iter: int, plot_name: str, scale: int=100) -> PolicyAndActionValueFunction:
    assert(epsilon > 0)

    scores = []
    average = 0.0
    iters = 0

    pi = {}
    q = {}

    for it in tqdm(range(max_iter)):
        epsilon = max(.02, epsilon * .999985)
        env.reset()
        s = env.state_id()
        pi, q = init_state(env, s, pi, q)

        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1 - epsilon + epsilon / len(env.available_actions_ids())
            else:
                pi[s][a_key] = epsilon / len(env.available_actions_ids())
        chosen_action = np.random.choice(list(pi[s].keys()), 1, False, p=list(pi[s].values()))[0]

        while not env.is_game_over():
            s = env.state_id()
            pi, q = init_state(env, s, pi, q)

            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            s_p = env.state_id()

            if not env.is_game_over():
                pi, q = init_state(env, s_p, pi, q)

                optimal_a_p = list(q[s_p].keys())[np.argmax(list(q[s_p].values()))]
                for a_key_p, q_s_a_p in q[s_p].items():
                    if a_key_p == optimal_a_p:
                        pi[s_p][a_key_p] = 1 - epsilon + epsilon / len(env.available_actions_ids())
                    else:
                        pi[s_p][a_key_p] = epsilon / len(env.available_actions_ids())
                chosen_action_p = np.random.choice(list(pi[s_p].keys()), 1, False, p=list(pi[s_p].values()))[0]

                q[s][chosen_action] += alpha * (r + gamma * q[s_p][chosen_action_p] - q[s][chosen_action])
                chosen_action = chosen_action_p
            else:
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])

        average = (average * iters + env.score()) / (iters + 1)
        iters += 1

        if it % scale == 0 and it != 0:
            scores.append(average)
            drl_sample_project_python.main.plot_scores(plot_name, scores, scale)
            average = 0.0
            iters = 0

    for s in q.keys():
        optimal_a_t = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a_t:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    return PolicyAndActionValueFunction(pi=pi, q=q)
