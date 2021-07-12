import numpy as np
from tqdm import *

from drl_sample_project_python.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.do_not_touch.result_structures import PolicyAndActionValueFunction


def get_expected_sarsa(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_iter: int) -> PolicyAndActionValueFunction:
    assert(epsilon > 0)

    pi = {}
    b = {}
    q = {}

    for it in tqdm(range(max_iter)):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                b[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    b[s][a] = 1.0 / len(available_actions)

            available_actions_count = len(available_actions)
            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1 - epsilon + epsilon / available_actions_count
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(list(b[s].keys()), 1, False, p=list(b[s].values()))[0]
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            s_p = env.state_id()
            next_available_action = env.available_actions_ids()

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    q[s_p] = {}
                    b[s_p] = {}
                    for a in next_available_action:
                        pi[s_p][a] = 1.0 / len(next_available_action)
                        q[s_p][a] = 0.0
                        b[s_p][a] = 1.0 / len(next_available_action)

                somme = 0.0
                for a in next_available_action:
                    somme += pi[s_p][a] * q[s_p][a] - q[s][chosen_action]
                q[s][chosen_action] += alpha * (r + gamma * somme)

    for s in q.keys():
        optimal_a_t = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a_t:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    return PolicyAndActionValueFunction(pi=pi, q=q)
