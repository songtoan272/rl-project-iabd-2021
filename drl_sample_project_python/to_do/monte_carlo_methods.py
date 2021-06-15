from drl_sample_project_python.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_sample_project_python.do_not_touch.single_agent_env_wrapper import Env2
from drl_sample_project_python.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.to_do.single_agent.env_tictactoe_single_agent import EnvTicTacToeSingleAgent
import numpy as np
from pprint import PrettyPrinter


def monte_carlo_es(env: SingleAgentEnv,
                   gamma: float,
                   max_iter: int) -> PolicyAndActionValueFunction:
    pi = {}  # dict(int) as pi is deterministic
    q = {}  # dict(dict(float))
    returns = {}  # dict(dict(int)) count of visits

    for _ in range(max_iter):
        env.reset_random()

        # Generate an episode from starting state
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = np.random.choice(available_actions, 1, False)[0]
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    q[s][a] = 0.0
                    returns[s][a] = 0

            chosen_action = pi[s]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)
        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for s, a in zip(S[:t], A[:t]):
                if s_t == s and a_t == a:
                    found = True
                    break
            if not found:
                q[s_t][a_t] = (q[s_t][a_t] * returns[s_t][a_t] + G) / (returns[s_t][a_t] + 1)
                returns[s_t][a_t] += 1
                pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]

    return PolicyAndActionValueFunction(pi, q)


def on_policy_first_visit_monte_carlo_control(
        env: SingleAgentEnv,
        gamma: float,
        eps: float,
        max_iter: int
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert(eps > 0)
    pi = {}     # dict(dict(float)) = probability to choose an action a from state s
    q = {}      # dict(dict(float))
    returns = {}# dict(dict(int)) = nb of times choosing a from s

    for _ in range(max_iter):
        env.reset()

        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = 0

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for s, a in zip(S[:t], A[:t]):
                if s_t == s and a_t == a:
                    found = True
                    break
            if not found:
                q[s_t][a_t] = (q[s_t][a_t] * returns[s_t][a_t] + G) / (returns[s_t][a_t] + 1)
                returns[s_t][a_t] += 1
                optimal_s_a = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
                available_action_t_counts = len(q[s_t])
                for a in q[s_t].keys():
                    if a == optimal_s_a:
                        pi[s_t][a] = 1 - eps + eps/available_action_t_counts
                    else:
                        pi[s_t][a] = eps / available_action_t_counts

    return PolicyAndActionValueFunction(pi, q)


def off_policy_monte_carlo_control(
        env: SingleAgentEnv,
        gamma: float,
        max_iter: int
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    q = {}
    c = {}
    pi = {}     # dict(int) deterministic target policy

    def random_soft_policy_for_state(env: SingleAgentEnv, b: dict[dict[float]]):
        rng = np.random.default_rng()
        if env.state_id() not in b:
            b[env.state_id()] = {}
            b_s = b[env.state_id()]
            available_actions = env.available_actions_ids()
            nb_of_actions = len(available_actions)
            proba_s = rng.integers(1, nb_of_actions, nb_of_actions, endpoint=True).astype(float)
            proba_s /= sum(proba_s)
            for id_a, a in enumerate(available_actions):
                b_s[a] = proba_s[id_a]
            # print(b_s.values())

    for _ in range(max_iter):
        env.reset()
        b = {}  # dict(dict(float)) soft behavior policy
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            random_soft_policy_for_state(env, b)
            available_actions = env.available_actions_ids()
            if s not in q:
                q[s] = {}
                c[s] = {}
                for a in available_actions:
                    q[s][a] = 0.0
                    c[s][a] = 0.0
            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0.0
        W = 1.0
        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            G = gamma * G + R[t]
            c[s_t][a_t] += W
            q[s_t][a_t] += (W / c[s_t][a_t] * (G - q[s_t][a_t]))
            pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
            if a_t == pi[s_t]:
                break
            W = W / b[s_t][a_t]
    return PolicyAndActionValueFunction(pi, q)


def monte_carlo_es_on_tic_tac_toe_solo(
        env: EnvTicTacToeSingleAgent = None,
        gamma: float = 0.9999,
        max_iter: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = EnvTicTacToeSingleAgent() if env is None else env
    return monte_carlo_es(env, gamma=gamma, max_iter=max_iter)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(
        env: EnvTicTacToeSingleAgent = None,
        gamma: float = 0.9999,
        eps: float = 0.1,
        max_iter: int = 10000
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert(eps > 0)
    env = EnvTicTacToeSingleAgent() if env is None else env
    return on_policy_first_visit_monte_carlo_control(env, gamma, eps, max_iter)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo(
        env: EnvTicTacToeSingleAgent = None,
        gamma: float = 0.9999,
        eps: float = 0.1,
        max_iter: int = 10000
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = EnvTicTacToeSingleAgent() if env is None else env
    return off_policy_monte_carlo_control(env, gamma, max_iter)


def monte_carlo_es_on_secret_env2(
        env: EnvTicTacToeSingleAgent = None,
        gamma: float = 0.9999,
        max_iter: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2() if env is None else env
    return monte_carlo_es(env, gamma=gamma, max_iter=max_iter)



def on_policy_first_visit_monte_carlo_control_on_secret_env2(
        env: EnvTicTacToeSingleAgent = None,
        gamma: float = 0.9999,
        eps: float = 0.1,
        max_iter: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2() if env is None else env
    return on_policy_first_visit_monte_carlo_control(env, gamma=gamma, eps=eps, max_iter=max_iter)


def off_policy_monte_carlo_control_on_secret_env2(
        env: EnvTicTacToeSingleAgent = None,
        gamma: float = 0.9999,
        eps: float = 0.1,
        max_iter: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2() if env is None else env
    return off_policy_monte_carlo_control(env, gamma=gamma, max_iter=max_iter)


def demo():
    # print(monte_carlo_es_on_tic_tac_toe_solo())
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    print(monte_carlo_es_on_secret_env2())
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print(off_policy_monte_carlo_control_on_secret_env2())

if __name__ == "__main__":
    demo()