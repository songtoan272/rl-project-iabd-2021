import drl_sample_project_python.main
from drl_sample_project_python.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_sample_project_python.do_not_touch.single_agent_env_wrapper import Env2
from drl_sample_project_python.envs.env_tictactoe_single_agent import EnvTicTacToeSingleAgent
from drl_sample_project_python.envs.env_line_world_single_agent import EnvLineWorldSingleAgent
from drl_sample_project_python.envs.env_grid_world_single_agent import EnvGridWorldSingleAgent
from drl_sample_project_python.algos.monte_carlo_es import monte_carlo_es
from drl_sample_project_python.algos.off_policy_monte_carlo_control import off_policy_monte_carlo_control
from drl_sample_project_python.algos.on_policy_first_visit_monte_carlo_control import on_policy_first_visit_monte_carlo_control


max_iter = 20000
nb_generation = 1
tic_tac_toe = EnvTicTacToeSingleAgent(200)
secret_env = Env2()


def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    result = monte_carlo_es(tic_tac_toe, 0.9999, max_iter)
    for _ in range(0, nb_generation):
        result = monte_carlo_es(EnvTicTacToeSingleAgent(100, second_pi=result.pi), 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'monte_carlo_es_tic_tac_toe')
    return result


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = on_policy_first_visit_monte_carlo_control(tic_tac_toe, 0.9999, 0.1, max_iter)
    for _ in range(0, nb_generation):
        result = on_policy_first_visit_monte_carlo_control(EnvTicTacToeSingleAgent(100, second_pi=result.pi), 0.9999, 0.1, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'on_policy_monte_carlo_tic_tac_toe')
    return result


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = off_policy_monte_carlo_control(tic_tac_toe, 0.9999, max_iter)
    for _ in range(0, nb_generation):
        result = off_policy_monte_carlo_control(EnvTicTacToeSingleAgent(100, second_pi=result.pi), 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'off_policy_monte_carlo_tic_tac_toe')
    return result


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    result = monte_carlo_es(secret_env, 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'monte_carlo_es_secret_env_2')
    return result


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = on_policy_first_visit_monte_carlo_control(secret_env, 0.9999, 0.1, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'on_policy_monte_carlo_secret_env_2')
    return result


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = off_policy_monte_carlo_control(secret_env, 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'off_policy_monte_carlo_secret_env_2')
    return result


def monte_carlo_es_line_world():
    result = monte_carlo_es(EnvLineWorldSingleAgent(7, 1000), 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'monte_carlo_es_line_world')


def on_policy_first_visit_monte_carlo_control_on_line_world():
    result = on_policy_first_visit_monte_carlo_control(EnvLineWorldSingleAgent(7, 1000), 0.9999, 0.1, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'on_policy_monte_carlo_line_world')


def off_policy_monte_carlo_control_on_line_world():
    result = off_policy_monte_carlo_control(EnvLineWorldSingleAgent(7, 1000), 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'off_policy_monte_carlo_line_world')


def monte_carlo_es_grid_world():
    result = monte_carlo_es(EnvGridWorldSingleAgent(5, 5, 1000, (4, 4), (0, 0)), 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'monte_carlo_es_grid_world')


def on_policy_first_visit_monte_carlo_control_on_grid_world():
    result = on_policy_first_visit_monte_carlo_control(EnvGridWorldSingleAgent(5, 5, 1000, (4, 4), (0, 0)), 0.9999, 0.1, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'on_policy_monte_carlo_grid_world')


def off_policy_monte_carlo_control_on_grid_world():
    result = off_policy_monte_carlo_control(EnvGridWorldSingleAgent(5, 5, 1000, (4, 4), (0, 0)), 0.9999, max_iter)
    drl_sample_project_python.main.export_to_json(result.pi, 'off_policy_monte_carlo_grid_world')

def demo():
    print('\n\nMonte Carlo ES : Tic Tac Toe\n')
    print(monte_carlo_es_on_tic_tac_toe_solo())
    print('\n\nOn policy Monte Carlo : Tic Tac Toe\n')
    print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    print('\n\nOff policy Monte Carlo : Tic Tac Toe\n')
    print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    print('\n\nMonte Carlo ES : Secret Env 2\n')
    #print(monte_carlo_es_on_secret_env2())
    print('\n\nOn policy Monte Carlo : Secret Env 2\n')
    #print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print('\n\nOff policy Monte Carlo: Secret Env 2\n')
    #print(off_policy_monte_carlo_control_on_secret_env2())

    #monte_carlo_es_line_world()
    #on_policy_first_visit_monte_carlo_control_on_line_world()
    #off_policy_monte_carlo_control_on_line_world()
    #monte_carlo_es_grid_world()
    #on_policy_first_visit_monte_carlo_control_on_grid_world()
    #off_policy_monte_carlo_control_on_grid_world()