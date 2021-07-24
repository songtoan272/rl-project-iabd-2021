import drl_sample_project_python.main
from drl_sample_project_python.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_sample_project_python.do_not_touch.single_agent_env_wrapper import Env3
from drl_sample_project_python.envs.env_grid_world_single_agent import EnvGridWorldSingleAgent
from drl_sample_project_python.envs.env_line_world_single_agent import EnvLineWorldSingleAgent
from drl_sample_project_python.envs.env_tictactoe_single_agent import EnvTicTacToeSingleAgent
from drl_sample_project_python.algos.q_learning import get_q_learning
from drl_sample_project_python.algos.sarsa import get_sarsa
from drl_sample_project_python.algos.expected_sarsa import get_expected_sarsa


max_iter = 200000
scale = int(max_iter / 100)
tic_tac_toe = EnvTicTacToeSingleAgent(100)
secret_env = Env3()


def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    file_name = 'sarsa_tic_tac_toe'
    result = get_sarsa(tic_tac_toe, 0.8, 0.1, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)
    return result


def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    file_name = 'q_learning_tic_tac_toe'
    result = get_q_learning(tic_tac_toe, 0.8, 0.1, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)
    return result


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    file_name = 'expected_sarsa_tic_tac_toe'
    result = get_expected_sarsa(tic_tac_toe, 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)
    return result


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    file_name = 'sarsa_secret_env_3'
    result = get_sarsa(secret_env, 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)
    return result


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    file_name = 'q_learning_secret_env_3'
    result = get_q_learning(secret_env, 0.1, 1.0, 0.9, max_iter,file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)
    return result


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    file_name = 'expected_sarsa_secret_env_3'
    result = get_expected_sarsa(secret_env, 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)
    return result


def sarsa_on_line_world():
    file_name = 'sarsa_line_world'
    result = get_sarsa(EnvLineWorldSingleAgent(7, 1000), 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)


def q_learning_on_line_world():
    file_name = 'q_learning_line_world'
    result = get_q_learning(EnvLineWorldSingleAgent(7, 1000), 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)


def expected_sarsa_on_line_world():
    file_name = 'expected_sarsa_line_world'
    result = get_expected_sarsa(EnvLineWorldSingleAgent(7, 1000), 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)


def sarsa_on_grid_world():
    file_name = 'sarsa_grid_world'
    result = get_sarsa(EnvGridWorldSingleAgent(5, 5, 1000, (4, 4), (0, 0)), 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)


def q_learning_on_grid_world():
    file_name = 'q_learning_grid_world'
    result = get_q_learning(EnvGridWorldSingleAgent(5, 5, 1000, (4, 4), (0, 0)), 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)


def expected_sarsa_on_grid_world():
    file_name = 'expected_sarsa_grid_world'
    result = get_expected_sarsa(EnvGridWorldSingleAgent(5, 5, 1000, (4, 4), (0, 0)), 0.1, 1.0, 0.9, max_iter, file_name, scale)
    drl_sample_project_python.main.export_to_json(result.pi, file_name, result.q)


def demo():
    #print('\n\nSarsa : Tic Tac Toe\n')
    #print(sarsa_on_tic_tac_toe_solo())
    #print('\n\nQ learning : Tic Tac Toe\n')
    #print(q_learning_on_tic_tac_toe_solo())
    #print('\n\nExpected sarsa : Tic Tac Toe\n')
    #print(expected_sarsa_on_tic_tac_toe_solo())

    print('\n\nSarsa : Secret Env 3\n')
    print(sarsa_on_secret_env3())
    print('\n\nQ learning : Secret Env 3\n')
    print(q_learning_on_secret_env3())
    print('\n\nExpected sarsa : Secret Env 3\n')
    print(expected_sarsa_on_secret_env3())

    #sarsa_on_line_world()
    #q_learning_on_line_world()
    #expected_sarsa_on_line_world()
    #sarsa_on_grid_world()
    #q_learning_on_grid_world()
    #expected_sarsa_on_grid_world()
