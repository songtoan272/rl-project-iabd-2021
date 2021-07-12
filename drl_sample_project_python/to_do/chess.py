from drl_sample_project_python.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_sample_project_python.do_not_touch.single_agent_env_wrapper import Env3
from drl_sample_project_python.envs.env_grid_world_single_agent import EnvGridWorldSingleAgent
from drl_sample_project_python.envs.env_line_world_single_agent import EnvLineWorldSingleAgent
from drl_sample_project_python.envs.env_tictactoe_single_agent import EnvTicTacToeSingleAgent
from drl_sample_project_python.envs.env_chess_single_agent import EnvChessSingleAgent
from drl_sample_project_python.algos.q_learning import get_q_learning
from drl_sample_project_python.algos.sarsa import get_sarsa
from drl_sample_project_python.algos.expected_sarsa import get_expected_sarsa

max_steps = 20
line_world = EnvLineWorldSingleAgent(7, max_steps)
grid_world = EnvGridWorldSingleAgent(5, 5, max_steps, (4, 4), (0, 0))
tic_tac_toe = EnvTicTacToeSingleAgent(max_steps)
chess_env = EnvChessSingleAgent(max_steps)


def sarsa_on_chess() -> PolicyAndActionValueFunction:
    # return get_sarsa(tic_tac_toe, 0.1, 1.0, 0.9, 1000)
    return get_sarsa(chess_env, 0.1, 1.0, 0.9, 20)


def demo():
    p = sarsa_on_chess()
    # print(p.pi, p.q)

    #available_actions = chess_env.available_actions_ids()
    #print(available_actions)

