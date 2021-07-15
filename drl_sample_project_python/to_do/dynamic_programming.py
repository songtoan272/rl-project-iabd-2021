from drl_sample_project_python.do_not_touch.mdp_env_wrapper import Env1
from drl_sample_project_python.do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from drl_sample_project_python.envs.line_world_mdp_definition import EnvLineWorldMDP
from drl_sample_project_python.envs.grid_world_mdp_definition import EnvGridWorldMDP
from drl_sample_project_python.algos.policy_evaluation import get_policy_evaluation
from drl_sample_project_python.algos.policy_iteration import get_policy_iteration
from drl_sample_project_python.algos.value_iteration import get_value_iteration
from drl_sample_project_python.main import export_to_json

line_world = EnvLineWorldMDP(7)
grid_world = EnvGridWorldMDP(5, 5, (4, 4), (0, 0))
secret_env = Env1()


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    return get_policy_evaluation(line_world, 0.9999, 0.0001)


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    result = get_policy_iteration(line_world, 0.99, 0.01)
    export_to_json(result.pi, 'policy_iteration_line_world')
    return result


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    result = get_value_iteration(line_world, 0.99, 0.01)
    export_to_json(result.pi, 'value_iteration_line_world')
    return result


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    return get_policy_evaluation(grid_world, 0.9999, 0.0001)


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    result = get_policy_iteration(grid_world, 0.99, 0.01)
    export_to_json(result.pi, 'policy_iteration_grid_world')
    return result


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    result = get_value_iteration(grid_world, 0.99, 0.01)
    export_to_json(result.pi, 'value_iteration_grid_world')
    return result


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    return get_policy_evaluation(secret_env, 0.9999, 0.0001)


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    result = get_policy_iteration(secret_env, 0.99, 0.01)
    export_to_json(result.pi, 'policy_iteration_secret_env_1')
    return result


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    result = get_value_iteration(secret_env, 0.99, 0.01)
    export_to_json(result.pi, 'value_iteration_secret_env_1')
    return result


def demo():
    print('\n\nPolicy evaluation : Line World\n')
    print(policy_evaluation_on_line_world())
    print('\n\nPolicy iteration : Line World\n')
    print(policy_iteration_on_line_world())
    print('\n\nValue iteration : Line World\n')
    print(value_iteration_on_line_world())

    print('\n\nPolicy evaluation : Grid World\n')
    print(policy_evaluation_on_grid_world())
    print('\n\nPolicy iteration : Grid World\n')
    print(policy_iteration_on_grid_world())
    print('\n\nValue iteration : Grid World\n')
    print(value_iteration_on_grid_world())

    print('\n\nPolicy evaluation : Secret Env 1\n')
    print(policy_evaluation_on_secret_env1())
    print('\n\nPolicy iteration : Secret Env 1\n')
    print(policy_iteration_on_secret_env1())
    print('\n\nValue iteration : Secret Env 1\n')
    print(value_iteration_on_secret_env1())
