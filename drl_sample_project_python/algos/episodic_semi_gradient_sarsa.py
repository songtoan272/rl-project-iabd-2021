import tqdm

import drl_sample_project_python.main
from drl_sample_project_python.do_not_touch.contracts import DeepSingleAgentEnv
import tensorflow as tf
import numpy as np


def get_episodic_semi_gradient_sarsa(env: DeepSingleAgentEnv, gamma: float, epsilon: float,
                                     max_iters: int, neural_net, model_name: str):
    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    scores = []
    average = 0.0
    iters = 0
    scale = 10

    for episode_id in tqdm.tqdm(range(max_iters)):
        env.reset()
        epsilon = max(epsilon * .999985, .02)

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            if np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])

                all_q_values = np.squeeze(neural_net.predict(all_q_inputs))
                chosen_action = available_actions[np.argmax(all_q_values)]
                chosen_action_q_value = np.max(all_q_values)
                # q_value = neural_net.predict(np.array([q_inputs]))[0][0]
                # if chosen_action is None or chosen_action_q_value < q_value:
                #     chosen_action = a
                #     chosen_action_q_value = q_value

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            if env.is_game_over():
                target = r
                q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
                neural_net.train_on_batch(np.array([q_inputs]), np.array([target]))
                break

            next_available_actions = env.available_actions_ids()

            if np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                for a in next_available_actions:
                    q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, max_actions_count)])
                    q_value = neural_net.predict(np.array([q_inputs]))[0][0]
                    if next_chosen_action is None or next_chosen_action_q_value < q_value:
                        next_chosen_action = a
                        next_chosen_action_q_value = q_value

            next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, max_actions_count)])

            next_chosen_action_q_value = neural_net.predict(np.array([next_q_inputs]))[0][0]

            target = r + gamma * next_chosen_action_q_value

            q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
            neural_net.train_on_batch(np.array([q_inputs]), np.array([target]))

        average = (average * iters + env.score()) / (iters + 1)
        iters += 1

        if episode_id % scale == 0:
            scores.append(average)
            average = 0.0
            iters = 0
            drl_sample_project_python.main.plot_scores(model_name, scores, scale)
            drl_sample_project_python.main.save_neural_net(neural_net, model_name)

    drl_sample_project_python.main.plot_scores(model_name, scores, scale)

    return neural_net