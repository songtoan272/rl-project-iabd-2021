import tqdm

import drl_sample_project_python.main
from drl_sample_project_python.do_not_touch.contracts import DeepSingleAgentEnv
import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')


def get_reinforce_with_baseline(env: DeepSingleAgentEnv, gamma: float, max_iters: int, neural_net, v_neural_net, model_name: str):

    scores = []
    average = 0.0
    iters = 0
    scale = 20

    kills = 0

    for episode_id in tqdm.tqdm(range(max_iters)):
        env.reset_random()
        transitions = []

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            all_q_values = np.squeeze(neural_net.predict(s.reshape(-1, len(s))))
            masked_q_value = []
            for a in available_actions:
                masked_q_value.append(all_q_values[a])
            masked_q_value /= sum(masked_q_value)
            chosen_action = np.random.choice(available_actions, 1, False, p=masked_q_value)[0]
            chosen_q_value = all_q_values[list(available_actions).index(chosen_action)]

            v_value = np.squeeze(v_neural_net.predict(s.reshape(-1, len(s))))

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            transitions.append([s, chosen_action, r, s_p, chosen_q_value, v_value])

        calculated_rewards = []
        for t in range(len(transitions)):
            g = 0
            for k, r in enumerate(transitions[t:]):
                g += gamma**(k-t-1) * r[2]
            calculated_rewards.append(g)

        input_batch = []
        rewards = []
        v_values = []
        for i, t in enumerate(transitions):
            input_batch.append(t[0])
            v_values.append(calculated_rewards[i] - t[5])
            reward = np.zeros(env.max_actions_count())
            reward[t[1]] = gamma ** i * (calculated_rewards[i] - t[5])
            rewards.append(reward)

        neural_net.train_on_batch(np.array(input_batch), np.array(rewards))
        v_neural_net.train_on_batch(np.array(input_batch), np.array(v_values))

        average = (average * iters + env.score()) / (iters + 1)
        iters += 1

        if env.score() == -1:
            kills += 1

        if episode_id % scale == 0 and episode_id != 0:
            scores.append(average)
            #print(env.get_pacgum_count())
            print(f'kills : {kills}')
            kills = 0
            save = False
            if average == max(scores):
                save = True
                drl_sample_project_python.main.save_neural_net(neural_net, model_name)
                print('model saved !')
            drl_sample_project_python.main.plot_scores(model_name, scores, scale, save)
            average = 0.0
            iters = 0

    return neural_net
