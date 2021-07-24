import collections

import tqdm

import drl_sample_project_python.main
from drl_sample_project_python.do_not_touch.contracts import DeepSingleAgentEnv
import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'new_state', 'terminal', 'q_value'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return True if len(self.buffer) >= self.capacity else False

    def place_available(self):
        return self.capacity - len(self.buffer)

    def capacity(self):
        return self.capacity()

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size, replace=False)
        states, actions, rewards, next_states, terminale, q_value = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(terminale), np.array(q_value)


def get_deep_q_learning(env: DeepSingleAgentEnv, gamma: float, epsilon: float, max_iters: int, neural_net, model_name: str):
    scores = []
    average = 0.0
    iters = 0
    scale = 10

    batch_size = 32
    buffer = ExperienceReplay(50)

    for episode_id in tqdm.tqdm(range(max_iters)):
        epsilon = max(epsilon * .99, .1)
        env.reset()

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            if np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                all_q_inputs = np.zeros((len(available_actions), env.state_description_length() + env.max_actions_count()))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, env.max_actions_count())])

                all_q_values = np.squeeze(neural_net.predict(all_q_inputs))
                chosen_action = available_actions[np.argmax(all_q_values)]

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            go = env.is_game_over()

            if env.is_game_over():
                exp = Experience(s, chosen_action, r, s_p, go, -1)
                buffer.append(exp)
            else:
                next_available_actions = env.available_actions_ids()

                if np.random.uniform(0.0, 1.0) < epsilon:
                    next_chosen_action = np.random.choice(next_available_actions)
                else:
                    next_chosen_action = None
                    next_chosen_action_q_value = None
                    for a in next_available_actions:
                        q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, env.max_actions_count())])
                        q_value = neural_net.predict(np.array([q_inputs]))[0][0]
                        if next_chosen_action is None or next_chosen_action_q_value < q_value:
                            next_chosen_action = a
                            next_chosen_action_q_value = q_value

                next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, env.max_actions_count())])
                next_chosen_action_q_value = neural_net.predict(np.array([next_q_inputs]))[0][0]

                exp = Experience(s, chosen_action, r, s_p, go, next_chosen_action_q_value)
                buffer.append(exp)

            state_batch, action_batch, reward_batch, state_p_batch, terminale_batch, q_value_batch = buffer.sample(batch_size)

            inputs = []
            rewards = []
            for i, s in enumerate(state_batch):
                inputs.append(np.hstack([s, tf.keras.utils.to_categorical(action_batch[i], env.max_actions_count())]))
                rewards.append(reward_batch[i] if terminale_batch[i] else reward_batch[i] + gamma * q_value_batch[i])
            neural_net.train_on_batch(np.array(inputs), np.array(rewards))

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
