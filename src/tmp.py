import wandb
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import argparse
import numpy as np
from collections import deque
import random

from gamewrapper import GameWrapper

tf.keras.backend.set_floatx('float64')
wandb.init(project="explainable-breakout")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        # states = np.array(states).reshape(args.batch_size, -1)
        # next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)

    def save(self, path):
        pass


class ActionStateModel:
    def __init__(self, state_dim, action_dim, history_length = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps
        self.history_length = history_length

        self.model = self.create_model()

    def create_model(self):

        model_input = Input(shape=(self.state_dim[0], self.state_dim[1], self.history_length))
        x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255

        x = Conv2D(
            32,
            (8, 8),
            strides=4,
            kernel_initializer=VarianceScaling(scale=2.0),
            activation="relu",
            use_bias=False,
        )(x)
        x = Conv2D(
            64,
            (4, 4),
            strides=2,
            kernel_initializer=VarianceScaling(scale=2.0),
            activation="relu",
            use_bias=False,
        )(x)
        x = Conv2D(
            64,
            (3, 3),
            strides=1,
            kernel_initializer=VarianceScaling(scale=2.0),
            activation="relu",
            use_bias=False,
        )(x)
        x = Conv2D(
            1024,
            (7, 7),
            strides=1,
            kernel_initializer=VarianceScaling(scale=2.0),
            activation="relu",
            use_bias=False,
        )(x)

        # Split into value and advantage streams
        val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(
            x
        )  # custom splitting layer

        val_stream = Flatten()(val_stream)
        val = Dense(1, kernel_initializer=VarianceScaling(scale=2.0))(val_stream)

        adv_stream = Flatten()(adv_stream)
        adv = Dense(self.action_dim, kernel_initializer=VarianceScaling(scale=2.0))(adv_stream)

        # Combine streams into Q-Values
        reduce_mean = Lambda(
            lambda w: tf.reduce_mean(w, axis=1, keepdims=True)
        )  # custom layer for reduce mean

        q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

        # Build model
        model = Model(model_input, q_vals)
        model.compile(Adam(args.lr), loss=tf.keras.losses.Huber())

        return model

    def predict(self, state):
        print(state.shape)
        return self.model.predict(state)

    def get_action(self, state):
        q_value = self.action_probabilities(state)
        print(np.argmax(q_value))
        return np.argmax(q_value)

    def action_probabilities(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)
        print(q_value)
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return q_value

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, path):
        self.model.save_weights(path)


class Agent:
    def __init__(self, env, state_dim, action_dim):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states)[
                range(args.batch_size), np.argmax(self.model.predict(next_states), axis=1)]
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state

            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            wandb.log({'Reward': total_reward})

    def save(self, path, save_replay_buffer=False):
        self.model.save(path)
        if save_replay_buffer:
            self.buffer.save(path)


def main():
    env = GameWrapper('BreakoutNoFrameskip-v4', 20)
    agent = Agent(env, (84, 84), 4)
    agent.train(max_episodes=10)
    agent.save(".")


if __name__ == "__main__":
    main()
