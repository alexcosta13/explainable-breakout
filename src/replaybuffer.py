import os
import random

import numpy as np


class ReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""

    def __init__(self, size=1000000, input_shape=(84, 84), history_length=4):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty(
            (self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8
        )
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

    # Saves a transition to the replay buffer
    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        if frame.shape != self.input_shape:
            raise ValueError("Dimension of frame is wrong!")

        if clip_reward:
            reward = np.sign(reward)

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32):
        """Returns a minibatch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
        """

        if self.count < self.history_length:
            raise ValueError("Not enough memories to get a minibatch")

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements. If either are
                # True, the index is invalid.
                if index >= self.current >= index - self.history_length:
                    continue
                if self.terminal_flags[index - self.history_length : index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx - self.history_length : idx, ...])
            new_states.append(self.frames[idx - self.history_length + 1 : idx + 1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        return (
            states,
            self.actions[indices],
            self.rewards[indices],
            new_states,
            self.terminal_flags[indices],
        )

    def save(self, folder_name):
        """Save the replay buffer to a folder"""
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + "/actions.npy", self.actions)
        np.save(folder_name + "/frames.npy", self.frames)
        np.save(folder_name + "/rewards.npy", self.rewards)
        np.save(folder_name + "/terminal_flags.npy", self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + "/actions.npy")
        self.frames = np.load(folder_name + "/frames.npy")
        self.rewards = np.load(folder_name + "/rewards.npy")
        self.terminal_flags = np.load(folder_name + "/terminal_flags.npy")
