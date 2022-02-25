import random


class Agent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def next_action(self):
        return random.randint(0, self.n_actions - 1)

    def __str__(self):
        pass
