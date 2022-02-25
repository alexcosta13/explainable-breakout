import time

import gym

from agent import Agent

RENDER = True


def main():
    env = gym.make('Breakout-v4')
    env.reset()
    agent = Agent(env.action_space.n)
    is_done = False
    while not is_done:
        action = agent.next_action()
        new_state, reward, is_done, info = env.step(action)
        print(info)
        if RENDER:
            env.render()
            time.sleep(0.1)


if __name__ == "__main__":
    main()
