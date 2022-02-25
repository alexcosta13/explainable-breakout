import gym

from agent import Agent


def main():
    env = gym.make('Breakout-v4')
    env.reset()
    agent = Agent(env.action_space.n)
    for _ in range(100):
        new_state, reward, is_done, info = env.step(agent.next_action())
        print(info)


if __name__ == "__main__":
    main()
