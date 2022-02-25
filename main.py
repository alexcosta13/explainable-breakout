import gym


def main():
    env = gym.make('Breakout-v4')
    env.reset()
    for _ in range(1000):
        a = env.step(env.action_space.sample())
        print(a[-1])


if __name__ == "__main__":
    main()
