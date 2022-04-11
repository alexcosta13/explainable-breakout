import numpy as np

from matplotlib.colors import LinearSegmentedColormap

from agent import Agent
from replaybuffer import ReplayBuffer


def load_agent(args):
    replay_buffer = ReplayBuffer(
        size=args["REPLAY_BUFFER_SIZE"],
        input_shape=args["INPUT_SHAPE"],
        use_per=args["USE_PER"],
    )
    agent = Agent(
        None,
        None,
        replay_buffer,
        4,
        input_shape=args["INPUT_SHAPE"],
        batch_size=args["AGENT_BATCH_SIZE"],
        use_per=args["USE_PER"],
    )

    # Training and evaluation
    if args["LOAD_AGENT_FROM"] is None:
        raise ValueError("LOAD_AGENT_FROM is null, you need to train the agent first.")

    print("Loading from", args["LOAD_AGENT_FROM"])
    agent.load(args["LOAD_AGENT_FROM"], args["LOAD_REPLAY_BUFFER"])

    return agent


def get_red_transparent_blue():
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, l))
    for l in np.linspace(0, 1, 100):
        colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, l))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colors)
