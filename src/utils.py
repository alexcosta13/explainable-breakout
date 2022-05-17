import numpy as np

from agent import Agent
from replaybuffer import ReplayBuffer


def load_agent(args):
    replay_buffer = ReplayBuffer(
        size=args["REPLAY_BUFFER_SIZE"],
        input_shape=args["INPUT_SHAPE"],
    )
    agent = Agent(
        None,
        None,
        replay_buffer,
        4,
        input_shape=args["INPUT_SHAPE"],
        batch_size=args["AGENT_BATCH_SIZE"],
    )

    if args["LOAD_AGENT_FROM"] is None:
        raise ValueError("LOAD_AGENT_FROM is null, you need to train the agent first.")

    if args["WRITE_TERMINAL"]:
        print("Loading from", args["LOAD_AGENT_FROM"])
    agent.load(args["LOAD_AGENT_FROM"], args["LOAD_REPLAY_BUFFER"])

    return agent


def get_mask():
    mask = np.zeros((210, 160, 3))
    mask[17:196, :8, :] = [142, 142, 142]
    mask[17:195, 152:, :] = [142, 142, 142]
    mask[17:32, :, :] = [142, 142, 142]
    return np.concatenate((mask, mask, mask, mask), axis=1)
