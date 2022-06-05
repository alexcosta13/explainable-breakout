import numpy as np
from utils.utils import ALGOS, create_test_env, get_saved_hyperparams


def load_agent(args):
    env_id = "BreakoutNoFrameskip-v4"
    folder = "rl-trained-agents"
    algo = "dqn"
    seed = 0
    no_render = False

    model_path = "rl-trained-agents/dqn/BreakoutNoFrameskip-v4.pkl"
    #stats_path = "rl-trained-agents/dqn/BreakoutNoFrameskip-v4.pkl"
    #hyperparams, stats_path = get_saved_hyperparams(stats_path)

    #env = create_test_env(env_id, n_envs=1, is_atari=True,
    #                      stats_path=stats_path, seed=seed, log_dir=None,
    #                      should_render=not no_render, hyperparams=hyperparams)
    model = ALGOS[algo].load(model_path)
    return model


def get_mask():
    mask = np.zeros((210, 160, 3))
    mask[17:196, :8, :] = [142, 142, 142]
    mask[17:195, 152:, :] = [142, 142, 142]
    mask[17:32, :, :] = [142, 142, 142]
    return np.concatenate((mask, mask, mask, mask), axis=1)
