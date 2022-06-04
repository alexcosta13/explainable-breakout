import os
import sys

import numpy as np
from stable_baselines3.common.utils import set_random_seed

from utils.utils import ALGOS, create_test_env, get_model_path, get_saved_hyperparams


def load_agent():
    env_id = "BreakoutNoFrameskip-v4"
    folder = "rl-trained-agents"
    algo = "dqn"
    seed = 0

    name_prefix, model_path, log_path = get_model_path(
        0, folder, algo, env_id, False, 0, False
    )

    set_random_seed(seed)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path)

    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs={},
    )

    kwargs = dict(seed=0)
    # Dummy buffer size as we don't need memory to enjoy the trained agent
    kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(
        model_path, env=env, custom_objects=custom_objects, **kwargs
    )

    return model


def get_mask():
    mask = np.zeros((210, 160, 3))
    mask[17:196, :8, :] = [142, 142, 142]
    mask[17:195, 152:, :] = [142, 142, 142]
    mask[17:32, :, :] = [142, 142, 142]
    return np.concatenate((mask, mask, mask, mask), axis=1)
