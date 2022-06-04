import argparse
import pickle
import yaml

import numpy as np

from lime_implementation import lime_explain
from shap_implementation import shap_explain
from gradcam import gradcam_explain
from utils import create_test_env, get_saved_hyperparams
from video import make_movie, save_frames
from own_utils import load_agent


def run_episode(args):
    model = load_agent(args)

    history = {"state": [], "raw_state": [], "action": []}

    env_id = "BreakoutNoFrameskip-v4"
    algo = "dqn"
    seed = 0
    no_render = True
    hyperparams, stats_path = get_saved_hyperparams(
        "/content/explainable-breakout/src/rl-trained-agents/dqn/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4"
    )

    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=seed,
        log_dir=None,
        should_render=not no_render,
        hyperparams=hyperparams,
        # env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=seed)
    kwargs.update(dict(buffer_size=1))

    deterministic = False
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    for _ in range(args["EVAL_LENGTH"] + 1):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs, _, dones, _ = env.step(action)
        episode_starts = dones
        raw_state = env.render("rgb_array")
        history["state"].append(obs.squeeze())
        history["raw_state"].append(raw_state.squeeze())
        history["action"].append(action.squeeze())

    env.close()

    history["raw_state"] = np.stack(history["raw_state"], axis=0)
    history["state"] = np.stack(history["state"], axis=0)
    history["action"] = np.array(history["action"])
    print(
        "SHAPES",
        history["raw_state"].shape,
        history["state"].shape,
        history["action"].shape,
    )

    return history


def parse_cli_arguments(args):
    parser = argparse.ArgumentParser(description="Process some integers.")
    for key, value in args.items():
        parser.add_argument("--" + key, type=type(value), default=value)
    # parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction)

    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    with open("explain.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = parse_cli_arguments(config)

    if config["LOAD_HISTORY_FROM"] is None:
        h = run_episode(config)
        if config["SAVE_HISTORY_TO"] is not None:
            with open(config["SAVE_HISTORY_TO"], "wb") as f:
                pickle.dump(h, f)

    else:
        with open(config["LOAD_HISTORY_FROM"], "rb") as f:
            h = pickle.load(f)

    h = {
        k: v[
            config["VIDEO_FIRST_FRAME"] : config["VIDEO_FIRST_FRAME"]
            + config["VIDEO_LENGTH_FRAMES"]
        ]
        for k, v in h.items()
    }

    if config["EXPLAINABILITY_METHOD"] == "":
        frames = h["raw_state"]
    elif config["EXPLAINABILITY_METHOD"] == "shap":
        frames = shap_explain(config, h)
    elif config["EXPLAINABILITY_METHOD"] == "lime":
        frames = lime_explain(config, h)
    elif config["EXPLAINABILITY_METHOD"] == "gradcam":
        frames = gradcam_explain(config, h)
    else:
        raise NotImplementedError(
            "EXPLAINABILITY_METHOD not supported, enter shap, lime or gradcam"
        )

    if config["save_frames"]:
        save_frames(
            frames,
            config["MOVIE_SAVE_DIR"] + config["EXPLAINABILITY_METHOD"],
        )
    else:
        make_movie(
            frames,
            config["VIDEO_FPS"],
            config["MOVIE_SAVE_DIR"],
            config["MOVIE_TITLE"],
        )
