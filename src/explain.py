import pickle
import yaml

import numpy as np

from lime_implementation import lime_explain
from shap_implementation import shap_explain
from gradcam import gradcam_implementation
from video import make_movie
from utils import load_agent


def run_episode(args):
    from gamewrapper import GameWrapper

    game_wrapper = GameWrapper(args["ENV_NAME"], args["MAX_NOOP_STEPS"])
    agent = load_agent(args)

    history = {"state": [], "raw_state": [], "action": []}

    terminal = True
    eval_rewards = []
    evaluate_frame_number = 0

    for frame in range(args["EVAL_LENGTH"]):
        if terminal:
            game_wrapper.reset(evaluation=True)
            life_lost = True
            episode_reward_sum = 0

        action = (
            1 if life_lost else agent.get_action(0, game_wrapper.state, evaluation=True)
        )

        new_frame, state, reward, terminal, life_lost = game_wrapper.step(
            action, render_mode="explain"
        )

        history["state"].append(game_wrapper.state)
        history["raw_state"].append(new_frame)
        history["action"].append(action)
        evaluate_frame_number += 1
        episode_reward_sum += reward

        if terminal:
            eval_rewards.append(episode_reward_sum)

            if args["WRITE_TERMINAL"]:
                print(
                    f'Game over, reward: {episode_reward_sum}, frame: {frame}/{args["EVAL_LENGTH"]}'
                )

    print(
        "Average reward:",
        np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum,
    )

    history["raw_state"] = np.stack(history["raw_state"], axis=0)[-50:]
    history["state"] = np.stack(history["state"], axis=0)[-50:]
    history["action"] = np.array(history["action"])[-50:]

    return history


if __name__ == "__main__":
    with open("explain.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["LOAD_HISTORY_FROM"] is None:
        h = run_episode(config)
        if config["SAVE_HISTORY_TO"] is not None:
            with open(config["SAVE_HISTORY_TO"], "wb") as f:
                pickle.dump(h, f)

    else:
        with open(config["LOAD_HISTORY_FROM"], "rb") as f:
            h = pickle.load(f)

    h = {k: v[: config["VIDEO_LENGTH_FRAMES"]] for k, v in h.items()}

    if config["EXPLAINABILITY_METHOD"] == "":
        make_movie(
            h["raw_state"],
            config["VIDEO_FPS"],
            config["MOVIE_SAVE_DIR"],
            config["MOVIE_TITLE"],
        )
    elif config["EXPLAINABILITY_METHOD"] == "shap":
        shap_explain(config, h)
    elif config["EXPLAINABILITY_METHOD"] == "lime":
        lime_explain(config, h)
    elif config["EXPLAINABILITY_METHOD"] == "gradcam":
        gradcam_implementation(config, h)
    else:
        raise NotImplementedError("EXPLAINABILITY_METHOD not supported")
    # NOOP, FIRE, RIGHT, LEFT
