import pickle

import matplotlib.pyplot as plt

import numpy as np
from shap_implementation import shap_explain
import yaml

# from gamewrapper import GameWrapper
from src.lime_implementation import lime_explain
from utils import load_agent


def run_episode(args):
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
    # h = run_episode(config)

    # with open('history.pkl', 'wb') as f:
    #    pickle.dump(h, f)

    with open("history.pkl", "rb") as f:
        h = pickle.load(f)

    lime_explain(config, h)
