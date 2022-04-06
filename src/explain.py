import pickle

import cv2
import matplotlib.pyplot as plt

import numpy as np
import shap
import yaml
from lime import lime_image

from agent import Agent
from gamewrapper import GameWrapper
from replaybuffer import ReplayBuffer
from video import make_movie


def load_agent(args):
    replay_buffer = ReplayBuffer(
        size=args["MEM_SIZE"], input_shape=args["INPUT_SHAPE"], use_per=args["USE_PER"]
    )
    agent = Agent(
        None,
        None,
        replay_buffer,
        4,
        input_shape=args["INPUT_SHAPE"],
        batch_size=args["BATCH_SIZE"],
        use_per=args["USE_PER"],
    )

    # Training and evaluation
    if args["LOAD_FROM"] is None:
        raise ValueError("LOAD_FROM is null, you need to train the agent first.")

    print("Loading from", args["LOAD_FROM"])
    agent.load(args["LOAD_FROM"], args["LOAD_REPLAY_BUFFER"])

    return agent


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
            action, render_mode="shap"
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


def shap_calculate(agent, history):
    def func(x):
        tmp = x.copy()
        tmp = tmp[..., 0].reshape(-1, 84, 84, 4)
        return agent.DQN(tmp)

    def gray2rgb(gray):
        return np.tile(gray, 3)

    def convert_state(s):
        a = gray2rgb(s.reshape(84, 84 * 4, 1))
        return a

    masker = shap.maskers.Image("inpaint_telea", (84, 84 * 4, 3))
    explainer = shap.Explainer(func, masker)
    shap_values = explainer(
        list(map(convert_state, history["state"][30:32])),
        max_evals=500,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:4],
    )
    return shap_values


def shap_postprocess(shap_values, original_shape=(160, 210)):
    arr = shap_values[..., 2].values
    new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype("uint8")
    transparency = (
        new_arr[:, :, :84, :]
        + new_arr[:, :, 84 : 84 * 2, :]
        + new_arr[:, :, 84 * 2 : 84 * 3, :]
        + new_arr[:, :, 84 * 3 :, :]
    ) / 4

    # max_transparency = np.maximum(
    #     np.maximum(new_arr[:, :84, :], new_arr[:, 84 : 84 * 2, :]),
    #     np.maximum(new_arr[:, 84 * 2 : 84 * 3, :], new_arr[:, 84 * 3 :, :]),
    # )

    transparency = transparency[..., 0] / 255

    # TODO resize transparency
    transparency = cv2.resize(
        transparency, (len(shap_values), *original_shape), interpolation=cv2.INTER_NEAREST
    )

    return transparency


def shap_explain(args, history):
    plt.imshow(history["raw_state"][30])
    plt.show()
    plt.imshow(history["state"][30])
    plt.show()

    agent = load_agent(args)

    shap_values = shap_calculate(agent, history)

    history["shap_values"] = shap_postprocess(shap_values)

    transparency = history["shap_values"][..., 0] / 255
    # max_transparency = np.where(max_transparency < 0.9, 0, 0.4)
    plt.imshow(history["state"][30])
    all_blue = np.array([[255 for _ in range(84)] for _ in range(84)])
    plt.imshow(all_blue, alpha=transparency)
    plt.title("max")
    plt.show()

    # TODO: concatenate image and transparency

    # make_movie(np.array(history['raw_state']), args["EVAL_LENGTH"],75,"play","Breakout","")
    # make_movie(np.array(history['raw_state']), len(history["raw_state"]), 75, "play", "Breakout", "")

    # arr = shap_values[0, ..., 2].values
    # new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype("uint8")
    # plt.imshow(new_arr)
    # plt.show()
    # transparency = (
    #         new_arr[:, :84, :]
    #         + new_arr[:, 84: 84 * 2, :]
    #         + new_arr[:, 84 * 2: 84 * 3, :]
    #         + new_arr[:, 84 * 3:, :]
    # )
    # plt.show()
    #
    # max_transparency = np.maximum(np.maximum(
    #     new_arr[:, :84, :]
    #     , new_arr[:, 84: 84 * 2, :]
    #
    # ),
    #     np.maximum(
    #         new_arr[:, 84 * 2: 84 * 3, :]
    #         , new_arr[:, 84 * 3:, :]))
    #
    # arr = shap_values[0, ..., 2].values
    # new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype("uint8")
    # plt.imshow(new_arr)
    # plt.show()
    #
    # transparency = (
    #     new_arr[:, :84, :]
    #     + new_arr[:, 84 : 84 * 2, :]
    #     + new_arr[:, 84 * 2 : 84 * 3, :]
    #     + new_arr[:, 84 * 3 :, :]
    # )
    # plt.show()
    #
    # all_blue = np.array([[255 for _ in range(84)] for _ in range(84)])
    # all_red = np.array([[[255, 0, 0] for _ in range(84)] for _ in range(84)])
    #
    # transparency = transparency[...,0] / 255 / 4
    # # transparency = np.where(transparency < 0.9, 0, 0.4)
    # plt.imshow(history["state"][30])
    # plt.imshow(all_blue, alpha=transparency)
    # plt.title('trans')
    # plt.show()
    #
    # max_transparency = max_transparency[..., 0] / 255
    # # max_transparency = np.where(max_transparency < 0.9, 0, 0.4)
    # plt.imshow(history["state"][30])
    # plt.imshow(all_blue, alpha=max_transparency)
    # plt.title('max')
    # plt.show()


def lime_explain(args, history):
    agent = load_agent(args)
    image = history["state"][0]
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.astype("double"),
        agent.DQN.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
    )
    print(explanation)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # h = run_episode(config)

    # with open('history.pkl', 'wb') as f:
    #    pickle.dump(h, f)

    with open("history.pkl", "rb") as f:
        h = pickle.load(f)

    shap_explain(config, h)
