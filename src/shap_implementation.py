import numpy as np

import shap
from preprocessing import process_multiple_frames
from utils import load_agent, get_mask


def shap_calculate(agent, history, max_evals, batch_size, number_of_outputs=4):
    def func(x):
        tmp = x.copy()
        tmp = process_multiple_frames(tmp)
        return agent.DQN(tmp)

    i = 0
    data = []
    while i < len(history["raw_state"]) - 4:
        data.append(
            np.concatenate(
                (
                    history["raw_state"][i],
                    history["raw_state"][i + 1],
                    history["raw_state"][i + 2],
                    history["raw_state"][i + 3],
                ),
                axis=1,
            )
        )
        i += 1

    masker = shap.maskers.Image(get_mask())
    explainer = shap.Explainer(func, masker)
    shap_values = explainer(
        np.array(data),
        max_evals=max_evals,
        batch_size=batch_size,
        # outputs=shap.Explanation.argsort.flip[:number_of_outputs],
    )

    return shap_values


def shap_postprocess(history, shap_values, percentile=90, alpha=0.7):
    shap_values = shap_values.values

    max_value = shap_values.max()
    min_value = shap_values.min()

    shap_values = (shap_values - min_value) / (max_value - min_value)
    # we set to zero non-relevant parts of the image (below 90-percentile)
    shap_values = np.where(
        shap_values > np.percentile(shap_values, percentile), shap_values, 0
    )

    right = shap_values[:, :, :160, :, 2]
    left = shap_values[:, :, :160, :, 3]

    red = np.zeros(right[0].shape)
    red[:, :, 0] = 1

    blue = np.zeros(right[0].shape)
    blue[:, :, 2] = 1

    frames = []

    for i in range(shap_values.shape[0]):
        raw_state = history["raw_state"][i]
        if history["action"][i] == 2:
            plot = np.where(
                right[i] == 0,
                raw_state / 255,
                raw_state / 255 * (1 - alpha) + blue * alpha * right[i],
            )
        elif history["action"][i] == 3:
            plot = np.where(
                left[i] == 0,
                raw_state / 255,
                raw_state / 255 * (1 - alpha) + red * alpha * left[i],
            )
        else:
            plot = raw_state
        frames.append(plot)

    return frames


def shap_explain(args, history):
    agent = load_agent(args)

    shap_values = shap_calculate(
        agent, history, args["SHAP_MAX_EVALS"], args["SHAP_BATCH_SIZE"]
    )

    frames = shap_postprocess(
        history, shap_values, args["PERCENTILE"], args["TRANSPARENCY"]
    )

    return frames
