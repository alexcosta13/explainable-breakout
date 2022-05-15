import numpy as np
from lime import lime_image

from preprocessing import process_multiple_frames
from utils import load_agent, get_mask


def lime_explain(args, history):
    agent = load_agent(args)

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

    history["explanation"] = []

    red = np.zeros((210, 160, 3))
    red[:, :, 0] = 1

    blue = np.zeros((210, 160, 3))
    blue[:, :, 2] = 1

    mask = get_mask()

    for i, image in enumerate(data):
        if history["action"][i] in (2, 3):
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                image,
                func,
                top_labels=4,
                hide_color=mask,
                num_samples=args["LIME_NUM_SAMPLES"],
            )

        percentile = args["PERCENTILE"]
        alpha = args["TRANSPARENCY"]
        image = image[:, :160, :] / 255

        if history["action"][i] == 2:
            # Map each explanation weight to the corresponding superpixel
            dict_heatmap_right = dict(explanation.local_exp[2])
            right = np.vectorize(dict_heatmap_right.get)(explanation.segments)[:, :160]
            right /= right.max()
            right = np.where(right > np.percentile(right, percentile), right, 0)
            right = np.repeat(right[:, :, np.newaxis], 3, axis=2)
            plot = np.where(
                right == 0,
                image,
                image * (1 - alpha) + blue * alpha * right,
            )
        elif history["action"][i] == 3:
            # Map each explanation weight to the corresponding superpixel
            dict_heatmap_left = dict(explanation.local_exp[3])
            left = np.vectorize(dict_heatmap_left.get)(explanation.segments)[:, :160]
            left /= left.max()
            left = np.where(left > np.percentile(left, percentile), left, 0)
            left = np.repeat(left[:, :, np.newaxis], 3, axis=2)
            plot = np.where(
                left == 0,
                image,
                image * (1 - alpha) + red * alpha * left,
            )
        else:
            plot = image

        history["explanation"].append(plot)

    return history["explanation"]
