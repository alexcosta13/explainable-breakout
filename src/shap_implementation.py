import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import shap
from preprocessing import process_multiple_frames
from video import make_movie_explanation
from utils import load_agent, get_red_transparent_blue


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

    masker = shap.maskers.Image("inpaint_telea", (210, 160 * 4, 3))
    explainer = shap.Explainer(func, masker)
    shap_values = explainer(
        data,
        max_evals=max_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:number_of_outputs],
    )

    return shap_values


def shap_postprocess(shap_values):
    shap_values = shap_values.values

    max_value = shap_values.max()
    min_value = shap_values.min()

    shap_values = (shap_values - min_value) / (max_value - min_value)

    right = shap_values[:, :, :160, 0, 2]
    left = shap_values[:, :, :160, 0, 3]

    return right - left


def shap_explain(args, history):
    agent = load_agent(args)

    shap_values = shap_calculate(
        agent,
        {k: v[30:40] for k, v in history.items()},
        args["SHAP_MAX_EVALS"],
        args["SHAP_BATCH_SIZE"],
    )
    # shap_values = shap_calculate(agent, history, args["SHAP_MAX_EVALS"], args["SHAP_BATCH_SIZE"])

    history["explanation"] = shap_postprocess(shap_values)

    image = history["raw_state"][0]
    show = history["explanation"][0]
    plt.imshow(show, cmap=get_red_transparent_blue(), vmin=-show.max(), vmax=show.max())
    plt.imshow(image, alpha=args["TRANSPARENCY"])
    plt.title(f"just custom {args['TRANSPARENCY']}")
    plt.show()

    make_movie_explanation(
        history["raw_state"],
        history["explanation"],
        resolution=args["VIDEO_RESOLUTION"],
        movie_title=args["MOVIE_TITLE"],
        save_dir=args["MOVIE_SAVE_DIR"],
        transparency=args["TRANSPARENCY"],
    )
