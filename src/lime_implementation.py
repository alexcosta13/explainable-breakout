import numpy as np
from lime import lime_image

from preprocessing import process_multiple_frames
from src.video import make_movie_explanation
from utils import load_agent


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

    for image in data:
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image,
            func,
            top_labels=4,
            hide_color=0,
            num_samples=args["LIME_NUM_SAMPLES"],
        )

        # temp, mask = explanation.get_image_and_mask(
        #     2, positive_only=True, num_features=10, hide_rest=False, min_weight=0.2
        # )
        #
        # temp, mask = temp[:, :160, :], mask[:, :160]
        # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        # plt.show()

        # Map each explanation weight to the corresponding superpixel
        dict_heatmap_right = dict(explanation.local_exp[2])
        heatmap_right = np.vectorize(dict_heatmap_right.get)(explanation.segments)[
            :, :160
        ]

        dict_heatmap_left = dict(explanation.local_exp[3])
        heatmap_left = np.vectorize(dict_heatmap_left.get)(explanation.segments)[
            :, :160
        ]

        history["explanation"].append(heatmap_right - heatmap_left)

    # Plot. The visualization makes more sense if a symmetrical colorbar is used.
    # plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
    # plt.colorbar()
    # plt.show()

    make_movie_explanation(
        history["raw_state"],
        history["explanation"],
        resolution=args["VIDEO_RESOLUTION"],
        movie_title=args["MOVIE_TITLE"],
        save_dir=args["MOVIE_SAVE_DIR"],
        transparency=args["TRANSPARENCY"],
    )
