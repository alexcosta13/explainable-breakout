import pickle

import numpy as np
import cv2
import yaml
from PIL import Image
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.python.keras.backend as K

from dqn import build_dqn
from video import make_movie
from utils import load_agent


# def build_guided_model(action_space_n):
#     if "GuidedBackProp" not in ops._gradient_registry._registry:
#
#         @ops.RegisterGradient("GuidedBackProp")
#         def _GuidedBackProp(op, grad):
#             dtype = op.inputs[0].dtype
#             return (
#                 grad * tf.cast(grad > 0.0, dtype) * tf.cast(op.inputs[0] > 0.0, dtype)
#             )
#
#     g = K.get_session().graph
#     with g.gradient_override_map({"Relu": "GuidedBackProp"}):
#         return build_dqn(action_space_n)


# def guided_backpropagation(model, frame):
#     grad_model = tf.keras.models.Model(inputs=[model.input], outputs=[model.output])
#     input_ = tf.convert_to_tensor(frame, dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(input_)
#         layer_output = grad_model(input_)
#     grads_val = tape.gradient(layer_output, input_)
#     return grads_val[0]


def grad_cam(model, layer_number, frame):
    # CHANGED LINE grad_model = tf.keras.models.Model(inputs=[model.input], outputs=[model.output, model.get_layer(
    # layer_name).output])
    grad_model = tf.keras.models.Model(
        inputs=[model.input], outputs=[model.output, model.layers[layer_number].output]
    )
    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(frame)
        loss = predictions[:, 1]
    output = conv_outputs
    grads_val = tape.gradient(loss, conv_outputs)[0]
    # weights = np.mean(grads_val, axis=(2, 3))
    weights = tf.reduce_mean(grads_val, axis=(0, 1))

    # weights = weights[0, :]
    output = output[0, :, :, :]

    # weights = np.expand_dims(weights, axis=0)
    # cam = np.dot(output, weights)
    a = model.layers[layer_number].output.shape
    cam = tf.zeros(model.layers[layer_number].output.shape[1:-1])
    for i in range(weights.shape[0]):
        cam += weights[i] * output[:, :, i]
    cam = cam.numpy()
    cam = cv2.resize(cam, (84, 84), cv2.INTER_LINEAR)

    # cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max
    cam[cam < 0.0] = 0
    return cam


# def guided_gradcam(gradcam, guided_prop):
#     guided_cam = np.dot(gradcam, guided_prop)
#     return guided_cam


def gradcam_implementation(args, history, resolution=75):
    agent = load_agent(args)
    model = agent.DQN
    # g_model = build_guided_model(4)
    total_frames = len(history["state"])
    # history["gradcam"], history["gbp"], history["ggc"] = [], [], []
    history["gradcam"] = []

    for i in range(total_frames):
        if i < total_frames:
            frame = history["state"][i].copy()
            frame = np.expand_dims(frame, axis=0)
            if i % 10 == 0:
                print(i)

            cam_heatmap = grad_cam(model, 4, frame)
            # gbp_heatmap = guided_backpropagation(g_model, frame)
            # ggc_heatmap = guided_gradcam(cam_heatmap, gbp_heatmap)

            # cam_heatmap = np.asarray(cam_heatmap)
            history["gradcam"].append(cam_heatmap)
            # history["gbp"].append(gbp_heatmap)
            # history["ggc"].append(ggc_heatmap)

    history_gradcam = history["gradcam"].copy()
    # history_gbp = history["gbp"].copy()
    # history_ggc = history["ggc"].copy()

    fig_array = normalization(history_gradcam, history, visu="cam")
    # fig_array2 = normalization(history_gbp, history, visu="gbp")
    # fig_array3 = normalization(history_ggc, history, visu="gbp")
    make_movie(
        fig_array,
        fps=args["VIDEO_FPS"],
        save_dir=args["MOVIE_SAVE_DIR"],
        movie_title="gradcam" + args["MOVIE_TITLE"],
        resolution=resolution,
    )
    # make_movie(
    #     fig_array2,
    #     fps=args["VIDEO_FPS"],
    #     save_dir=args["MOVIE_SAVE_DIR"],
    #     movie_title="gbp" + args["MOVIE_TITLE"],
    #     resolution=resolution
    # )
    # make_movie(
    #     fig_array3,
    #     fps=args["VIDEO_FPS"],
    #     save_dir=args["MOVIE_SAVE_DIR"],
    #     movie_title="gcc" + args["MOVIE_TITLE"],
    #     resolution=resolution
    # )


def normalization(heatmap, history, visu):
    heatmap = np.asarray(heatmap)
    print("heat map shape", heatmap.shape)
    # if visu == "gbp":
    #     heatmap = heatmap[:, :, :]
    #     heatmap -= heatmap.mean()
    #     heatmap /= heatmap.std() + 1e-5
    #
    #     # heatmap *= 50
    #     heatmap *= 0.1
    #
    #     # clip to [0, 1]
    #     # gbp_heatmap += 0.5
    #     heatmap = np.clip(heatmap, -1, 1)
    #     # TODO fix later
    #     heatmap_pic1 = heatmap[:, :, :, 0]
    if visu == "cam":
        heatmap *= 1
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_pic = heatmap[:, :, :]

    all_frames = history["raw_state"].copy()
    frame = np.zeros((len(all_frames), 84, 84, 3))
    for i in range(len(all_frames)):
        frame[i, :, :, :] = (
            np.asarray(Image.fromarray(all_frames[i]).resize((84, 84), Image.BILINEAR))
            / 255
        )
    proc_frame = overlap(frame, heatmap_pic)
    return proc_frame


def overlap(frame, gbp_heatmap):
    color_neg = [1.0, 0.0, 0.0]
    color_pos = [0.0, 1.0, 0.0]
    # color_chan = np.ones((FRAMES, 84, 84, 2), dtype=gbp_heatmap.dtype)
    alpha = 0.2
    # beta = 0.25
    # gbp_heatmap = np.expand_dims(gbp_heatmap, axis=4)
    _gbp_heatmap = [gbp_heatmap for _ in range(3)]
    _gbp_heatmap = np.stack(_gbp_heatmap, axis=3)
    gbp_heatmap = _gbp_heatmap
    # gbp_heatmap = np.concatenate((gbp_heatmap,color_chan),axis=3)
    gbp_heatmap_pos = np.asarray(gbp_heatmap.copy())
    gbp_heatmap_neg = np.asarray(gbp_heatmap.copy())
    gbp_heatmap_pos[gbp_heatmap_pos < 0.0] = 0
    gbp_heatmap_neg[gbp_heatmap_neg >= 0.0] = 0
    gbp_heatmap_neg = -gbp_heatmap_neg
    gbp_heatmap = (
        color_pos * gbp_heatmap_pos[:, :, :, :]
        + color_neg * gbp_heatmap_neg[:, :, :, :]
    )
    # gbp_heatmap = color_pos * gbp_heatmap_pos[:,:,:,:] + color_neg * gbp_heatmap_neg[:,:,:,:]
    mixed = alpha * gbp_heatmap + (1.0 - alpha) * frame
    mixed = np.clip(mixed, 0, 1)

    return mixed
