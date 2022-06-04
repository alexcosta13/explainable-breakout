import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

from own_utils import load_agent


def gradcam_calculate(model, layer_number, frame):
    gradient_model = tf.keras.models.Model(
        inputs=[model.input], outputs=[model.output, model.layers[layer_number].output]
    )
    with tf.GradientTape() as tape:
        predictions, conv_outputs = gradient_model(frame)
        loss = predictions[:, 1]
    output = conv_outputs
    grads_val = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads_val, axis=(0, 1))
    output = output[0, :, :, :]
    cam = tf.zeros(model.layers[layer_number].output.shape[1:-1])
    for i in range(weights.shape[0]):
        cam += weights[i] * output[:, :, i]
    cam = cam.numpy()
    cam = cv2.resize(cam, (84, 84), cv2.INTER_LINEAR)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max
    cam[cam < 0.0] = 0
    return cam


def postprocess(heatmap, history):
    heatmap = np.asarray(heatmap)
    heatmap *= 1
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_pic = heatmap[:, :, :]
    all_frames = history["raw_state"].copy() / 255
    final_heatmap = np.zeros(all_frames.shape[:-1])
    for i in range(len(all_frames)):
        final_heatmap[i, ...] = np.asarray(
            Image.fromarray(heatmap_pic[i]).resize(
                (all_frames.shape[1:-1][::-1]), Image.BILINEAR
            )
        )
    proc_frame = overlap(all_frames, final_heatmap)
    return proc_frame


def overlap(frame, heatmap, alpha=0.4):
    color_neg = [1.0, 0.0, 0.0]
    color_pos = [0.0, 1.0, 0.0]
    _heatmap = [heatmap for _ in range(3)]
    _heatmap = np.stack(_heatmap, axis=3)
    heatmap = _heatmap / _heatmap.max()
    heatmap_pos = np.asarray(heatmap.copy())
    heatmap_neg = np.asarray(heatmap.copy())
    heatmap_pos[heatmap_pos < 0.0] = 0
    heatmap_neg[heatmap_neg >= 0.0] = 0
    heatmap_neg = -heatmap_neg
    heatmap = color_pos * heatmap_pos[:, :, :, :] + color_neg * heatmap_neg[:, :, :, :]
    mixed = (1.0 - alpha) * frame + alpha * heatmap
    mixed = np.clip(mixed, 0, 1)

    return mixed


def gradcam_explain(args, history):
    agent = load_agent(args)
    model = agent.DQN
    total_frames = len(history["state"])
    history["gradcam"] = []

    for i in range(total_frames):
        if i < total_frames:
            frame = history["state"][i].copy()
            frame = np.expand_dims(frame, axis=0)
            cam_heatmap = gradcam_calculate(model, args["GRADCAM_LAYER"], frame)
            history["gradcam"].append(cam_heatmap)

    fig_array = postprocess(history["gradcam"], history)
    return fig_array
