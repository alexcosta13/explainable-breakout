import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from utils import get_red_transparent_blue


def make_movie(fig_array, num_frames, resolution, prefix, env_name, save_dir):
    movie_title = "{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    ff_mpeg_writer = manimation.writers["ffmpeg"]
    metadata = dict(title="test", artist="mateus", comment="atari-video")
    writer = ff_mpeg_writer(fps=8, metadata=metadata)
    fig = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
    print("fig_array.shape: ", fig_array.shape)
    with writer.saving(fig, save_dir + movie_title, resolution):
        for i in range(num_frames):
            img = fig_array[i, ...]
            plt.imshow(img)
            writer.grab_frame()
            fig.clear()
            if i % 100 == 0:
                print(i)


def make_movie_explanation(
    fig_array, explanation_heatmap, resolution, movie_title, save_dir, transparency=0.15
):
    ff_mpeg_writer = manimation.writers["ffmpeg"]
    metadata = dict(title=movie_title, artist="mateus", comment="atari-video")
    movie_title = "{}.mp4".format(movie_title)
    writer = ff_mpeg_writer(fps=8, metadata=metadata)
    fig = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
    print("fig_array.shape: ", fig_array.shape)

    with writer.saving(fig, save_dir + movie_title, resolution):
        for i, (image, show) in enumerate(zip(fig_array, explanation_heatmap)):
            plt.imshow(
                show, cmap=get_red_transparent_blue(), vmin=-show.max(), vmax=show.max()
            )
            plt.imshow(image, alpha=transparency)
            writer.grab_frame()
            fig.clear()
            if i % 100 == 0:
                print(i)
