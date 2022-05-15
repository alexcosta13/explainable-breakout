import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_movie(fig_array, fps=8, save_dir="", movie_title="", resolution=75):
    ff_mpeg_writer = manimation.writers["ffmpeg"]
    metadata = dict(title="test", artist="alex", comment="atari-video")
    movie_title = f"{movie_title}.mp4"
    writer = ff_mpeg_writer(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
    num_frames = len(fig_array) if type(fig_array) == list else fig_array.shape[0]
    with writer.saving(fig, save_dir + movie_title, resolution):
        for i in tqdm(range(num_frames)):
            img = fig_array[i]
            plt.imshow(img)
            writer.grab_frame()
            fig.clear()


def save_frames(fig_array, save_dir=""):
    metadata = dict(title="test", artist="alex", comment="atari-video")
    num_frames = len(fig_array) if type(fig_array) == list else fig_array.shape[0]
    for i in tqdm(range(num_frames)):
        img = fig_array[i]
        plt.imshow(img)
        plt.savefig(f"{save_dir}/frame{i:02d}.png", metadata=metadata)
