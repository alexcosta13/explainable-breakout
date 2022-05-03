import matplotlib.animation as manimation
import matplotlib.pyplot as plt


def make_movie(fig_array, fps=8, save_dir="", movie_title="", resolution=75):
    ff_mpeg_writer = manimation.writers["ffmpeg"]
    metadata = dict(title="test", artist="alex", comment="atari-video")
    movie_title = "{}.mp4".format(movie_title)
    writer = ff_mpeg_writer(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
    num_frames = len(fig_array) if type(fig_array) == list else fig_array.shape[0]
    with writer.saving(fig, save_dir + movie_title, resolution):
        for i in range(num_frames):
            img = fig_array[i]
            plt.imshow(img)
            writer.grab_frame()
            fig.clear()
