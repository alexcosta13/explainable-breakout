import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np


def make_movie(fig_array, num_frames, resolution, prefix, env_name, save_dir, explanation=None):
    movie_title = "{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    ff_mpeg_writer = manimation.writers['ffmpeg']
    metadata = dict(title='test', artist='mateus', comment='atari-video')
    writer = ff_mpeg_writer(fps=8, metadata=metadata)
    fig = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
    print("fig_array.shape: ", fig_array.shape)
    with writer.saving(fig, save_dir + movie_title, resolution):
        for i in range(num_frames):
            img = fig_array[i, ...]
            plt.imshow(img)
            if explanation:
                all_blue = np.array([[255 for _ in range(84)] for _ in range(84)])
                plt.imshow(all_blue, alpha=explanation)
            writer.grab_frame()
            fig.clear()
            if i % 100 == 0:
                print(i)
