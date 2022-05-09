import cv2


import numpy as np


# This function can resize to any shape, but was built to resize to 84x84
def process_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34 : 34 + 160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame


def process_multiple_frames(data, shape=(84, 84)):
    output = []
    for frame in data:
        processed = [
            process_frame(frame[:, :160, :], shape),
            process_frame(frame[:, 160 : 160 * 2, :], shape),
            process_frame(frame[:, 160 * 2 : 160 * 3, :], shape),
            process_frame(frame[:, 160 * 3 :, :], shape),
        ]
        processed = np.concatenate(processed, axis=2)
        output.append(processed[np.newaxis, ...])
    return np.concatenate(output, axis=0)
