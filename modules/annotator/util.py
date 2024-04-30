import numpy as np
import cv2
import os


annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)

    # resolution = 512
    # for (768w, 512h)
    # min is 512
    # so k = 512/512 = 1
    # so if you specify the minimum edge for resolution, k is always 1.
    k = float(resolution) / min(H, W)  # e.g. 512 / smaller edge len
    H *= k
    W *= k

    # Resize the image closer to the multiples of 64
    # If the width and height of the image is already the multiples of 64,
    # then there will be no changes.
    # ir(x) => int(round(x))
    # 1 -> 1 / 64 = 0, 0 * 64 = 0
    # 2 -> 2 / 64 = 0, 0 * 64 = 0
    # 64 -> 64 / 64 = 1, 1 * 64 = 64
    # 65 -> ir(65 / 64) = 1, 1 * 64 = 64
    # 95 -> ir(95 / 64) * 64 = 64
    # 96 -> ir(96 / 64) * 64 = 128
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img
