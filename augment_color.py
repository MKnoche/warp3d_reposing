import cv2
import numpy as np


def augment_brightness(im, in_colorspace, rng):
    if in_colorspace != 'rgb':
        cv2.cvtColor(im, cv2.COLOR_HSV2RGB, dst=im)

    im += rng.uniform(-0.125, 0.125)
    return 'rgb'


def augment_contrast(im, in_colorspace, rng):
    if in_colorspace != 'rgb':
        cv2.cvtColor(im, cv2.COLOR_HSV2RGB, dst=im)
    im -= 0.5
    im *= 1 + rng.uniform(-0.5, 0.5)
    im += 0.5
    return 'rgb'


def augment_hue(im, in_colorspace, rng):
    if in_colorspace != 'hsv':
        np.clip(im, 0, 1, out=im)
        cv2.cvtColor(im, cv2.COLOR_RGB2HSV, dst=im)
    hue = im[:, :, 0]
    hue += rng.uniform(-72, 72)
    hue[hue < 0] += 360
    hue[hue > 360] -= 360
    return 'hsv'


def augment_saturation(im, in_colorspace, rng):
    if in_colorspace != 'hsv':
        np.clip(im, 0, 1, out=im)
        cv2.cvtColor(im, cv2.COLOR_RGB2HSV, dst=im)

    saturation = im[:, :, 1]
    saturation *= 1 + rng.uniform(-0.5, 0.5)
    saturation[saturation > 1] = 1
    return 'hsv'


def augment_color(im, rng):
    im += 1
    im /= 2
    result = np.empty_like(im, dtype=np.float32)
    cv2.divide(im, (1, 1, 1, 1), dst=result, dtype=cv2.CV_32F)

    augmentation_functions = [augment_brightness, augment_contrast, augment_hue, augment_saturation]
    rng.shuffle(augmentation_functions)

    colorspace = 'rgb'
    for fn in augmentation_functions:
        colorspace = fn(result, colorspace, rng)

    if colorspace != 'rgb':
        cv2.cvtColor(result, cv2.COLOR_HSV2RGB, dst=result)

    np.clip(result, 0, 1, out=result)

    result = result.astype(np.float32)

    return result * 2 - 1
