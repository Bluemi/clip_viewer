import matplotlib.pyplot as plt
import numpy as np
import torch

_imshow_initialized = False
pressed_key = None
plt.rcParams['keymap.save'].remove('s')


def _press(event):
    global pressed_key
    if event.name == 'button_press_event':
        if event.xdata is None or event.ydata is None:
            pressed_key = (f'mouse{event.button}', None, None)
        else:
            pressed_key = (f'mouse{event.button}', int(round(event.xdata)), int(round(event.ydata)))
    else:
        pressed_key = event.key


def imshow(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if isinstance(img, np.ndarray):
        if np.issubdtype(img.dtype, float):
            img = np.minimum(np.maximum(img, 0.0), 1.0)

    plt.ion()

    cmap = None
    if img.ndim == 2:
        cmap = 'gray'
    plt.clf()
    plt.imshow(img, cmap=cmap)
    plt.show()

    global _imshow_initialized
    if not _imshow_initialized:
        plt.gcf().canvas.mpl_connect('key_press_event', _press)
        plt.gcf().canvas.mpl_connect('button_press_event', _press)
        _imshow_initialized = True

    plt.waitforbuttonpress()

    return pressed_key
