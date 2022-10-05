import math
from moviepy.editor import ImageSequenceClip
import numpy as np

import jax
import jax.numpy as jnp


def topk_sample(rng, logits, top_k=None):
    if top_k is not None:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < jax.lax.top_k(logits, top_k)[0][..., -1, None]
        logits = jnp.where(indices_to_remove, jnp.finfo(logits.dtype).min, logits)
 
    samples = jax.random.categorical(rng, logits, axis=-1)
    return samples 


def add_border(video, color, width=0.025):
    # video: BTHWC in [0, 1]
    S = math.ceil(int(video.shape[3] * width))

    # top
    video[:, :, :S, :, 0] = color[0]
    video[:, :, :S, :, 1] = color[1]
    video[:, :, :S, :, 2] = color[2]

    # bottom
    video[:, :, -S:, :, 0] = color[0]
    video[:, :, -S:, :, 1] = color[1]
    video[:, :, -S:, :, 2] = color[2]

    # left
    video[:, :, :, :S, 0] = color[0]
    video[:, :, :, :S, 1] = color[1]
    video[:, :, :, :S, 2] = color[2]

    # right
    video[:, :, :, -S:, 0] = color[0]
    video[:, :, :, -S:, 1] = color[1]
    video[:, :, :, -S:, 2] = color[2]


def flatten(x, start=0, end=None):
    i, j = start, end
    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    return reshape_range(x, i, j, (np.prod(x.shape[i:j]),))

    
def reshape_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i
    
    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j
    
    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return jnp.reshape(x, target_shape)


def save_video_grid(video, fname=None, nrow=None, fps=10):
    b, t, h, w, c = video.shape
    video = (video * 255).astype('uint8')

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * ncol + padding,
                          (padding + w) * nrow + padding, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    if fname is not None:
        clip = ImageSequenceClip(list(video_grid), fps=fps)
        clip.write_gif(fname, fps=fps)
        print('saved videos to', fname)
    
    return video_grid # THWC, uint8
