from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import lpips_jax


lpips_eval = None


def compute_metric(prediction, ground_truth, metric_fn, replicate=True, average_dim=1): 
    # BTHWC in [0, 1]
    assert prediction.shape == ground_truth.shape
    B, T = prediction.shape[0], prediction.shape[1]
    prediction = prediction.reshape(-1, *prediction.shape[2:])
    ground_truth = ground_truth.reshape(-1, *ground_truth.shape[2:])

    if replicate:
        prediction = np.reshape(prediction, (jax.local_device_count(), -1, *prediction.shape[-3:]))
        ground_truth = np.reshape(ground_truth, (jax.local_device_count(), -1, *ground_truth.shape[-3:]))

    metrics = metric_fn(prediction, ground_truth)
    metrics = np.reshape(metrics, (B, T))

    metrics = metrics.mean(axis=average_dim) # B or T depending on dim

    return metrics


# all methods below take as input pairs of images
# of shape BCHW. They DO NOT reduce batch dimension
# NOTE: Assumes that images are in [0, 1]

def get_ssim(replicate=True, average_dim=1):
    def fn(imgs1, imgs2):
        ssim_fn = jax.pmap(ssim) if replicate else ssim
        ssim_val = ssim_fn(imgs1, imgs2)
        return jax.device_get(ssim_val)
    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn, replicate=replicate, average_dim=average_dim)

def get_psnr(replicate=True, average_dim=1):
    def fn(imgs1, imgs2):
        psnr_fn = jax.pmap(psnr) if replicate else psnr
        psnr_val = psnr_fn(imgs1, imgs2)
        return jax.device_get(psnr_val)
    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn, replicate=replicate, average_dim=average_dim)


def psnr(a, b, max_val=1.0):
    mse = jnp.mean((a - b) ** 2, axis=[-3, -2, -1])
    val = 20 * jnp.log(max_val) / jnp.log(10.0) - np.float32(10 / np.log(10)) * jnp.log(mse)
    return val


def get_lpips(replicate=True, average_dim=1):
    global lpips_eval
    if lpips_eval is None:
        lpips_eval = lpips_jax.LPIPSEvaluator(net='alexnet', replicate=replicate)
    def fn(imgs1, imgs2):
        imgs1 = 2 * imgs1 - 1
        imgs2 = 2 * imgs2 - 1

        lpips = lpips_eval(imgs1, imgs2)
        lpips = np.reshape(lpips, (-1,))
        return jax.device_get(lpips)
    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn, replicate=replicate, average_dim=average_dim)


def ssim(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val, filter_size, filter_sigma, k1, k2)
    return jnp.mean(ssim_per_channel, axis=-1)
         

def _ssim_per_channel(img1, img2, max_val, filter_size, filter_sigma, k1, k2):
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = jnp.tile(kernel, [1, 1, img1.shape[-1], 1])
    kernel = jnp.transpose(kernel, [2, 3, 0, 1])
    
    compensation = 1.0
    
    def reducer(x):
        x_shape = x.shape
        x = jnp.reshape(x, (-1, *x.shape[-3:]))
        x = jnp.transpose(x, [0, 3, 1, 2])
        y = jax.lax.conv_general_dilated(x, kernel, [1, 1], 
                                         'VALID', feature_group_count=x.shape[1])

        y = jnp.reshape(y, [*x_shape[:-3], *y.shape[1:]])
        return y

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2)
    ssim_val = jnp.mean(luminance * cs, axis=[-3, -2])
    cs = jnp.mean(cs, axis=[-3, -2])
    return ssim_val, cs

        
def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    
    mean0 = reducer(x)
    mean1 = reducer(y)

    num0 = mean0 * mean1 * 2.0
    den0 = jnp.square(mean0) + jnp.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(x * y) * 2.0
    den1 = reducer(jnp.square(x) + jnp.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    return luminance, cs
    
    
def _fspecial_gauss(size, sigma):
    coords = jnp.arange(size, dtype=jnp.float32)
    coords -= (size - 1.0) / 2.0

    g = jnp.square(coords)
    g *= -0.5 / jnp.square(sigma)
    
    g = jnp.reshape(g, [1, -1]) + jnp.reshape(g, [-1, 1])
    g = jnp.reshape(g, [1, -1])
    g = jax.nn.softmax(g, axis=-1)
    return jnp.reshape(g, [size, size, 1, 1])
 

import tensorflow.compat.v2 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

i3d_model = None


# FVD
def fvd_preprocess(videos, target_resolution):
    # videos: BTHWC in [0, 1]
    videos = tf.convert_to_tensor(videos * 255., dtype=tf.float32)
    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    resized_videos = tf.image.resize(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
    output_videos = tf.reshape(resized_videos, target_shape)
    scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
    return scaled_videos

    
def create_id3_embedding(videos):
    global i3d_model
    module_spec = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'

    if not i3d_model:
        base_model = hub.load(module_spec)
        input_tensor = base_model.graph.get_tensor_by_name('input_frames:0')
        i3d_model = base_model.prune(input_tensor, 'RGB/inception_i3d/Mean:0')
    
    output = i3d_model(videos)
    return output


def calculate_fd(real_activations, generated_activations):
    return tfgan.eval.frechet_classifier_distance_from_activations(
        real_activations, generated_activations
    ).numpy()


def fvd(video_1, video_2):
    video_1 = fvd_preprocess(video_1, (224, 224))
    video_2 = fvd_preprocess(video_2, (224, 224))
    x = create_id3_embedding(video_1)
    y = create_id3_embedding(video_2)
    result = calculate_fd(x, y)
    return result


video_model, video_state = None, None

def compute_feats(state, videos, rng):
    rng, new_rng = jax.random.split(rng)
    variables = {'params': state.params, **state.model_state}
    feats = video_model.apply(variables, videos, return_features=True, rngs={'rng': rng})
    return feats, new_rng

    
def create_video_embedding(videos):
    BATCH_SIZE = 32
    global video_model, video_state
    rngs = jax.random.PRNGKey(0)
    rngs = jax.random.split(rngs, jax.local_device_count())

    if video_model is None:
        from .models import load_ckpt
        path = '/home/TODO/logs/hier_video/dl_maze_video_contr_1657861689.9321504'
        video_model, video_state = load_ckpt(path, data_path='dummy')
    
    pbar = tqdm(total=videos.shape[0] // BATCH_SIZE)
    feats = []
    for i in range(0, videos.shape[0], BATCH_SIZE):
        inp = videos[i:i + BATCH_SIZE]
        inp = np.reshape(inp, (jax.local_device_count(), -1, *inp.shape[1:]))
        f, rngs = jax.pmap(compute_feats)(video_state, inp, rngs)
        f = jax.device_get(f)
        f = np.reshape(f, (-1, *f.shape[2:]))
        feats.append(f)
        pbar.update(1)
    feats = np.concatenate(feats)
    return feats
