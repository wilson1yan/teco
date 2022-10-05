from tqdm import tqdm
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

i3d_model = None

def fvd_preprocess(videos, target_resolution):
  videos = tf.convert_to_tensor(videos * 255.0, dtype=tf.float32)
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def create_id3_embedding(videos):
  """Get id3 embeddings."""
  global i3d_model
  module_spec = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'

  if not i3d_model:
    base_model = hub.load(module_spec)
    input_tensor = base_model.graph.get_tensor_by_name('input_frames:0')
    i3d_model = base_model.prune(input_tensor, 'RGB/inception_i3d/Mean:0')

  output = i3d_model(videos)
  return output

def calculate_fvd(real_activations, generated_activations):
  return tfgan.eval.frechet_classifier_distance_from_activations(
      real_activations, generated_activations)

def embed(videos):
  pbar = tqdm(total=sum([v.shape[0] for v in videos]))
  embs = []
  for video in videos:
    for v in video:
      v = v[None]
      v = fvd_preprocess(v, (224, 224)).numpy()
      emb = create_id3_embedding(tf.convert_to_tensor(v, dtype=tf.float32))
      embs.append(emb)
      pbar.update(1)
  embs = np.concatenate(embs)
  return embs

def fvd(video_1, video_2):
  if not isinstance(video_1, (tuple, list)):
      video_1, video_2 = [video_1], [video_2]

  embed_1 = embed(video_1)
  embed_2 = embed(video_2)
  result = calculate_fvd(embed_1, embed_2)
  return result.numpy()
