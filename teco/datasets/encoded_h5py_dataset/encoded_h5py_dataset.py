"""encoded_h5py_dataset dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import numpy as np

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
"""

class EncodedH5pyConfig(tfds.core.BuilderConfig):
    def __init__(self, *, path, min_length, size, **kwargs):
        super().__init__(version=tfds.core.Version('1.0.0'), **kwargs)
        self.path = path
        self.min_length = min_length
        self.size = size


class EncodedH5pyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for encoded_h5py_dataset dataset."""

  BUILDER_CONFIGS = [
    EncodedH5pyConfig(
      name='something-something',
      path='something-something.hdf5',
      min_length=16,
      size=16
    ),
    EncodedH5pyConfig(
      name='something-something_r128',
      path='something-something_r128.hdf5',
      min_length=16,
      size=16
    ),
    EncodedH5pyConfig(
      name='bdd100k',
      path='bdd100k.hdf5',
      min_length=300,
      size=16
    ),
    EncodedH5pyConfig(
      name='bair',
      path='encoded_bair.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='robonet',
      path='encoded_robonet.hdf5',
      min_length=16,
      size=16
    ),
    EncodedH5pyConfig(
      name='kinetics_200',
      path='encoded_kinetics.hdf5',
      min_length=200,
      size=16
    ),
    EncodedH5pyConfig(
      name='kinetics_100',
      path='encoded_kinetics.hdf5',
      min_length=100,
      size=16
    ),
    EncodedH5pyConfig(
      name='gqn_mazes',
      path='encoded_gqn_mazes.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='dl_maze',
      path='encoded_dl_maze.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_turn',
      path='encoded_minerl_turn.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_fwd0.5',
      path='encoded_minerl_fwd0.5.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_fwd0.9',
      path='encoded_minerl_fwd0.9.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_fwd0.9_maxfwd25',
      path='encoded_minerl_fwd0.9_maxfwd25.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_forest',
      path='encoded_minerl_forest.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_forest_easy',
      path='encoded_minerl_forest_easy.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_forest_med',
      path='encoded_minerl_forest_med.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_biomes',
      path='encoded_minerl_biomes.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_marsh',
      path='encoded_minerl_marsh.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_desert',
      path='encoded_minerl_desert.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_marsh_v2',
      path='encoded_minerl_marsh_v2.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='minerl_marsh_v2_l500',
      path='encoded_minerl_marsh_v2_l500.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='cater_8x8',
      path='encoded_cater_8x8.hdf5',
      min_length=None,
      size=8
    ),
    EncodedH5pyConfig(
      name='cater_16x16',
      path='encoded_cater_16x16.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='habitat_simple',
      path='encoded_habitat_simple.hdf5',
      min_length=None,
      size=16
    ),
    EncodedH5pyConfig(
      name='habitat_l300',
      path='encoded_habitat_l300.hdf5',
      min_length=None,
      size=16
    ),
  ]

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place the `*.hdf5` file in the `manual_dir/`.
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    size = self.builder_config.size
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'video': tfds.features.Tensor(shape=(None, size, size), dtype=tf.uint16),
            'actions': tfds.features.Tensor(shape=(None,), dtype=tf.int32),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.manual_dir / self.builder_config.path

    return {
        'train': self._generate_examples(path, 'train'),
        'test': self._generate_examples(path, 'test'),
    }

  def _generate_examples(self, path, split):
    """Yields examples."""
    data = h5py.File(path, 'r')
    images = data[f'{split}_data']
    if f'{split}_actions' in data and 'bair' not in self.builder_config.path:
        actions = data[f'{split}_actions'][:]
    else:
        print('Did not find actions... Generating dummy actions')
        actions = np.zeros((images.shape[0],), dtype=np.int32)
    idxs = data[f'{split}_idx'][:]

    for i in range(len(idxs)):
      start = idxs[i]
      end = idxs[i + 1] if i < len(idxs) - 1 else len(images)
      video = images[start:end]
      action = actions[start:end].astype(np.int32)
      if self.builder_config.min_length is not None and video.shape[0] < self.builder_config.min_length:
          continue

      yield i, {
        'video': video,
        'actions': action,
      }
