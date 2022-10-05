import glob
import os.path as osp
import numpy as np
from flax import jax_utils
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio
from tensorflow.python.lib.io import file_io
import io


def get_size(config, train):
    split = 'train' if train else 'test'
    folder = osp.join(config.data_path, split, '*', '*.mp4')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    return len(fns)


def load_video(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*', '*.mp4')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(num_ds_shards, ds_shard_id).tolist()

    # TODO resizing video
    def read(path):
        path = path.decode('utf-8')

        video = tfio.experimental.ffmpeg.decode_video(tf.io.read_file(path)).numpy()
        start_idx = np.random.randint(0, video.shape[0] - config.seq_len + 1)
        video = video[start_idx:start_idx + config.seq_len]
        video = 2 * (video / np.array(255., dtype=np.float32)) - 1
        
        np_path = path[:-3] + 'npz'
        if tf.io.gfile.exists(np_path):
            if path.startswith('gs://'):
                np_path = io.BytesIO(file_io.FileIO(np_path, 'rb').read())
            np_data = np.load(np_path)
            actions = np_data['actions'].astype(np.int32)
            actions = actions[start_idx:start_idx + config.seq_len]
        else:
            actions = np.zeros((video.shape[0],), dtype=np.int32)
        
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset
                

class Data:
    def __init__(self, config, xmap=False):
        self.config = config
        self.xmap = xmap

        if osp.exists(self.config.data_path) or self.config.data_path.startswith('gs://'):
            self.train_size = get_size(config, train=True)
            self.test_size = get_size(config, train=False)
        else:
            dataset_builder = tfds.builder(
                osp.basename(self.config.data_path),
                data_dir=osp.dirname(config.data_path))
            dataset_builder.download_and_prepare()

            self.train_size = dataset_builder.info.splits['train'].num_examples
            self.test_size = dataset_builder.info.splits['test'].num_examples
        print(f'Dataset {config.data_path} of size {self.train_size} / {self.test_size}')

    @property
    def train_itr_per_epoch(self):
        return self.train_size // self.config.batch_size

    @property
    def test_itr_per_epoch(self):
        return self.test_size // self.config.batch_size

    def create_iterator(self, train, repeat=True, prefetch=True):
        if self.xmap:
            num_data = jax.device_count() // self.config.num_shards
            num_data_local = max(1, jax.local_device_count() // self.config.num_shards)
            if num_data >= jax.process_count():
                num_ds_shards = jax.process_count()
                ds_shard_id = jax.process_index()
            else:
                num_ds_shards = num_data
                n_proc_per_shard = jax.process_count() // num_data
                ds_shard_id = jax.process_index() // n_proc_per_shard
        else:
            num_data_local = jax.local_device_count()
            num_ds_shards = jax.process_count()
            ds_shard_id = jax.process_index()

        batch_size = self.config.batch_size // num_ds_shards
        split_name = 'train' if train else 'test'

        if osp.exists(self.config.data_path) or self.config.data_path.startswith('gs://'):
            dataset = load_video(self.config, split_name, num_ds_shards, ds_shard_id)
        else:
            seq_len = self.config.seq_len
            def process(features):
                video = tf.cast(features['video'], tf.int32)
                T = tf.shape(video)[0]
                start_idx = tf.random.uniform((), 0, T - seq_len + 1, dtype=tf.int32)
                video = tf.identity(video[start_idx:start_idx + seq_len])
                actions = tf.cast(features['actions'], tf.int32)
                actions = tf.identity(actions[start_idx:start_idx + seq_len])
                return dict(video=video, actions=actions)

            dataset_builder = tfds.builder(
                osp.basename(self.config.data_path),
                data_dir=osp.dirname(self.config.data_path))
            dataset_builder.download_and_prepare()
            num_examples = dataset_builder.info.splits[split_name].num_examples
            split_size = num_examples // num_ds_shards
            start = ds_shard_id * split_size
            split = '{}[{}:{}]'.format(split_name, start, start + split_size)
            dataset = dataset_builder.as_dataset(split=split)

        if self.config.cache:
            dataset = dataset.cache()

        options = tf.data.Options()
        options.threading.private_threadpool_size = 48
        options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        dataset = dataset.map(process)

        if repeat:
            dataset = dataset.repeat()
        if train:
            dataset = dataset.shuffle(batch_size * 32, seed=self.config.seed)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)

        def prepare_tf_data(xs):
            def _prepare(x):
                x = x._numpy()
                x = x.reshape((num_data_local, -1) + x.shape[1:])
                return x
            xs = jax.tree_map(_prepare, xs)
            return xs

        iterator = map(prepare_tf_data, dataset)

        if prefetch:
            iterator = jax_utils.prefetch_to_device(iterator, 2)

        return iterator
