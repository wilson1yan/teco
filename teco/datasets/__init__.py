from flax import jax_utils
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from . import encoded_h5py_dataset


GCS_PATH = 'gs://wilson_smae/datasets'


class Data:
    def __init__(self, config, xmap=False):
        self.config = config
        self.xmap = xmap

        dataset_builder = tfds.builder(self.config.data_path,
                data_dir=GCS_PATH if config.download else None)
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

        seq_len = self.config.seq_len
        def process(features):
            video = tf.cast(features['video'], tf.int32)
            T = tf.shape(video)[0]
            start_idx = tf.random.uniform((), 0, T - seq_len + 1, dtype=tf.int32)
            video = tf.identity(video[start_idx:start_idx + seq_len])
            actions = tf.cast(features['actions'], tf.int32)
            actions = tf.identity(actions[start_idx:start_idx + seq_len])
            return dict(video=video, actions=actions)

        dataset_builder = tfds.builder(self.config.data_path,
                data_dir=GCS_PATH if self.config.download else None)
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
            if 'actions' not in xs:
                xs['actions'] = None
            return xs

        iterator = map(prepare_tf_data, dataset)

        if prefetch:
            iterator = jax_utils.prefetch_to_device(iterator, 2)

        return iterator
