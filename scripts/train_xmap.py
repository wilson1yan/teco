import os
import os.path as osp
import numpy as np
import time
import argparse
import yaml
import pickle
import wandb
import gcsfs
import glob

import jax
from jax import random
import jax.numpy as jnp
from jax.experimental.maps import xmap
from flax.training import checkpoints

from teco.datasets import Data
from teco.models.xmap.train_utils import create_xmap_train_state_spec, \
    shard_train_state, unshard_train_state
from teco.train_utils import init_model_state, \
        get_first_device, ProgressMeter, seed_all
from teco.utils import flatten, add_border, save_video_grid
from teco.models.xmap import get_model, get_sample
from teco.models.xmap.mesh import default_mesh


GCS_LOG_DIR = 'TODO'


def main():
    global model
    global model_shard
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    seed_all(config.seed)

    fs = gcsfs.GCSFileSystem(project='rll-tpus')
    # Download VQ-GAN or AE if needed
    if hasattr(config, 'vqvae_ckpt'):
        gcs_path = osp.join(GCS_LOG_DIR, osp.basename(config.vqvae_ckpt))
        fs.get(gcs_path, config.vqvae_ckpt, recursive=True)
    if hasattr(config, 'ae_ckpt'):
        gcs_path = osp.join(GCS_LOG_DIR, osp.basename(config.ae_ckpt))
        fs.get(gcs_path, config.ae_ckpt, recursive=True)

    # Check if need to load previous checkpoint
    gcs_path = osp.join(GCS_LOG_DIR, config.run_id)

    files = glob.glob(osp.join(config.output_dir, 'checkpoints', '*'))
    if len(files) > 0:
        print('Found previous checkpoints', files)
        config.ckpt = config.output_dir
    elif fs.exists(gcs_path):
        print('Found previous checkpoint at', gcs_path)
        fs.get(gcs_path, config.output_dir, recursive=True)
        config.ckpt = config.output_dir
    else:
        config.ckpt = None

    if is_master_process:
        root_dir = os.environ['DATA_DIR']
        os.makedirs(osp.join(root_dir, 'wandb'), exist_ok=True)

        wandb.init(project='teco', config=config,
                   dir=root_dir, id=config.run_id, resume='allow')
        wandb.run.name = config.run_id
        wandb.run.save()

    data = Data(config, xmap=True)
    train_loader = data.create_iterator(train=True, prefetch=False)
    test_loader = data.create_iterator(train=False, prefetch=False)

    batch = next(train_loader)
    batch = get_first_device(batch)
    model, model_shard = get_model(config)
    state, schedule_fn = init_model_state(init_rng, model, batch, config)
    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print('Restored from checkpoint')

    mesh = default_mesh(config.num_shards)
    jax.experimental.maps.thread_resources.env = (
        jax.experimental.maps.ResourceEnv(physical_mesh=mesh, loops=())
    )
    state = jax.device_get(state)
    state_spec = create_xmap_train_state_spec(model_shard, state)

    print('Moving params to devices')
    move_params = xmap(
        lambda state: state, in_axes=(state_spec,), out_axes=state_spec,
        axis_resources={'data': 'dp', 'model': 'mp'}
    )
    state = move_params(shard_train_state(model_shard, state))
    iteration = int(state.step)

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    server = jax.profiler.start_server(9999)

    print('Starting training...')
    rngs = random.split(rng, max(1, jax.local_device_count() // config.num_shards))
    while iteration <= config.total_steps:
        iteration, state, rngs = train(iteration, state_spec, state, train_loader,
                                       schedule_fn, rngs)
        if iteration % config.save_interval == 0:
            if is_master_process:
                state_ = unshard_train_state(model_shard, jax.device_get(state))
                save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1)

                if args.sync_freq and os.environ.get('DEBUG') != '1':
                    gcs_path = osp.join(GCS_LOG_DIR, osp.basename(config.output_dir))
                    if fs.exists(gcs_path):
                        fs.rm(gcs_path, recursive=True)
                    fs.put(config.output_dir, GCS_LOG_DIR, recursive=True)
                print('Saved checkpoint to', save_path)
                del state_ # Needed to prevent a memory leak bug
        if iteration % config.viz_interval == 0:
            visualize(model_shard, iteration, state_spec, state, test_loader)
        iteration += 1

    # Final sync with GCS
    gcs_path = osp.join(GCS_LOG_DIR, osp.basename(config.output_dir))
    if fs.exists(gcs_path):
        fs.rm(gcs_path, recursive=True)
    fs.put(config.output_dir, GCS_LOG_DIR, recursive=True)


def train_step(batch, state, rng):
    new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    rngs = {k: r for k, r in zip(config.rng_keys, rngs)}
    def loss_fn(params):
        variables = {'params': params, **state.model_state}
        out = state.apply_fn(
            variables,
            video=batch['video'],
            actions=batch['actions'],
            deterministic=False,
            rngs=rngs
        )
        loss = out['loss']
        return loss, out

    aux, grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)
    out = model_shard.aggregate_metrics(aux[1])

    grads = model_shard.model_spec.reduce_grad(grads)
    grads = jax.lax.pmean(grads, axis_name='data')

    new_state = state.apply_gradients(
        grads=grads,
    )

    grad_norm = jnp.linalg.norm(
        jax.tree_leaves(jax.tree_map(jnp.linalg.norm, grads))
    )
    grad_norm = jax.lax.pmean(grad_norm, 'model')
    out['grad_norm'] = grad_norm
    return new_state, out, new_rng



def train(iteration, state_spec, state, train_loader, schedule_fn, rngs):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model_shard.metrics + ['grad_norm']
    )

    p_train_step = xmap(
        train_step, in_axes=[('data', ...), state_spec, ('data', ...)],
        out_axes=(state_spec, (...,), ('data', ...)),
        axis_resources={'data': 'dp', 'model': 'mp'}
    )

    end = time.time()
    while True:
        batch = next(train_loader)
        batch_size = batch['video'].shape[1]
        progress.update(data=time.time() - end)

        state, return_dict, rngs = p_train_step(batch, state, rngs)

        metrics = {k: return_dict[k] for k in model_shard.metrics}
        metrics['grad_norm'] = return_dict['grad_norm']
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process and iteration % config.log_interval == 0:
            wandb.log({'train/lr': schedule_fn(iteration)}, step=iteration)
            wandb.log({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.save_interval == 0 or \
        iteration % config.viz_interval == 0 or \
        iteration >= config.total_steps:
            return iteration, state, rngs

        iteration += 1


def visualize(model, iteration, state_spec, state, test_loader):
    batch = next(test_loader)

    sample = get_sample(config)
    predictions, real = sample(model, state, batch['video'], batch['actions'],
                               log_output=True, state_spec=state_spec)
    predictions, real = jax.device_get(predictions), jax.device_get(real)
    predictions, real = predictions * 0.5 + 0.5, real * 0.5 + 0.5
    predictions = flatten(predictions, 0, 2)
    add_border(predictions[:, :config.open_loop_ctx], (0., 1., 0.))
    add_border(predictions[:, config.open_loop_ctx:], (1., 0., 0.))

    original = flatten(real, 0, 2)
    video = np.stack((predictions, original), axis=1) # (NB)2THWC
    video = flatten(video, 0, 2) # (NB2)THWC
    video = save_video_grid(video)
    video = np.transpose(video, (0, 3, 1, 2))
    if is_master_process:
        wandb.log({'viz/sample': wandb.Video(video, fps=20, format='gif')}, step=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-s', '--sync_freq', action='store_true')
    args = parser.parse_args()

    args.run_id = args.output_dir

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')
    print(f'JAX local devices: {jax.local_device_count()}')

    if not osp.isabs(args.output_dir):
        if 'DATA_DIR' not in os.environ:
            raise Exception('DATA_DIR environment variable not set')
        root_folder = os.environ['DATA_DIR']
        args.output_dir = osp.join(root_folder, args.output_dir)

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['viz_interval'] = 10
        config['save_interval'] = 10
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    is_master_process = jax.process_index() == 0

    main()
