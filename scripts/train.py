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
from flax.training import checkpoints
from flax import jax_utils

from teco.datasets import Data
from teco.train_utils import init_model_state, \
        get_first_device, ProgressMeter, seed_all
from teco.utils import flatten, add_border, save_video_grid
from teco.models import get_model, get_sample


GCS_LOG_DIR = 'TODO'


def main():
    global model
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    seed_all(config.seed)

    fs = gcsfs.GCSFileSystem(project='rll-tpus')
    # Download VQ-GAN if needed
    if hasattr(config, 'vqvae_ckpt'):
        gcs_path = osp.join(GCS_LOG_DIR, osp.basename(config.vqvae_ckpt))
        fs.get(gcs_path, config.vqvae_ckpt, recursive=True)
    
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

    data = Data(config)
    train_loader = data.create_iterator(train=True)
    test_loader = data.create_iterator(train=False)

    batch = next(train_loader)
    batch = get_first_device(batch)
    model = get_model(config)
    state, schedule_fn = init_model_state(init_rng, model, batch, config)
    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print('Restored from checkpoint')

    iteration = int(state.step)
    state = jax_utils.replicate(state)

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    rngs = random.split(rng, jax.local_device_count())
    while iteration <= config.total_steps:
        iteration, state, rngs = train(iteration, model, state, train_loader,
                                       schedule_fn, rngs)
        if iteration % config.save_interval == 0:
            if is_master_process:
                state_ = jax_utils.unreplicate(state)
                save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1)
        
                if args.sync_freq and os.environ.get('DEBUG') != '1':
                    gcs_path = osp.join(GCS_LOG_DIR, osp.basename(config.output_dir))
                    if fs.exists(gcs_path):
                        fs.rm(gcs_path, recursive=True)
                    fs.put(config.output_dir, GCS_LOG_DIR, recursive=True)
                print('Saved checkpoint to', save_path)
                del state_ # Needed to prevent a memory leak bug
        if iteration % config.viz_interval == 0:
            visualize(model, iteration, state, test_loader)
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
    out = aux[1]
    grads = jax.lax.pmean(grads, axis_name='batch')
    grad_norm = jnp.linalg.norm(
        jax.tree_leaves(jax.tree_map(jnp.linalg.norm, grads))
    )
    new_state = state.apply_gradients(
        grads=grads,
    )

    out['grad_norm'] = grad_norm
    return new_state, out, new_rng



def train(iteration, model, state, train_loader, schedule_fn, rngs):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics + ['grad_norm']
    )

    p_train_step = jax.pmap(train_step, axis_name='batch')

    end = time.time()
    while True:
        batch = next(train_loader)
        batch_size = batch['video'].shape[1]
        progress.update(data=time.time() - end)

        state, return_dict, rngs = p_train_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics['grad_norm'] = return_dict['grad_norm'].mean()
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


def visualize(model, iteration, state, test_loader):
    batch = next(test_loader)

    sample = get_sample(config)
    predictions, real = sample(model, state, batch['video'], batch['actions'], log_output=True)
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
