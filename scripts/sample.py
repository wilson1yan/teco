import argparse
import numpy as np
import os.path as osp
import os
import jax

from teco.train_utils import seed_all
from teco.models import load_ckpt, sample
from teco.data import Data


def save(i, s, r, folder):
    s = s.reshape(-1, *s.shape[2:])
    s = s * 0.5 + 0.5
    s = (s * 255).astype(np.uint8)
    r = r.reshape(-1, *r.shape[2:])
    r = r * 0.5 + 0.5
    r = (r * 255).astype(np.uint8)

    s[:, :args.open_loop_ctx] = r[:, :args.open_loop_ctx]

    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(osp.join(folder, f'data_{i}.npz'), real=r, fake=s)


MAX_BATCH = 64
def main(args):
    global MAX_BATCH
    seed_all(args.seed)

    kwargs = dict()
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.open_loop_ctx is not None:
        kwargs['open_loop_ctx'] = args.open_loop_ctx
    
    model, state, config = load_ckpt(args.ckpt, return_config=True, 
                                     **kwargs, data_path=args.data_path,
                                     vqvae_ckpt=args.vqvae_ckpt)

    if args.include_actions:
        assert config.use_actions

    if config.use_actions and not args.include_actions:
        assert config.dropout_actions

    folder = osp.join(args.ckpt, 'samples')
    if args.include_actions:
        folder += '_action'
    folder += f'_{args.open_loop_ctx}'

    data = Data(config)
    loader = data.create_iterator(train=False, prefetch=False, repeat=False)
    batch = next(loader)

    MAX_BATCH = min(MAX_BATCH, args.batch_size)
    B = MAX_BATCH // jax.local_device_count()
    idx = 0
    for i in range(0, args.batch_size // jax.local_device_count(), B):
        v_in = batch['video'][:, i:i+B]
        act_in = batch['actions'][:, i:i+B]

        if config.use_actions and not args.include_actions:
            act_in = np.full_like(act_in, -1)
        s,r  = sample(model, state, v_in, act_in, seed=args.seed)
        save(idx, s, r, folder)
        idx += 1

    print('Saved to', folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('-v', '--vqvae_ckpt', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-n', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--seq_len', type=int, default=None)
    parser.add_argument('-o', '--open_loop_ctx', type=int, default=None)
    parser.add_argument('-a', '--include_actions', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
