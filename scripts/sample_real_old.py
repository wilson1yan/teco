import argparse
import numpy as np
import os.path as osp
import os
import jax
import time
import multiprocessing as mp

from teco.train_utils import seed_all
from teco.utils import flatten
from teco.models import load_ckpt, sample
from teco.utils import flatten
from teco.data_pyt import Data


def worker(queue):
    while True:
        data = queue.get()
        if data is None:
            break
        i, s, r = data
        print(s.shape, r.shape)
        start = time.time()
        s = s.reshape(-1, *s.shape[2:])
        s = s * 0.5 + 0.5
        s = (s * 255).astype(np.uint8)
        r = r.reshape(-1, *r.shape[2:])
        r = r * 0.5 + 0.5
        r = (r * 255).astype(np.uint8)

        if args.no_context:
            s = s[: args.open_loop_ctx:]
            r = r[:, args.open_loop_ctx:]
        else:
            s[:, :args.open_loop_ctx] = r[:, :args.open_loop_ctx]

        folder = osp.join(args.ckpt, 'samples')
        if args.include_actions:
            folder += '_action'
        folder += f'_{args.open_loop_ctx}'
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
                                     **kwargs, data_path=args.data_path)

    print(config)

    if args.include_actions:
        assert config.use_actions

    if config.use_actions and not args.include_actions:
        assert config.dropout_actions

    old_seq_len = config.seq_len
    config.seq_len = args.seq_len
    config.eval_seq_len = args.seq_len
    data = Data(config)
    loader = data.test_dataloader()
    batch = next(iter(loader))
    config.seq_len = old_seq_len

    queue = mp.Queue()
    procs = [mp.Process(target=worker, args=(queue,)) for _ in range(1)]
    [p.start() for p in procs]
    
    start = time.time()
    batch = {k: np.reshape(v.numpy(), (jax.local_device_count(), -1, *v.shape[1:])) 
             for k, v in batch.items()}
    print(batch['video'].shape)
    if 'actions' not in batch:
        batch['actions'] = None
    MAX_BATCH = min(MAX_BATCH, args.batch_size)
    B = MAX_BATCH // jax.local_device_count()
    idx = 0
    for _ in range(args.n_repeat):
        for i in range(0, args.batch_size // jax.local_device_count(), B):
            v_in = batch['video'][:, i:i+B]
            act_in = batch['actions'][:, i:i+B] if batch['actions'] is not None else None

            if config.use_actions and not args.include_actions:
                act_in = np.full_like(act_in, -1)
            s,r  = sample(model, state, v_in, act_in, seed=args.seed)
            queue.put((idx, s, r))
            idx += 1
    [queue.put(None) for _ in range(4)]
    print('sampling', time.time() - start)

    [p.join() for p in procs]

    folder = osp.join(args.ckpt, 'samples')
    if args.include_actions:
        folder += '_action'
    folder += f'_{args.open_loop_ctx}'
    print('Saved to', folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-n', '--batch_size', type=int, default=32)
    parser.add_argument('-r', '--n_repeat', type=int, default=1)
    parser.add_argument('-l', '--seq_len', type=int, default=None)
    parser.add_argument('-o', '--open_loop_ctx', type=int, default=None)
    parser.add_argument('-a', '--include_actions', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--no_context', action='store_true')
    args = parser.parse_args()

    main(args)
