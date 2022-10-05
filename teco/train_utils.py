from typing import Any
from collections import OrderedDict
import random
import numpy as np
import jax
from flax.training import train_state
from flax.core.frozen_dict import freeze
import optax


class TrainState(train_state.TrainState):
    model_state: Any


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_first_device(x):
    x = jax.tree_util.tree_map(lambda a: a[0], x)
    return jax.device_get(x)


def print_model_size(params, name=''):
    model_params_size = jax.tree_util.tree_map(lambda x: x.size, params)
    total_params_size = sum(jax.tree_util.tree_flatten(model_params_size)[0])
    print('model parameter count:', total_params_size)


def get_learning_rate_fn(config):
    if config.lr_schedule == 'cosine':
        learning_rate_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.,
            peak_value=config.lr,
            warmup_steps=config.warmup_steps,
            decay_steps=config.total_steps - config.warmup_steps
        )
    elif config.lr_schedule == 'constant':
        learning_rate_fn = optax.join_schedules([
            optax.linear_schedule(
                init_value=0.,
                end_value=config.lr,
                transition_steps=config.warmup_steps
            ),
            optax.constant_schedule(config.lr)
        ], [config.warmup_steps])
    else:
        raise ValueError(f'Unknown schedule: {config.lr_schedule}')
    
    return learning_rate_fn

    
def get_optimizer(config):
    learning_rate_fn = get_learning_rate_fn(config)
    tx = optax.adamw(learning_rate=learning_rate_fn, b1=0.9, b2=0.95, 
                     weight_decay=config.weight_decay)
    return tx, learning_rate_fn


def init_model_state(rng_key, model, sample, config):
    variables = model.init(
        rngs={k: rng_key for k in ['params', *config.rng_keys]},
        **{k: sample[k] for k in config.batch_keys}
    ).unfreeze()
    params = freeze(variables.pop('params'))
    model_state = variables
    print_model_size(params)

    tx, learning_rate_fn = get_optimizer(config)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        model_state=model_state
    ), learning_rate_fn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total_iters, meter_names, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = OrderedDict({mn: AverageMeter(mn, ':6.3f') 
                                   for mn in meter_names})
        self.prefix = prefix
    
    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, n=n)

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']' 
