import numpy as np
import argparse
import jax
import jax.lax as lax
from jax.experimental.maps import Mesh, xmap
from flax import jax_utils

from teco.models import load_vqvae
from teco.models.xmap.teco import TECOShard
from teco.models.teco import TECO


np.random.seed(100)

num_shards = 4
num_data = jax.device_count() // num_shards
mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(jax.device_count() // num_shards, num_shards), ['dp', 'mp'])

config = {
    'num_shards': num_shards,
    'action_dim': 6,
    'action_embed_dim': 16,
    'n_cond': 1,
    'dropout_actions': False,
    'drop_loss_rate': 1.0,
    'use_actions': True,
    'encoder': {
        'depths': [256, 512],
        'blocks': 1
    },
    'embedding_dim': 128,
    'codebook': {
        'n_codes': 1024,
        'proj_dim': 32
    },
    'z_ds': 4,
    'z_tfm_kwargs': {
        'embed_dim': 128,
        'mlp_dim': 512,
        'num_heads': 8,
        'num_layers': 2,
        'dropout': 0.,
        'attention_dropout': 0.
    },
    'seq_len': 32,
    'z_git': {
        'vocab_dim': 128, 
        'mask_schedule': 'cosine',
        'tfm_kwargs': {
            'embed_dim': 128,
            'mlp_dim': 512,
            'num_heads': 8,
            'num_layers': 2,
            'dropout': 0.,
            'attention_dropout': 0.
        }
    },
    'decoder': {
        'depths': [256, 512],
        'blocks': 1
    },
}
config = argparse.Namespace(**config)
vq_fns, vqvae = load_vqvae('/home/wilson/logs/hier_video/minerl_vqgan_jax', need_encode=False)
kwargs = {
    'config': config, 
    'vq_fns': vq_fns,
    'vqvae': vqvae
}

video = np.random.randint(0, 1024, size=(8, 32, 16, 16))
actions = np.random.randint(0, 3, size=(8, 32))
model_shard = TECOShard(**kwargs)
model = TECO(**kwargs)
spec = model_shard.model_spec

rng = jax.random.PRNGKey(0)
params = model.init({'params': rng, 'sample': rng}, video, actions, deterministic=True).unfreeze()['params']
params = jax.device_get(params)

params_sharded = spec.shard(params, num_shards)
params_unsharded = spec.unshard(params_sharded)

def check(p1, p2, parent=''):
    if isinstance(p1, np.ndarray):
        assert p1.shape == p2.shape, f'{parent}: {p1.shape} != {p2.shape}'
        assert np.allclose(p1, p2), f'error for {parent}'
        return
    
    k1, k2 = list(p1.keys()), list(p2.keys())
    k1.sort()
    k2.sort()
    assert tuple(k1) == tuple(k2), f'{parent}: {k1} != {k2}'

    for k in k1:
        check(p1[k], p2[k], parent=f'{parent}/{k}')

print('checking')
check(params, params_unsharded)
print('passed')


def step(params, video, actions):
    loss = model.apply({'params': params}, video, actions, 
                       deterministic=True, rngs={'sample': rng})['loss']
    return loss

def step_shard(params, video, actions):
    out = model_shard.apply({'params': params}, video, actions,
                             deterministic=True, rngs={'sample': rng})
    out = model_shard.aggregate_metrics(out)
    return out['loss']

print('Forward pass')

params = jax_utils.replicate(params)
step_pmap = jax.pmap(step, axis_name='data')
out_pmap = step_pmap(params, video.reshape(jax.local_device_count(), -1, *video.shape[1:]),
                     actions.reshape(jax.local_device_count(), -1, *actions.shape[1:]))
out_pmap = jax.device_get(out_pmap).reshape(-1)
print(np.mean(out_pmap), out_pmap.shape)

step_xmap = xmap(step_shard, in_axes=[spec.spec(), ('data', ...), ('data', ...)], out_axes=['data', ...], axis_resources={'data': 'dp', 'model': 'mp'})
with mesh:
    out_xmap = step_xmap(params_sharded, video.reshape(num_data, -1, *video.shape[1:]),
                         actions.reshape(num_data, -1, *actions.shape[1:]))
out_xmap = jax.device_get(out_xmap).reshape(-1)
print(np.mean(out_xmap), out_xmap.shape)

print('\nBackward pass')
def grad(params, video, actions):
    def loss_fn(params):
        variables = {'params': params}
        loss = model.apply(variables, video, actions, 
                           deterministic=True, rngs={'sample': rng})['loss']
        return loss
    grads = jax.value_and_grad(loss_fn)(params)[1]
    grads = lax.pmean(grads, axis_name='data')
    return grads
grad = jax.pmap(grad, axis_name='data')

def grad_shard(params, video, actions):
    def loss_fn(params):
        variables = {'params': params}
        loss = model_shard.apply(variables, video, actions,
                                 deterministic=True, rngs={'sample': rng})['loss']
        return loss
    grads = jax.value_and_grad(loss_fn)(params)[1]
    grads = spec.reduce_grad(grads)
    grads = lax.pmean(grads, axis_name='data')
    return grads
grad_shard = xmap(grad_shard, in_axes=[spec.spec(), ('data', ...), ('data', ...)], out_axes=spec.spec(), axis_resources={'data': 'dp', 'model': 'mp'})

g = grad(params, video.reshape(jax.local_device_count(), -1, *video.shape[1:]),
         actions.reshape(jax.local_device_count(), -1, *actions.shape[1:]))
g = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0], g)
g = spec.shard(g, num_shards)

with mesh:
    g_shard = grad_shard(params_sharded, video.reshape(num_data, -1, *video.shape[1:]),
                         actions.reshape(num_data, -1, *actions.shape[1:]))
g_shard = jax.tree_util.tree_map(lambda x: jax.device_get(x), g_shard)

def compute_errors(g1, g2, parent=''):
    if isinstance(g1, np.ndarray):
        err = np.max(np.abs(g1 - g2))
        print(f'{parent}: {err}')
        return [np.max(np.abs(g1 - g2))]
    k1, k2 = list(g1.keys()), list(g2.keys())
    k1.sort()
    k2.sort()
    assert tuple(k1) == tuple(k2), f'{parent}: {k1} != {k2}'

    errors = []
    for k in k1:
        errors.extend(compute_errors(g1[k], g2[k], parent=f'{parent}/{k}'))
    return errors
    
errors = compute_errors(g, g_shard)
print('errors:', min(errors), max(errors))

print('done')
