import jax.numpy as jnp

from .. import load_vqvae
from .teco import TECOShard
from ..teco import TECO


def get_model(config, need_encode=None, **kwargs):
    if config.model == 'teco':
        if need_encode is None:
            need_encode = not 'encoded' in config.data_path
        vq_fns, vqvae = load_vqvae(config.vqvae_ckpt, need_encode)
        kwargs.update(vq_fns=vq_fns, vqvae=vqvae)

    kwargs['dtype'] = jnp.float32

    model_shard = TECOShard(config, **kwargs)
    model = TECO(config, **kwargs)

    return model, model_shard


def load_ckpt(ckpt_path, training=False, replicate=True, return_config=False, 
              default_if_none=dict(), need_encode=None, **kwargs):
    import os.path as osp
    import pickle
    from flax import jax_utils
    from flax.training import checkpoints
    from .train_utils import TrainState

    config = pickle.load(open(osp.join(ckpt_path, 'args'), 'rb'))
    for k, v in kwargs.items():
        setattr(config, k, v)
    for k, v in default_if_none.items():
        if not hasattr(config, k):
            print('did not find', k, 'setting default to', v)
            setattr(config, k, v)
    
    model, _ = get_model(config, need_encode=need_encode)
    state = checkpoints.restore_checkpoint(osp.join(ckpt_path, 'checkpoints'), None)
    state = TrainState(
        step=state['step'],
        params=state['params'],
        opt_state=state['opt_state'],
        model_state=state['model_state'],
        apply_fn=model.apply,
        tx=None
    )

    assert state is not None, f'No checkpoint found in {ckpt_path}'

    if replicate:
        state = jax_utils.replicate(state)

    if return_config:
        return model, state, config
    else:
        return model, state


def get_sample(config):
    from .teco import sample
    return sample
