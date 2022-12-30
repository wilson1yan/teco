import jax.numpy as jnp

from .sample import sample
from .vqgan import VQGAN
from .vae import VAE
from .teco import TECO
from .xmap.teco import TECOShard


def load_vqvae(ckpt_path, need_encode=True):
    import jax
    import argparse

    model, state = load_ckpt(ckpt_path, training=False, replicate=False)

    def wrap_apply(fn):
        variables = {'params': state.params, **state.model_state}
        return lambda *args: model.apply(variables, *args, method=fn)

    def no_encode(encodings):
        variables = {'params': state.params, **state.model_state}
        embeddings = model.apply(variables, encodings, method=model.codebook_lookup) 
        return embeddings, encodings

    video_encode = jax.jit(wrap_apply(model.encode)) if need_encode else jax.jit(no_encode)
    video_decode = jax.jit(wrap_apply(model.decode))
    codebook_lookup = jax.jit(wrap_apply(model.codebook_lookup))

    return dict(encode=video_encode, decode=video_decode, lookup=codebook_lookup), argparse.Namespace(latent_shape=model.latent_shape, embedding_dim=model.embedding_dim, n_codes=model.n_codes)


def load_ckpt(ckpt_path, replicate=True, return_config=False, 
              default_if_none=dict(), need_encode=None, **kwargs):
    import os.path as osp
    import pickle
    from flax import jax_utils
    from flax.training import checkpoints
    from ..train_utils import TrainState

    config = pickle.load(open(osp.join(ckpt_path, 'args'), 'rb'))
    for k, v in kwargs.items():
        setattr(config, k, v)
    for k, v in default_if_none.items():
        if not hasattr(config, k):
            print('did not find', k, 'setting default to', v)
            setattr(config, k, v)
    
    model = get_model(config, need_encode=need_encode)
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


def get_model(config, need_encode=None, xmap=False, **kwargs):
    if config.model == 'teco':
        if need_encode is None:
            need_encode = not 'encoded' in config.data_path
        vq_fns, vqvae = load_vqvae(config.vqvae_ckpt, need_encode)
        kwargs.update(vq_fns=vq_fns, vqvae=vqvae)

    kwargs['dtype'] = jnp.float32

    if config.model == 'vqgan':
        model = VQGAN(config, **kwargs)
    elif config.model == 'autoencoder':
        model = VAE(config, **kwargs)
    elif config.model == 'teco':
        model = TECO(config, **kwargs)
        if xmap:
            model_shard = TECOShard(config, **kwargs)
    else:
        raise ValueError(f'Invalid model: {config.model}')

    return (model, model_shard) if xmap else model

