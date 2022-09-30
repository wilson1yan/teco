from typing import Optional, Any, Dict, Callable
from functools import cached_property
import optax
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp

from .utils import create_g_all_gather, create_g_index_select, f_pmean
from .transformer import TransformerShard
from ..base import ResNetEncoder, Codebook
from .decoder import ResNetDecoderShard
from .maskgit import MaskGitShard
from . import sharding


class TECOShard(nn.Module):
    config: Any
    vq_fns: Dict[str, Callable]
    vqvae: Any
    dtype: Optional[Any] = jnp.float32

    @property
    def metrics(self):
        metrics = ['loss', 'recon_loss', 'trans_loss', 'codebook_loss', 
                   'commitment_loss', 'perplexity']
        return metrics

    def setup(self):
        config = self.config

        self.action_embeds = nn.Embed(config.action_dim + 1, config.action_embed_dim, dtype=self.dtype)

        # Posterior
        self.sos_post = self.param('sos_post', nn.initializers.normal(stddev=0.02),
                                   (*self.vqvae.latent_shape, self.vqvae.embedding_dim), jnp.float32)
        self.encoder = nn.Sequential([
            ResNetEncoder(**config.encoder, dtype=self.dtype),
            nn.Dense(config.embedding_dim, dtype=self.dtype)
        ])
        ds = 2 ** (len(config.encoder['depths']) - 1)
        self.z_shape = tuple([d // ds for d in self.vqvae.latent_shape])
        self.codebook = Codebook(**self.config.codebook, embedding_dim=config.embedding_dim, 
                                 dtype=self.dtype)
        
        # Prior
        z_kernel = [config.z_ds, config.z_ds]
        self.z_tfm_shape = tuple([d // config.z_ds for d in self.z_shape])
        self.z_proj = nn.Conv(config.z_tfm_kwargs['embed_dim'], z_kernel,
                              strides=z_kernel, use_bias=False, padding='VALID', dtype=self.dtype)

        self.sos = self.param('sos', nn.initializers.normal(stddev=0.02),
                              (*self.z_tfm_shape, config.z_tfm_kwargs['embed_dim'],), jnp.float32)
        self.z_tfm = TransformerShard(
            **config.z_tfm_kwargs, pos_embed_type='sinusoidal',
            shape=(config.seq_len, *self.z_tfm_shape), 
            num_shards=config.num_shards, dtype=self.dtype,
        )
        self.z_unproj = nn.ConvTranspose(config.embedding_dim, z_kernel, strides=z_kernel,
                                         padding='VALID', use_bias=False, dtype=self.dtype)

        # Dynamics Prior
        self.z_git = nn.vmap(
            MaskGitShard, in_axes=(1, 1, None), out_axes=1,
            variable_axes={'params': None},
            split_rngs={'sample': True, 'dropout': True, 'params': False}
        )(
            shape=self.z_shape, vocab_size=self.codebook.n_codes,
            **config.z_git, dtype=self.dtype, num_shards=self.config.num_shards
        )

        # Decoder
        out_dim = self.vqvae.n_codes
        self.decoder = ResNetDecoderShard(**config.decoder, image_size=self.vqvae.latent_shape[0], 
                                         num_shards=self.config.num_shards, out_dim=out_dim, dtype=self.dtype)

    def sample_timestep(self, z_embeddings, actions, cond, t):
        t -= self.config.n_cond
        actions = self.action_embeds(actions)
        deter = self.temporal_transformer(
            z_embeddings, actions, cond, deterministic=True
        )
        deter = deter[:, t]

        sample = self.z_git.sample(z_embeddings.shape[0], self.config.T_draft,
                                   self.config.T_revise, self.config.M,
                                   cond=deter)
 
        z_t = self.codebook(None, encoding_indices=sample)

        recon = jnp.argmax(self.decoder(deter, z_t), axis=-1)
        return z_t, recon

    def _init_mask(self):
        n_per = np.prod(self.z_tfm_shape)
        mask = jnp.tril(jnp.ones((self.config.seq_len, self.config.seq_len), dtype=bool))
        mask = mask.repeat(n_per, axis=0).repeat(n_per, axis=1)
        return mask

    def encode(self, encodings):
        g_all_gather = create_g_all_gather(axis=1)
        g_index_select = create_g_index_select(axis=1)
        
        embeddings = self.vq_fns['lookup'](encodings)
        sos = jnp.tile(self.sos_post[None, None], (embeddings.shape[0], 1, 1, 1, 1))
        sos = jnp.asarray(sos, self.dtype)   
        embeddings = jnp.concatenate([sos, embeddings], axis=1)
        inp = jnp.concatenate([embeddings[:, :-1], embeddings[:, 1:]], axis=-1)

        # Each model device encodes a subset of the frames
        if self.config.seq_len % self.config.num_shards != 0:
            n_pad = self.config.num_shards - self.config.seq_len % self.config.num_shards
            pad = inp[:, -n_pad:] # pad using real data since still used for VQ loss
            inp = jnp.concatenate([inp, pad], axis=1)

        inp = inp.reshape(inp.shape[0], self.config.num_shards,
                          inp.shape[1] // self.config.num_shards,
                          *inp.shape[2:])
        inp = g_index_select(inp)
        
        out = jax.vmap(self.encoder, 1, 1)(inp) 
        vq_output = self.codebook(out)

        # Gather back
        out = g_all_gather(out)
        out = out.reshape(out.shape[0], -1, *out.shape[3:])

        vq_embeddings, vq_encodings = vq_output['embeddings'], vq_output['encodings']

        vq_embeddings = g_all_gather(vq_embeddings)
        vq_embeddings = vq_embeddings.reshape(vq_embeddings.shape[0], -1, *vq_embeddings.shape[3:])

        vq_encodings = g_all_gather(vq_encodings)
        vq_encodings = vq_encodings.reshape(vq_encodings.shape[0], -1, *vq_encodings.shape[3:])

        if self.config.seq_len % self.config.num_shards != 0:
            out = out[:, :-n_pad]
            vq_embeddings = vq_embeddings[:, :-n_pad]
            vq_encodings = vq_encodings[:, :-n_pad]

        vq_output = {
            'embeddings': vq_embeddings[:, self.config.n_cond:],
            'encodings': vq_encodings[:, self.config.n_cond:],
            'commitment_loss': vq_output['commitment_loss'],
            'codebook_loss': vq_output['codebook_loss'],
            'perplexity': vq_output['perplexity'],
        }
        
        return out[:, :self.config.n_cond], vq_output

    def temporal_transformer(self, z_embeddings, actions, cond, deterministic=False):
        inp = jnp.concatenate([cond, z_embeddings], axis=1)
        actions = jnp.tile(actions[:, :, None, None], (1, 1, *inp.shape[2:4], 1)) 
        inp = jnp.concatenate([inp[:, :-1], actions[:, 1:]], axis=-1)
        inp = jax.vmap(self.z_proj, 1, 1)(inp)
        
        sos = jnp.tile(self.sos[None, None], (z_embeddings.shape[0], 1, 1, 1, 1))
        sos = jnp.asarray(sos, self.dtype)

        inp = jnp.concatenate([sos, inp], axis=1)
        deter = self.z_tfm(inp, mask=self._init_mask(), deterministic=deterministic)
        deter = deter[:, self.config.n_cond:]
        deter = f_pmean(deter)

        deter = jax.vmap(self.z_unproj, 1, 1)(deter)

        return deter

    def __call__(self, video, actions, deterministic=False):
        # video: BTCHW, actions: BT
        if not self.config.use_actions:
            if actions is None:
                actions = jnp.zeros(video.shape[:2], dtype=jnp.int32)
            else:
                actions = jnp.zeros_like(actions)
 
        if self.config.dropout_actions:
            dropout_actions = jax.random.bernoulli(self.make_rng('sample'), p=0.5,
                                                shape=(video.shape[0],)) # B
            actions = jnp.where(dropout_actions[:, None], -1, actions)

        actions = self.action_embeds(actions)
        _, encodings = self.vq_fns['encode'](video)
        
        cond, vq_output = self.encode(encodings)
        z_embeddings, z_codes = vq_output['embeddings'], vq_output['encodings']

        deter = self.temporal_transformer(
            z_embeddings, actions, cond, deterministic=deterministic
        )
        
        encodings = encodings[:, self.config.n_cond:]
        labels = jax.nn.one_hot(encodings, num_classes=self.vqvae.n_codes)
        labels = labels * 0.99 + 0.01 / self.vqvae.n_codes # Label smoothing

        if self.config.drop_loss_rate is not None and self.config.drop_loss_rate > 0.0:
            n_sample = int((1 - self.config.drop_loss_rate) * deter.shape[1])
            n_sample = max(1, n_sample)
            idxs = jax.random.randint(self.make_rng('sample'),
                                      [n_sample],
                                      0, video.shape[1], dtype=jnp.int32)
        else:
            idxs = jnp.arange(deter.shape[1], dtype=jnp.int32)

        deter = deter[:, idxs]
        z_embeddings = z_embeddings[:, idxs]
        z_codes = z_codes[:, idxs]
        labels = labels[:, idxs]
            
        # Dynamics Prior loss
        z_logits, z_labels, z_mask = self.z_git(z_codes, deter, deterministic)
        trans_loss = optax.softmax_cross_entropy(z_logits, z_labels)
        trans_loss = (trans_loss * z_mask).sum() / z_mask.sum()
        trans_loss = trans_loss * np.prod(self.z_shape)
        
        # Decoder loss
        recon_logits = jax.vmap(self.decoder, 1, 1)(deter, z_embeddings)
        recon_loss = optax.softmax_cross_entropy(recon_logits, labels)
        recon_loss = recon_loss.sum(axis=(-2, -1)).mean()

        ns = self.config.num_shards
        loss = recon_loss + trans_loss + vq_output['commitment_loss'] / ns + \
             vq_output['codebook_loss'] / ns
 
        out = dict(loss=loss, recon_loss=recon_loss, trans_loss=trans_loss,
                   commitment_loss=vq_output['commitment_loss'],
                   codebook_loss=vq_output['codebook_loss'],
                   perplexity=vq_output['perplexity'])
        return out

    def aggregate_metrics(self, metrics):
        metrics = jax.lax.pmean(metrics, 'model')
        metrics['loss'] = metrics['recon_loss'] + metrics['trans_loss'] + metrics['commitment_loss'] + metrics['codebook_loss']
        metrics = jax.lax.pmean(metrics, 'data')
        return metrics

    @cached_property
    def model_spec(self):
        ds = 2 ** (len(self.config.encoder['depths']) - 1)
        z_shape = tuple([d // ds for d in self.vqvae.latent_shape])
        z_tfm_shape = tuple([d // self.config.z_ds for d in z_shape])
        return sharding.GenericDict({
            'action_embeds': sharding.GenericReplicated(reduce_mode='identity'),
            'sos_post': sharding.GenericReplicated(reduce_mode='identity'),
            'encoder': sharding.GenericReplicated(reduce_mode='sum'),
            'codebook': sharding.GenericReplicated(reduce_mode='sum'),
            'z_proj': sharding.GenericReplicated(reduce_mode='identity'),
            'sos': sharding.GenericReplicated(reduce_mode='identity'),
            'z_tfm': TransformerShard.model_spec(
                **self.config.z_tfm_kwargs, pos_embed_type='sinusoidal',
                shape=(self.config.seq_len, *z_tfm_shape)
            ),
            'z_unproj': sharding.GenericReplicated(reduce_mode='identity'),
            'z_git': MaskGitShard.model_spec(**self.config.z_git),
            'decoder': ResNetDecoderShard.model_spec(**self.config.decoder)
        })
        
        
