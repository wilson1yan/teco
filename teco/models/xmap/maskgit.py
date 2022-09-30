from typing import Optional, Any, Tuple, Dict, Callable
import math
import numpy as np
import jax
import flax.linen as nn
import jax.numpy as jnp

from . import transformer
from . import sharding
from .utils import f_psum, g_psum, create_g_all_gather
from ..maskgit import MASK_ID, schedule, mask_by_random_topk, sample_mask
from ...utils import topk_sample


class MaskGitShard(nn.Module):
    num_shards: int
    shape: Tuple[int]
    vocab_size: int
    vocab_dim: int
    mask_schedule: str
    tfm_kwargs: Dict[str, Any]
    dtype: Optional[Any] = jnp.float32

    def setup(self):
        self.token_embed = self.param('token_embed', nn.initializers.normal(stddev=0.02),
                                      [self.vocab_size + 1, self.vocab_dim // self.num_shards], 
                                      jnp.float32)

        self.net = transformer.TransformerShard(
            **self.tfm_kwargs,
            num_shards=self.num_shards,
            shape=self.shape,
            pos_embed_type='broadcast',
            dtype=self.dtype
        )
        self.mlm = MlmLayerShard(self.num_shards, self.vocab_dim, self.dtype)

    def _step(self, x, cond=None, deterministic=False):
        g_all_gather = create_g_all_gather(axis=1 + len(self.shape))
        token_embed = jnp.asarray(self.token_embed, self.dtype)
        x = token_embed[(x,)]
        x = g_all_gather(x)
        x = x.reshape(*x.shape[:-2], -1) 
        x = self.net(x, cond=cond, deterministic=deterministic)
        logits = self.mlm(x, self.token_embed[:self.vocab_size])
        return logits
        

    def sample(self, n, T_draft, T_revise, M, cond=None):
        sample = jnp.full((n, *self.shape), MASK_ID, dtype=jnp.int32)

        def _update(samples, masks):
            for mask in masks:
                samples = jnp.where(mask, MASK_ID, samples)
                logits = self._step(samples, cond=cond, deterministic=True)
                s = topk_sample(self.make_rng('sample'), logits)
                samples = jnp.where(mask, s, samples)
            return samples
        
        # Draft
        masks = sample_mask(self.shape, T_draft, self.make_rng('sample'))
        sample = _update(sample, masks)
        
        # Revise
        for _ in range(M):
            masks = sample_mask(self.shape, T_revise, self.make_rng('sample'))
            sample = _update(sample, masks)
        
        return sample

    def __call__(self, x, cond=None, deterministic=False):
        # x: B..., cond: B...D
        # TODO check rng, xs should be replicated on all model shards
        B, L = x.shape[0], np.prod(x.shape[1:])

        ratio = jax.random.uniform(self.make_rng('sample'), shape=(B,), dtype=self.dtype)
        ratio = schedule(ratio, L, method=self.mask_schedule)
        ratio = jnp.maximum(1, jnp.floor(ratio * L))

        sample = jnp.arange(L)[None, :].repeat(B, axis=0)
        sample = jax.random.permutation(self.make_rng('sample'), sample, axis=-1, independent=True)
        mask = sample < ratio[:, None]
        mask = mask.reshape(x.shape)

        masked_x = jnp.where(mask, MASK_ID, x)
        logits = self._step(masked_x, cond=cond, deterministic=deterministic)
        labels = jax.nn.one_hot(x, num_classes=self.vocab_size)
        return logits, labels, mask

    @staticmethod
    def model_spec(tfm_kwargs, **kwargs):
        return sharding.GenericDict({
            'token_embed': sharding.GenericShardedTensor(axis=1),
            'net': transformer.TransformerShard.model_spec(
                pos_embed_type='broadcast', num_layers=tfm_kwargs['num_layers']
            ),
            'mlm': MlmLayerShard.model_spec()
        })
    
        
class MlmLayerShard(nn.Module):
    num_shards: int
    vocab_dim: int
    dtype: Optional[Any] = jnp.float32

    @nn.compact
    def __call__(self, x, embeddings):
        x = f_psum(x)
        x = nn.Dense(self.vocab_dim // self.num_shards, dtype=self.dtype,
                     kernel_init=nn.initializers.normal(stddev=0.02))(x)
        x = nn.gelu(x)
        x = transformer.LayerNormShard(dtype=self.dtype, name='LayerNorm_0')(x)

        output_weights = jnp.transpose(embeddings)
        logits = jnp.matmul(x, output_weights)
        logits = g_psum(logits)
        logits = transformer.AddBias(self.dtype)(logits)
        return logits

    @staticmethod
    def model_spec():
        return sharding.GenericDict({
            'Dense_0': sharding.Dense(use_bias=True, axis=1),
            'LayerNorm_0': transformer.LayerNormShard.model_spec(use_bias=True, use_scale=True),
            'AddBias_0': sharding.GenericReplicated(reduce_mode='identity')
        })
