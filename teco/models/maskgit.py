from typing import Optional, Any, Tuple, Dict, Callable
import math
import numpy as np
import jax
import flax.linen as nn
import jax.numpy as jnp

from ..utils import topk_sample
from .transformer import Transformer, LayerNorm
from .base import AddBias

Array = Any
Dtype = Any


MASK_ID = -1


def schedule(ratio, total_unknown, method='cosine'):
    if method == 'uniform':
        mask_ratio = 1. - ratio
    elif 'pow' in method:
        exponent = float(method.replace('pow', ''))
        mask_ratio = 1. - ratio ** exponent
    elif method == 'cosine':
        mask_ratio = jax.lax.cos(math.pi / 2. * ratio)
    elif method == 'log':
        mask_ratio = -jnp.log2(ratio) / jnp.log2(total_unknown)
    elif method == 'exp':
        mask_ratio = 1 - jnp.exp2(-jnp.log2(total_unknown) * (1 - ratio))
    mask_ratio = jnp.clip(mask_ratio, 1e-6, 1.)
    return mask_ratio

    
def mask_by_random_topk(rng, mask_len, probs, temperature=1.0):
    confidence = jnp.log(probs) + temperature * jax.random.gumbel(rng, probs.shape)
    sorted_confidence = jnp.sort(confidence, axis=-1)
    cut_off = jnp.take_along_axis(sorted_confidence, mask_len, axis=-1)
    masking = (confidence < cut_off)
    return masking


def sample_mask(Z, T, rng):
    N = np.prod(Z)
    idxs = jnp.arange(N, dtype=jnp.int32)
    idxs = jax.random.permutation(rng, idxs)
    chunks = jnp.array_split(idxs, T)

    masks = []
    for t in range(T):
        mask = jax.nn.one_hot(chunks[t], N).sum(axis=0).astype(bool)
        mask = jnp.reshape(mask, Z)
        masks.append(mask)
    return masks


class MaskGit(nn.Module):
    shape: Tuple[int]
    vocab_size: int
    vocab_dim: int
    mask_schedule: str
    tfm_kwargs: Dict[str, Any]
    dtype: Optional[Any] = jnp.float32

    def setup(self):
        self.token_embed = self.param('token_embed', nn.initializers.normal(stddev=0.02),
                                      [self.vocab_size + 1, self.vocab_dim], 
                                      jnp.float32)

        self.net = Transformer(
            **self.tfm_kwargs,
            shape=self.shape,
            pos_embed_type='broadcast',
            dtype=self.dtype
        )
        self.mlm = MlmLayer(self.vocab_dim, self.dtype)

    def _step(self, x, cond=None, deterministic=False):
        token_embed = jnp.asarray(self.token_embed, self.dtype)
        x = token_embed[(x,)]
 
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

        
class MlmLayer(nn.Module):
    vocab_dim: int
    dtype: Optional[Any] = jnp.float32

    @nn.compact
    def __call__(self, x, embeddings):
        x = nn.Dense(self.vocab_dim, dtype=self.dtype,
                     kernel_init=nn.initializers.normal(stddev=0.02))(x)
        x = nn.gelu(x)
        x = LayerNorm(dtype=self.dtype)(x)

        output_weights = jnp.transpose(embeddings)
        logits = jnp.matmul(x, output_weights)
        logits = AddBias(self.dtype)(logits)
        return logits
