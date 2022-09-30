from typing import Any, Tuple, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


def constant(value, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jnp.full(shape, value, dtype=dtype)
    return init


class ResNetEncoder(nn.Module):
    depths: Tuple
    blocks: int
    dtype: Optional[Any] = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.depths[0], [3, 3], dtype=self.dtype)(x)
        x = ResNetBlock(self.depths[0], dtype=self.dtype)(x)
        for i in range(1, len(self.depths)):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2))
            for _ in range(self.blocks):
                x = ResNetBlock(self.depths[i], dtype=self.dtype)(x)
        return x
     
     
class ResNetDecoder(nn.Module):
    image_size: int
    depths: Tuple
    blocks: int
    out_dim: int
    dtype: Optional[Any] = jnp.float32
    
    @nn.compact
    def __call__(self, deter, embeddings=None):
        depths = list(reversed(self.depths))
        x = deter
        if embeddings is not None:
            x = jnp.concatenate([x, embeddings], axis=-1)

        x = nn.Conv(self.depths[0], [3, 3], dtype=self.dtype)(x)
            
        for i in range(len(depths) - 1):
            for _ in range(self.blocks):
                x = ResNetBlock(depths[i], dtype=self.dtype)(x)
            x = jax.image.resize(x, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
                                 jax.image.ResizeMethod.NEAREST)
        for _ in range(self.blocks):
            x = ResNetBlock(depths[-1], dtype=self.dtype)(x)
        x = nn.LayerNorm(dtype=self.dtype)(x)

        x = nn.Dense(self.out_dim, dtype=self.dtype)(x) 
        return x
     

class ResNetBlock(nn.Module):
    depth: int
    dtype: Optional[Any] = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        skip = x
        if skip.shape[-1] != self.depth:
            skip = nn.Conv(self.depth, [1, 1], use_bias=False, 
                           dtype=self.dtype, name='skip')(skip)

        x = nn.elu(nn.GroupNorm(dtype=self.dtype)(x))
        x = nn.Conv(self.depth, [3, 3], dtype=self.dtype)(x)
        x = nn.elu(nn.GroupNorm(dtype=self.dtype)(x))
        x = nn.Conv(self.depth, [3, 3], dtype=self.dtype, use_bias=False)(x)
        x = AddBias(dtype=self.dtype)(x)
        return skip + 0.1 * x 


class Codebook(nn.Module):
    n_codes: int
    proj_dim: int
    embedding_dim: int
    dtype: Optional[Any] = jnp.float32

    @nn.compact
    def __call__(self, z, encoding_indices=None):
        z = jnp.asarray(z, jnp.float32)
        
        # z: B...D
        codebook = self.param('codebook', nn.initializers.normal(stddev=0.02),
                              [self.n_codes, self.proj_dim])
        codebook = normalize(codebook)

        embedding_dim = self.embedding_dim
        proj_in = nn.Dense(self.proj_dim, use_bias=False)
        proj_out = nn.Dense(embedding_dim, use_bias=False)

        if encoding_indices is not None:
            z = codebook[(encoding_indices,)]
            z = proj_out(z)
            return z
        
        z_proj = normalize(proj_in(z))
        flat_inputs = jnp.reshape(z_proj, (-1, self.proj_dim))
        distances = 2 - 2 * flat_inputs @ codebook.T

        encoding_indices = jnp.argmin(distances, axis=1)
        encode_onehot = jax.nn.one_hot(encoding_indices, self.n_codes, dtype=flat_inputs.dtype)
        encoding_indices = jnp.reshape(encoding_indices, z.shape[:-1])

        quantized = codebook[(encoding_indices,)]

        commitment_loss = 0.25 * optax.l2_loss(z_proj, jax.lax.stop_gradient(quantized)).mean()
        codebook_loss = optax.l2_loss(jax.lax.stop_gradient(z_proj), quantized).mean()
        
        quantized_st = jax.lax.stop_gradient(quantized - z_proj) + z_proj
        quantized_st = proj_out(quantized_st)

        avg_probs = jnp.mean(encode_onehot, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        quantized_st = jnp.asarray(quantized_st, self.dtype)

        return dict(embeddings=quantized_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, codebook_loss=codebook_loss,
                    perplexity=perplexity)


class AddBias(nn.Module):
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],), self.param_dtype)
        x += bias
        return x


def normalize(x):
    x = x / jnp.clip(jnp.linalg.norm(x, axis=-1, keepdims=True), a_min=1e-6, a_max=None)
    return x
