from typing import Optional, Any, Tuple
from functools import partial
import numpy as np
import flax.linen as nn
import flax.linen.initializers as nn_init
import jax
import jax.numpy as jnp

from .base import AddBias


class Transformer(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout: float
    attention_dropout: float
    vocab_size: Optional[int] = None
    vocab_dim: Optional[int] = None
    shape: Optional[Tuple[int]] = None
    pos_embed_type: str = 'absolute'
    out_dim: Optional[int] = None
    use_fc_in: bool = True
    right_shift: bool = False
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=False, cond=None):
        x = inputs

        if self.vocab_size is not None and self.vocab_dim is not None:
            x = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.vocab_dim,                
                dtype=self.dtype,
                embedding_init=nn.initializers.normal(stddev=1.0)
            )(x)

        if cond is not None:
            x = jnp.concatenate((x, cond), axis=-1) 

        if self.use_fc_in:
            x = nn.Dense(self.embed_dim, dtype=self.dtype)(x)
        
        old_shape = x.shape[1:-1]
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.right_shift:
            x = RightShift(self.dtype)(x)
            
        if self.pos_embed_type == 'absolute':
            position_bias = AbsolutePositionBiases(dtype=self.dtype)(x)
        elif self.pos_embed_type == 'broadcast':
            position_bias = BroadcastPositionBiases(shape=self.shape,
                                                    dtype=self.dtype)(x)
        elif self.pos_embed_type == 'sinusoidal':
            position_bias = SinusoidalPositionBiases(dtype=self.dtype)(x)
        elif self.pos_embed_type == 'none':
            position_bias = None
        else:
            raise Exception(f'Invalid pos_embed_type: {self.pos_embed_type}')

        if position_bias is not None:
            x += position_bias

        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = x.astype(self.dtype)

        for _ in range(self.num_layers):
            x = TransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                shape=self.shape,
                dtype=self.dtype
            )(x, mask=mask, deterministic=deterministic)
        
        x = nn.Dense(
            self.embed_dim, dtype=self.dtype
        )(x)
        x = nn.gelu(x)
        x = LayerNorm(dtype=self.dtype)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        if self.out_dim is not None:
            x = nn.Dense(self.out_dim, dtype=jnp.float32, use_bias=False)(x)
            x = AddBias()(x)
        
        x = x.reshape(x.shape[0], *old_shape, x.shape[-1])
        return x 


class TransformerLayer(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout: float
    attention_dropout: float
    shape: Optional[Tuple] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=False):
        x = LayerNorm(dtype=self.dtype)(inputs)

        x = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            dropout_rate=self.attention_dropout,
            dtype=self.dtype
        )(x, x, mask=mask, 
            deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = x + inputs

        y = LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            intermediate_dim=self.mlp_dim,
            intermediate_dropout_rate=self.dropout,
            dtype=self.dtype
        )(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        y = y + x
        
        return y


class MlpBlock(nn.Module):
    intermediate_dim: int
    kernel_init: Any = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        x = nn.Dense(
            self.intermediate_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='wi'
        )(inputs)
        x = gelu2(x)
        x = nn.Dropout(rate=self.intermediate_dropout_rate,
                       broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = nn.Dense(
            inputs.shape[-1],
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            use_bias=False,
            name='wo'
        )(x)
        x = AddBias(name='wo_bias')(x)

        return x

 
class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.
    kernel_init: Any = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None, deterministic=False):
        projection = partial(
            nn.DenseGeneral,
            axis=-1,
            features=(self.num_heads, self.head_dim),
            dtype=self.dtype
        )

        query = projection(kernel_init=self.kernel_init, name='query')(inputs_q)
        key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
        value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

        if mask is not None:
            attention_bias = jax.lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype)
            )
        else:
            attention_bias = None
        
        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng('dropout')

        x = nn.attention.dot_product_attention(
            query, key, value, bias=attention_bias,
            dropout_rng=dropout_rng, dropout_rate=self.dropout_rate,
            deterministic=deterministic, dtype=self.dtype
        )
        out = nn.DenseGeneral(
            features=inputs_q.shape[-1],
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            use_bias=False,
            name='out'
        )(x)
        out = AddBias(name='out_bias')(out)

        return out

         
class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: Optional[Any] = None
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Any = nn_init.zeros
    scale_init: Any = nn_init.ones
    reduction_axes: Any = -1
    feature_axes: Any = -1

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]
        x = jnp.asarray(x, jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        var = jnp.maximum(0., mean2 - jax.lax.square(mean))

        y = x - mean
        mul = jax.lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale', self.scale_init, (features,), jnp.float32)
            mul *= scale
        y *= mul

        if self.use_bias:
            bias = self.param('bias', self.bias_init, (features,), jnp.float32)
            y += bias
        
        y = jnp.asarray(y, self.dtype)
        return y


class RightShift(nn.Module):
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        sos = self.param('sos', nn.initializers.normal(stddev=0.02),
                         (x.shape[-1],), self.dtype)
        sos = jnp.tile(sos[None, None], (x.shape[0], 1, 1))
        sos = jnp.asarray(sos, self.dtype)
        x = jnp.concatenate([sos, x[:, :-1]], axis=1)
        return x

        
class SinusoidalPositionBiases(nn.Module):
    shape: Optional[Tuple[int]] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        embed_dim = x.shape[-1]
        length = np.prod(self.shape or x.shape[1:-1])
        pos_seq = jnp.arange(length, dtype=self.dtype)

        inv_freq = jnp.arange(0.0, embed_dim, 2.0) / embed_dim
        inv_freq = 1. / (10000 ** inv_freq)
        inv_freq = jnp.asarray(inv_freq, self.dtype)

        sinusoid_inp = jnp.outer(pos_seq, inv_freq)
        position_bias = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
        return position_bias

        
class AbsolutePositionBiases(nn.Module):
    dtype: Any = jnp.float32
    embedding_init: Any = nn.linear.default_embed_init

    @nn.compact
    def __call__(self, x):
        position_bias = self.param('abs_embedding', self.embedding_init,
                                   x.shape[1:], jnp.float32)
        return position_bias
    

class BroadcastPositionBiases(nn.Module):
    shape: Optional[Tuple[int]] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):    
        shape = self.shape or x.shape[1:-1]
        n_dim = len(self.shape)
        embed_dim = x.shape[-1]

        chunk_sizes = [embed_dim // n_dim + (i < (embed_dim % n_dim))
                       for i in range(n_dim)]
        assert sum(chunk_sizes) == embed_dim, f'sum({chunk_sizes}) = {sum(chunk_sizes)} != {embed_dim}'

        embs = [
            self.param(f'd_{i}', nn.initializers.normal(stddev=0.02),
                            (shape[i], chunk_sizes[i]), jnp.float32)
            for i in range(n_dim)
        ]

        out = []
        for i in range(n_dim):
            e = embs[i]
            e = jnp.reshape(e, (1,) + (1,) * i + (shape[i],) + (1,) * (n_dim - i - 1) + (-1,))
            e = jnp.broadcast_to(e, (1, *shape, e.shape[-1]))
            out.append(e)
        out = jnp.concatenate(out, axis=-1)
        out = jnp.asarray(out, self.dtype)

        out = jnp.reshape(out, (np.prod(shape), embed_dim))

        return out    


def gelu2(x):
    return nn.sigmoid(1.702 * x) * x
