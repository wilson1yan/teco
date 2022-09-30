from typing import Optional, Any, Tuple
from functools import partial
import numpy as np
import flax.linen as nn
import flax.linen.initializers as nn_init
import jax
import jax.numpy as jnp

from .utils import f_psum, g_psum, create_g_all_gather
from . import sharding
from ..transformer import *


class TransformerShard(nn.Module):
    num_shards: int
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
    use_fc_in: str = True
    right_shift: bool = False
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=False, cond=None):
        g_all_gather = create_g_all_gather(axis=2) 
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

        old_shape = x.shape[1:-1]
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.use_fc_in:
            x = f_psum(x)
            x = nn.Dense(self.embed_dim // self.num_shards, dtype=self.dtype)(x)
            x = g_all_gather(x)
            x = x.reshape(*x.shape[:-2], -1)
        else:
            assert self.fc_in_mode is None
        
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

        for i in range(self.num_layers):
            x = TransformerLayerShard(
                num_shards=self.num_shards,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                shape=self.shape,
                dtype=self.dtype,
                name=f'TransformerLayer_{i}'
            )(x, mask=mask, deterministic=deterministic)
        
        x = f_psum(x)
        x = nn.Dense(
            self.embed_dim // self.num_shards, dtype=self.dtype
        )(x)
        x = nn.gelu(x)
        x = LayerNormShard(dtype=self.dtype, name='LayerNorm_0')(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        if self.out_dim is not None:
            x = nn.Dense(self.out_dim, dtype=jnp.float32, use_bias=False,
                         kernel_init=nn.initializers.variance_scaling(
                             1.0 / self.num_shards, 'fan_in', 'normal'
                        ))(x) 
            x = g_psum(x)
            x = AddBias(dtype=jnp.float32)(x)
        else:
            x = g_all_gather(x)
            x = x.reshape(*x.shape[:-2], -1)
        
        x = x.reshape(x.shape[0], *old_shape, x.shape[-1])
        return x

    @staticmethod
    def model_spec(vocab_size=None, vocab_dim=None, use_fc_in=True, right_shift=False,
                   pos_embed_type='absolute', num_layers=1, out_dim=None, **kwargs):
        spec = dict()
        if vocab_size is not None and vocab_dim is not None:
            spec['Embed_0'] = sharding.GenericReplicated(reduce_mode='identity')
        
        dense_idx, bias_idx = 0, 0
        if use_fc_in:
            spec[f'Dense_{dense_idx}'] = sharding.Dense(use_bias=True, axis=1)
            dense_idx += 1
        
        if right_shift:
            spec['RightShift_0'] = sharding.GenericReplicated(reduce_mode='identity')
        
        if pos_embed_type == 'absolute':
            spec['AbsolutePositionBiases_0'] = sharding.GenericReplicated(reduce_mode='identity')
        elif pos_embed_type == 'broadcast':
            spec['BroadcastPositionBiases_0'] = sharding.GenericReplicated(reduce_mode='identity')
        
        for i in range(num_layers):
            spec[f'TransformerLayer_{i}'] = TransformerLayerShard.model_spec()
        
        spec[f'Dense_{dense_idx}'] = sharding.Dense(use_bias=True, axis=1)
        dense_idx += 1
        spec['LayerNorm_0'] = LayerNormShard.model_spec(use_bias=True, use_scale=True)

        if out_dim is not None:
            spec[f'Dense_{dense_idx}'] = sharding.Dense(use_bias=False, axis=0)
            spec[f'AddBias_{bias_idx}'] = sharding.GenericReplicated(reduce_mode='identity')

        return sharding.GenericDict(spec)
        

class TransformerLayerShard(nn.Module):
    num_shards: int
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout: float
    attention_dropout: float
    shape: Optional[Tuple] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, mask=None, deterministic=False):
        x = f_psum(inputs)
        x = LayerNorm(dtype=self.dtype)(x)

        x = MultiHeadAttentionShard(
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            num_shards=self.num_shards,
            dropout_rate=self.attention_dropout,
            dtype=self.dtype,
            name='MultiHeadAttention_0'
        )(x, x, mask=mask,
            deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = x + inputs

        y = f_psum(x)
        y = LayerNorm(dtype=self.dtype)(y)
        y = MlpBlockShard(
            num_shards=self.num_shards,
            intermediate_dim=self.mlp_dim,
            intermediate_dropout_rate=self.dropout,
            dtype=self.dtype,
            name='MlpBlock_0'
        )(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        y = y + x
        
        return y

    @staticmethod
    def model_spec():
        return sharding.GenericDict({
            'LayerNorm_0': sharding.GenericReplicated(reduce_mode='sum'),
            'MultiHeadAttention_0': MultiHeadAttentionShard.model_spec(),
            'LayerNorm_1': sharding.GenericReplicated(reduce_mode='sum'),
            'MlpBlock_0': MlpBlockShard.model_spec()
        })


class MlpBlockShard(nn.Module):
    num_shards: int
    intermediate_dim: int
    kernel_init: Any = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, deterministic=False):
        assert self.intermediate_dim % self.num_shards == 0
        x = nn.Dense(
            self.intermediate_dim // self.num_shards,
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
            kernel_init=nn.initializers.variance_scaling(
                1.0 / self.num_shards, 'fan_in', 'truncated_normal'
            ),  # since input to this dense is D / num_shards
            use_bias=False,
            name='wo'
        )(x)

        x = g_psum(x)
        x = AddBias(name='wo_bias')(x)

        return x

    @staticmethod
    def model_spec():
        return sharding.GenericDict({
            'wi': sharding.Dense(use_bias=True, axis=1),
            'wo': sharding.Dense(use_bias=False, axis=0),
            'wo_bias': sharding.GenericReplicated(reduce_mode='identity')
        })

 
class MultiHeadAttentionShard(nn.Module):
    num_heads: int
    head_dim: int
    num_shards: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.
    kernel_init: Any = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None, deterministic=False):
        assert self.num_heads % self.num_shards == 0
        num_heads_per_shard = self.num_heads // self.num_shards

        projection = partial(
            nn.DenseGeneral,
            axis=-1,
            features=(num_heads_per_shard, self.head_dim),
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
            kernel_init=nn.initializers.variance_scaling(
                1.0 / self.num_shards, 'fan_in', 'normal' 
            ), # to account for only partial activations in fan_in init
            dtype=self.dtype,
            use_bias=False,
            name='out'
        )(x)
        out = g_psum(out)
        out = AddBias(name='out_bias')(out)

        return out

    @staticmethod
    def model_spec():
        return sharding.GenericDict({
            'query': sharding.DenseGeneral(use_bias=True, shard_mode='out'),
            'key': sharding.DenseGeneral(use_bias=True, shard_mode='out'),
            'value': sharding.DenseGeneral(use_bias=True, shard_mode='out'),
            'out': sharding.DenseGeneral(use_bias=False, shard_mode='in'),
            'out_bias': sharding.GenericReplicated(reduce_mode='identity')
        })

        
class LayerNormShard(nn.Module):
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
        mean = jax.lax.pmean(mean, 'model')
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        mean2 = jax.lax.pmean(mean2, 'model')
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

    @staticmethod
    def model_spec(use_bias, use_scale):
        return sharding.LayerNormShard(use_bias=use_bias, use_scale=use_scale)
