from typing import Tuple, Optional, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from .utils import f_psum, g_psum, create_g_all_gather
from . import sharding
from ..base import AddBias


g_all_gather = create_g_all_gather(axis=3)


class ResNetDecoderShard(nn.Module):
    image_size: int
    num_shards: int
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

        x = f_psum(x)
        x = nn.Conv(self.depths[0] // self.num_shards, [3, 3], dtype=self.dtype)(x)
        x = g_all_gather(x)
        x = x.reshape(*x.shape[:-2], -1)
            
        for i in range(len(depths) - 1):
            for _ in range(self.blocks):
                x = ResNetBlock(self.num_shards, depths[i], dtype=self.dtype)(x)
            x = jax.image.resize(x, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
                                 jax.image.ResizeMethod.NEAREST)
        for _ in range(self.blocks):
            x = ResNetBlock(self.num_shards, depths[-1], dtype=self.dtype)(x)
        x = nn.LayerNorm(dtype=self.dtype)(x)

        assert self.out_dim % self.num_shards == 0
        x = f_psum(x)
        x = nn.Dense(self.out_dim // self.num_shards, dtype=self.dtype)(x) 
        x = g_all_gather(x)
        x = x.reshape(*x.shape[:-2], -1)
        
        return x
     
    @staticmethod        
    def model_spec(depths, blocks, **kwargs):
        spec = dict()
        spec['Conv_0'] = sharding.Conv(axis=1, use_bias=True)
        
        block_idx = 0
        cur_dim = depths[0]
        depths = list(reversed(depths))
        for i in range(len(depths) - 1):
            for _ in range(blocks):
                has_skip = cur_dim != depths[i]
                spec[f'ResNetBlock_{block_idx}'] = ResNetBlock.model_spec(has_skip)
                block_idx += 1
                cur_dim = depths[i]
        
        for _ in range(blocks):
            has_skip = cur_dim != depths[-1]
            spec[f'ResNetBlock_{block_idx}'] = ResNetBlock.model_spec(has_skip)
            block_idx += 1
            cur_dim = depths[-1]

        spec['LayerNorm_0'] = sharding.GenericReplicated(reduce_mode='identity')
        spec['Dense_0'] = sharding.Dense(axis=1, use_bias=True)
        return sharding.GenericDict(spec)
                


class ResNetBlock(nn.Module):
    num_shards: int
    depth: int
    dtype: Optional[Any] = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        # Input is unsharded / synced
        assert self.depth % self.num_shards == 0
        skip = x
        if skip.shape[-1] != self.depth:
            skip = f_psum(skip)
            skip = nn.Conv(self.depth // self.num_shards, [1, 1], use_bias=False, 
                           dtype=self.dtype, name='skip')(skip)
            skip = g_all_gather(skip)
            skip = skip.reshape(*skip.shape[:-2], -1)
            
        x = f_psum(x)
        x = nn.elu(nn.GroupNorm(dtype=self.dtype)(x))
        x = nn.Conv(self.depth // self.num_shards, [3, 3], dtype=self.dtype)(x)
        x = nn.elu(nn.GroupNorm(num_groups=32 // self.num_shards, dtype=self.dtype)(x))
        x = nn.Conv(self.depth, [3, 3], dtype=self.dtype, use_bias=False,
                    kernel_init=nn.initializers.variance_scaling(1.0 / self.num_shards, 'fan_in', 'normal'))(x)
        x = g_psum(x)
        x = AddBias(dtype=self.dtype)(x)
        return skip + 0.1 * x # Output is unsharded / synced 

    @staticmethod
    def model_spec(has_skip):
        spec = dict()
        if has_skip:
            spec['skip'] = sharding.Conv(axis=1, use_bias=False)
        
        spec.update({
            'GroupNorm_0': sharding.GenericReplicated(reduce_mode='sum'),
            'Conv_0': sharding.Conv(use_bias=True, axis=1),
            'GroupNorm_1': sharding.GroupNorm(),
            'Conv_1': sharding.Conv(use_bias=False, axis=0),
            'AddBias_0': sharding.GenericReplicated(reduce_mode='identity'),
        })
        return sharding.GenericDict(spec)
        
        
