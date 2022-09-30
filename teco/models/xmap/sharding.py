from dataclasses import dataclass
from typing import Any, Optional
import jax
import numpy as np


@dataclass
class GenericDict:
    model: Any

    def spec(self):
        return {k: v.spec() for k, v in self.model.items()}
    
    def unshard(self, v):
        return {k: self.model[k].unshard(v[k]) for k in self.model.keys()}
    
    def shard(self, v, num_shards):
        return {k: self.model[k].shard(v[k], num_shards) for k in self.model.keys()}
    
    def reduce_grad(self, v):
        return {k: self.model[k].reduce_grad(v[k]) for k in self.model.keys()}


@dataclass
class GenericShardedTensor:
    axis: int

    def spec(self):
        return ('model', ...)
    
    def unshard(self, v):
        return np.concatenate(list(v), axis=self.axis)

    def shard(self, v, num_shards):
        shape = (
            *v.shape[:self.axis], num_shards, 
            v.shape[self.axis] // num_shards,
            *v.shape[self.axis + 1:]
        )
        v = v.reshape(shape)

        permute_order = list(range(len(v.shape)))
        del permute_order[self.axis]
        permute_order = (self.axis, *permute_order)
        v = np.transpose(v, permute_order)
        return v

    def reduce_grad(self, v):
        return v


@dataclass
class GenericReplicated:
    reduce_mode: str

    def spec(self):
        return (...,)
    
    def unshard(self, v):
#        return jax.tree_util.tree_map(lambda x: x[0], v)
        return v

    def shard(self, v, num_shards):
        return v

    def reduce_grad(self, v):
        if self.reduce_mode == 'sum':
            reduce_fn = jax.lax.psum
        elif self.reduce_mode == 'mean':
            reduce_fn = jax.lax.pmean
        elif self.reduce_mode == 'identity':
            reduce_fn = lambda x, axis_name: x
        else:
            raise Exception(f'Invalid reduce_mean: {self.reduce_mode}')
        return jax.tree_util.tree_map(
            lambda x: reduce_fn(x, 'model'), v
        )

        
@dataclass
class LayerNormShard:
    use_scale: bool
    use_bias: bool
    
    def spec(self):
        return ('model', ...)
    
    def unshard(self, v):
        out = dict()
        if self.use_scale:
            out['scale'] = v['scale'].reshape(-1)
        if self.use_bias:
            out['bias'] = v['bias'].reshape(-1)
        return out

    def shard(self, v, num_shards):
        out = dict()
        if self.use_scale:
            out['scale'] = v['scale'].reshape(num_shards, -1)
        if self.use_bias:
            out['bias'] = v['bias'].reshape(num_shards, -1)
        return out

    def reduce_grad(self, v):
        return v

        
@dataclass
class GroupNorm:

    def spec(self):
        return ('model', ...)
    
    def unshard(self, v):
        out = dict()
        out['scale'] = v['scale'].reshape(-1)
        out['bias'] = v['bias'].reshape(-1)
        return out

    def shard(self, v, num_shards):
        out = dict()
        out['scale'] = v['scale'].reshape(num_shards, -1)
        out['bias'] = v['bias'].reshape(num_shards, -1)
        return out

    def reduce_grad(self, v):
        return v
    

        
@dataclass
class Conv:
    axis: int
    use_bias: bool = True

    def spec(self):
        return ('model', ...)

    def unshard(self, v):
        out = dict()
        kernel = v['kernel']
        
        if self.axis == 0: # input is sharded
            kernel = np.transpose(
                kernel, (1, 2, 0, 3, 4) # PHWIO -> HWPIO
            )
            out['kernel'] = kernel.reshape(*kernel.shape[:2], -1, kernel.shape[-1])
            assert not self.use_bias
        elif self.axis == 1:
            kernel = np.transpose(
                kernel, (1, 2, 3, 0, 4) # PHWIO -> HWIPO
            )
            out['kernel'] = kernel.reshape(*kernel.shape[:3], -1)
            if self.use_bias:
                out['bias'] = v['bias'].reshape(-1)
        else:
            raise Exception(f'Invalid axis: {self.axis}')
        return out

    def shard(self, v, num_shards):
        out = dict()
        kernel = v['kernel']

        if self.axis == 0:
            kernel = kernel.reshape( # HWIO -> HWPIO
                *kernel.shape[:2], num_shards, kernel.shape[-2] // num_shards, kernel.shape[-1]
            )
            out['kernel'] = np.transpose(kernel, (2, 0, 1, 3, 4))
            assert not self.use_bias, 'Does not support bias!'
        elif self.axis == 1:
            kernel = kernel.reshape( # HWIO -> HWIPO
                *kernel.shape[:2], kernel.shape[-2], num_shards, kernel.shape[-1] // num_shards
            ) 
            out['kernel'] = np.transpose(kernel, (3, 0, 1, 2, 4))
            if self.use_bias: 
                out['bias'] = v['bias'].reshape(
                    num_shards, v['bias'].shape[0] // num_shards
                )
        else:
            raise Exception(f'Unsupported axis {self.axis}')
        return out

    def reduce_grad(self, v):
        return v


@dataclass
class Dense:
    use_bias: bool = True
    axis: Optional[int] = None

    def spec(self):
        if self.axis is None:
            return (...,)
        else:
            return ('model', ...)
    
    def unshard(self, v):
        out = dict()
        kernel = v['kernel']
        if self.axis is None: # replicated
            out['kernel'] = kernel
            if self.use_bias:
                out['bias'] = v['bias']
        elif self.axis == 0:
            out['kernel'] = kernel.reshape(-1, kernel.shape[-1])
            assert not self.use_bias, 'Does not support bias!'
        elif self.axis == 1:
            out['kernel'] = kernel.transpose(1, 0, 2).reshape(kernel.shape[1], -1)
            if self.use_bias:
                out['bias'] = v['bias'].reshape(-1)
        else:
            raise Exception(f'Unsupported axis {self.axis}')
        return out

    def shard(self, v, num_shards):
        out = dict()
        kernel = v['kernel']
        if self.axis is None: # replicated
            out['kernel'] = kernel
            if self.use_bias:
                out['bias'] = v['bias']
        elif self.axis == 0:
            out['kernel'] = kernel.reshape(
                num_shards, kernel.shape[0] // num_shards, kernel.shape[1]
            )
            assert not self.use_bias, 'Does not support bias!'
        elif self.axis == 1:
            out['kernel'] = kernel.reshape(
                kernel.shape[0], num_shards, kernel.shape[1] // num_shards
            ).transpose(1, 0, 2)
            if self.use_bias:
                out['bias'] = v['bias'].reshape(
                    num_shards, v['bias'].shape[0] // num_shards
                )
        else:
            raise Exception(f'Unsupported axis {self.axis}')
        return out

    def reduce_grad(self, v):
        return v
        

@dataclass
class DenseGeneral:
    # Currently only supports the DenseGenerals for the attention layer
    use_bias: bool = True
    shard_mode: str = 'replicate'

    def spec(self):
        if self.shard_mode == 'replicate':
            return (...,)
        else:
            return ('model', ...)

    def unshard(self, v):
        out = dict()
        kernel = v['kernel']
        if self.shard_mode == 'replicate': # replicated
            out['kernel'] = kernel[0]
            if self.use_bias:
                out['bias'] = v['bias'][0]
        elif self.shard_mode == 'in':
            out['kernel'] = kernel.reshape(
                -1, *kernel.shape[2:]
            )
            assert not self.use_bias
        elif self.shard_mode == 'out':
            out['kernel'] = kernel.transpose(1, 0, 2, 3).reshape(
                kernel.shape[1], -1, kernel.shape[-1]
            )
            if self.use_bias:
                out['bias'] = v['bias'].reshape(-1, v['bias'].shape[-1])
        else:
            raise Exception(f'Unsupported shard_mode {self.shard_mode}')
        return out

    def shard(self, v, num_shards):
        out = dict() 
        kernel = v['kernel']
        if self.shard_mode == 'replicate':
            out['kernel'] = kernel
            if self.use_bias:
                out['bias'] = v['bias']
        elif self.shard_mode == 'in':
            out['kernel'] = kernel.reshape(
                num_shards, kernel.shape[0] // num_shards, *kernel.shape[1:]
            )
            assert not self.use_bias
        elif self.shard_mode == 'out':
            out['kernel'] = kernel.reshape(
                kernel.shape[0], num_shards, 
                kernel.shape[1] // num_shards, 
                *kernel.shape[2:]
            ).transpose(1, 0, 2, 3)
            if self.use_bias:
                out['bias'] = v['bias'].reshape(
                    num_shards, v['bias'].shape[0] // num_shards, *v['bias'].shape[1:]
                )
        else:
            raise Exception(f'Unsupported shard_mode {self.shard_mode}')
        return out

    def reduce_grad(self, v):
        return v
