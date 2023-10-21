from typing import Any, Tuple, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np


def constant(value, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jnp.full(shape, value, dtype=dtype)
    return init


class Encoder(nn.Module):
    """
    Encoder of the input data from Weather4Cast competition.
    Contains 3 heads: IR, WV and VR.
    The input to the encoder should follow the structure: (N x T x H x W x C).
    The batch and time dimensions are collapsed into one.
    The output of the encoder follows the shape: (N x T x 3 x embd_size).
    """
    num_blocks: int
    filters: list
    embeddings: int

    @nn.compact
    def __call__(self, x, train: bool):
        # NOTE: Assume the input sequence follows the channel structure in the weather4cast repository.
        # infra-red
        ir_input = x[:, :, :, :, :7].reshape(-1, x.shape[2], x.shape[3], 7)
        # visible
        vr_input = x[:, :, :, :, 7:9].reshape(-1, x.shape[2], x.shape[3], 2)
        # water vapor
        wv_input = x[:, :, :, :, 9:].reshape(-1, x.shape[2], x.shape[3], 2)

        ir_embeddings = EncoderHead(num_blocks=self.num_blocks, filters=self.filters, embeddings=self.embeddings)(ir_input, train)
        vr_embeddings = EncoderHead(num_blocks=self.num_blocks, filters=self.filters, embeddings=self.embeddings)(vr_input, train)
        wv_embeddings = EncoderHead(num_blocks=self.num_blocks, filters=self.filters, embeddings=self.embeddings)(wv_input, train)

        # Concatenate embeddings.
        print(ir_embeddings.shape)
        embeddings = np.concatenate((ir_embeddings, vr_embeddings, wv_embeddings), axis=1)
        print(embeddings.shape)
        # Bring back time dimension.
        embeddings = embeddings.reshape(x.shape[0], x.shape[1], 3, self.embeddings)
        return embeddings

class Decoder(nn.Module):
    """
    Decoder of the TECO embeddings from Weather4Cast competition.
    The input to the decoder should follow the structure: (N x T x embd)
    The batch and time dimensions are collapsed into one.
    The output of the decoder follows the shape: ((N x T) x H x W x C).
    """
    num_blocks: int
    filters: int
    embeddings: int
    shape: tuple

    @nn.compact
    def __call__(self, x, train: bool):
        # Collapse the first 2 dims.
        x = x.reshape(-1, x.shape[2])

        ir_bands = DecoderHead(channels=7, num_blocks=self.num_blocks, filters=self.filters, embeddings=self.embeddings,
                shape=self.shape)(x, train)
        vr_bands = DecoderHead(channels=2, num_blocks=self.num_blocks, filters=self.filters, embeddings=self.embeddings,
                shape=self.shape)(x, train)
        wv_bands = DecoderHead(channels=2, num_blocks=self.num_blocks, filters=self.filters, embeddings=self.embeddings,
                shape=self.shape)(x, train)

        # Concatenate the channels.

        reconstruction = np.concatenate((ir_bands, vr_bands, wv_bands), axis=-1)
        return reconstruction




class EncoderHead(nn.Module):
    """
    Encoder head from specific bands.
    Composed of multiple EncoderHead blocks and a projection layer
    """
    num_blocks: int
    filters: list
    embeddings: int

    @nn.compact
    def __call__(self, x, train: bool):
        for i in range(self.num_blocks):
            x = HeadBlock(self.filters[i])(x, train)

        # flatten input
        x = x.reshape(x.shape[0], -1)
        # projection layer to the embeddings dimensions
        x = nn.Dense(self.embeddings)(x)
        x = nn.relu(x)

        return x

class DecoderHead(nn.Module):
    """
    Decoder head from taking embeddings from TECO.
    """
    channels: int
    num_blocks: int
    filters: list
    embeddings: int
    shape: tuple

    @nn.compact
    def __call__(self, x, train: bool):

        # Do not include batch and time dimensions.
        x = nn.Dense(np.prod(self.shape[1:]))(x)
        x = nn.relu(x)
        x = x.reshape(self.shape)
        for filters in reversed(self.filters):
            x = DecoderHeadBlock(filters)(x, train)
            x = jax.image.resize(x, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
                                 "bilinear")

        x = nn.Conv(self.channels, (3, 3))(x)
        x = nn.relu(x)
        return x


class DecoderHeadBlock(nn.Module):
    """
    One Encoder Head Block with the following structure:
    1. 1 conv layer
    2. ReLU
    3. BatchNorm
    x2

    4. 2x2 max pooling with (2, 2) stride
    """
    filters: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(self.filters, (3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.Conv(self.filters, (3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        return x


class HeadBlock(nn.Module):
    """
    One Encoder Head Block with the following structure:
    1. 1 conv layer
    2. ReLU
    3. BatchNorm
    x2

    4. 2x2 max pooling with (2, 2) stride
    """
    filters: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(self.filters, (3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.Conv(self.filters, (3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        x = nn.max_pool(x, (2, 2), (2, 2))
        return x


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
