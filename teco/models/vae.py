from typing import Any, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from ..utils import flatten, reshape_range


class VAE(nn.Module):
    config: Any
    training: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        assert not self.training, 'Does not support training'
        config = self.config

        self.encoder = Encoder(resolution=config.resolution,
                               ch=config.ch, ch_mult=config.ch_mult,
                               num_res_blocks=config.num_res_blocks,
                               attn_resolutions=config.attn_resolutions,
                               z_channels=config.z_channels,
                               double_z=config.double_z,
                               dropout=config.dropout)
        self.decoder = Decoder(resolution=config.resolution,
                               ch=config.ch, ch_mult=config.ch_mult,
                               out_ch=3, num_res_blocks=config.num_res_blocks,
                               attn_resolutions=config.attn_resolutions,
                               dropout=config.dropout)

        self.quant_conv = nn.Conv(2 * config.embed_dim, [1, 1])
        self.post_quant_conv = nn.Conv(config.z_channels, [1, 1])

    @property
    def embedding_dim(self):
        return self.config.embed_dim

    @property
    def latent_shape(self):
        size = self.config.resolution // 2 ** (len(self.config.ch_mult) - 1)
        return (size, size, self.config.embed_dim)

    def encode(self, video, rng=None, return_mode='sample'):
        is_image = len(video.shape) == 4
        if not is_image:
            B, T = video.shape[:2]
            video = flatten(video, 0, 2)
            
        encodings = []
        for i in range(0, video.shape[0], 64):
            v = video[i:i + 64]
            h = self.encoder(v)
            h = self.quant_conv(h)
            mean, logvar = jnp.split(h, 2, axis=-1)
            logvar = jnp.clip(logvar, -30, 20)
            std = jnp.exp(0.5 * logvar)
            
            if return_mode == 'sample':
                if rng is None:
                    rng = jax.random.PRNGKey(0)
                enc = mean + std * jax.random.normal(rng, shape=mean.shape)
            elif return_mode == 'stats':
                enc = jnp.concatenate([mean, std], axis=-1)
            elif return_mode == 'mean':
                enc = mean
            else:
                raise Exception('Unknown return mode:', return_mode)
            encodings.append(enc)
        encodings = jnp.concatenate(encodings)

        if not is_image:
            encodings = reshape_range(encodings, 0, 1, (B, T))
        return encodings
    
    def decode(self, z):
        is_image = len(z.shape) == 4
        h = z
        if is_image:
            recon = self.decoder(self.post_quant_conv(h))
        else:
            B, T = h.shape[:2]
            h = flatten(h, 0, 2)
            h = self.post_quant_conv(h)
            recon = self.decoder(h)
            recon = reshape_range(recon, 0, 1, (B, T))
        return recon
 
    def __call__(self, video):
        z = self.encode(video, return_mode='sample')
        recon = self.decode(z)
        return recon


class VectorQuantizer(nn.Module):
    n_e: int
    e_dim: int

    @nn.compact
    def __call__(self, z, encoding_indices=None):
        def quantize(encoding_indices):
            w = embeddings.value
            w = jax.device_put(w)
            return w[(encoding_indices,)]
        embeddings = self.variable('stats', 'embeddings',
                                   nn.initializers.zeros, None,
                                   [self.n_e, self.e_dim])
        
        if encoding_indices is not None:
            return quantize(encoding_indices)

        z_flattened = flatten(z, 0, -1)
        d = jnp.sum(z_flattened ** 2, axis=1, keepdims=True) + \
            jnp.sum(embeddings.value.T ** 2, axis=0, keepdims=True) - \
            2 * jnp.einsum('bd,nd->bn', z_flattened, embeddings.value)
        
        min_encoding_indices = jnp.argmin(d, axis=1)
        z_q = quantize(min_encoding_indices)
        z_q = jnp.reshape(z_q, z.shape)
        min_encoding_indices = jnp.reshape(min_encoding_indices, z.shape[:-1])

        return z_q, min_encoding_indices
        
        

class Encoder(nn.Module):
    resolution: int
    ch: int
    ch_mult: Tuple
    num_res_blocks: int
    attn_resolutions: Tuple
    z_channels: int
    double_z: bool = True
    dropout: float = 0.
    resample_with_conv: bool = True

    
    @nn.compact
    def __call__(self, x):
        num_resolutions = len(self.ch_mult)

        cur_res = self.resolution
        h = nn.Conv(self.ch, [3, 3], padding=[(1, 1), (1, 1)])(x)
        for i_level in range(num_resolutions):
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                h = ResnetBlock(dropout=self.dropout,
                                out_channels=block_out)(h)
                if cur_res in self.attn_resolutions:
                    h = AttnBlock()(h) 
            if i_level != num_resolutions - 1:
                h = Downsample(self.resample_with_conv)(h)
                cur_res //= 2
        
        h = ResnetBlock(dropout=self.dropout)(h)
        h = AttnBlock()(h)
        h = ResnetBlock(dropout=self.dropout)(h)
        
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(2 * self.z_channels if self.double_z else self.z_channels,
                    [3, 3], padding=[(1, 1), (1, 1)])(h)
        return h


class Decoder(nn.Module):
    resolution: int
    ch: int
    ch_mult: Tuple
    out_ch: int
    num_res_blocks: int
    attn_resolutions: Tuple
    dropout: float = 0.
    resamp_with_conv: bool = True
    
    @nn.compact
    def __call__(self, z):
        num_resolutions = len(self.ch_mult)

        block_in = self.ch * self.ch_mult[num_resolutions - 1]
        h = nn.Conv(block_in, [3, 3], padding=[(1, 1), (1, 1)])(z)

        h = ResnetBlock(dropout=self.dropout)(h)
        h = AttnBlock()(h)
        h = ResnetBlock(dropout=self.dropout)(h)
        
        cur_res = self.resolution // 2 ** (num_resolutions - 1)
        for i_level in reversed(range(num_resolutions)):
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                h = ResnetBlock(dropout=self.dropout,
                                out_channels=block_out)(h)
                if cur_res in self.attn_resolutions:
                    h = AttnBlock()(h)
            if i_level != 0:
                h = Upsample(self.resamp_with_conv)(h)
                cur_res *= 2
        
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, [3, 3], padding=[(1, 1), (1, 1)])(h)
        return h


class Upsample(nn.Module):
    with_conv: bool

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * 2, W * 2, C), 
                             jax.image.ResizeMethod.NEAREST, antialias=False)
        if self.with_conv:
            x = nn.Conv(C, [3, 3], padding=[(1, 1), (1, 1)])(x)
        return x

                
class Downsample(nn.Module):
    with_conv: bool

    @nn.compact
    def __call__(self, x):
        if self.with_conv:
            pad = ((0, 0), (0, 1), (0, 1), (0, 0))
            x = jnp.pad(x, pad, mode='constant')
            x = nn.Conv(x.shape[-1], [3, 3], strides=[2, 2], padding=[(0, 0), (0, 0)])(x)
        else:
            x = nn.avg_pool(x, [2, 2], strides=[2, 2], padding=[(0, 0), (0, 0)])
        return x


class ResnetBlock(nn.Module):
    dropout: float
    use_conv_shortcut: bool = False
    out_channels: Optional[int] = None
    
    @nn.compact
    def __call__(self, x):
        out_channels = self.out_channels or x.shape[-1]
        
        h = x
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(out_channels, [3, 3], padding=[(1, 1), (1, 1)])(h)

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Dropout(self.dropout)(h, deterministic=True)
        h = nn.Conv(out_channels, [3, 3], padding=[(1, 1), (1, 1)])(h)

        if x.shape[-1] != out_channels:
            if self.use_conv_shortcut:
                x = nn.Conv(out_channels, [3, 3], padding=[(1, 1), (1, 1)])(x)
            else:
                x = nn.Conv(out_channels, [1, 1])(x)
        return x + h


class AttnBlock(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]

        h_ = x
        h_ = nn.GroupNorm(num_groups=32)(h_)
        q = nn.Conv(channels, [1, 1])(h_)
        k = nn.Conv(channels, [1, 1])(h_)
        v = nn.Conv(channels, [1, 1])(h_)

        B, H, W, C = q.shape
        q = jnp.reshape(q, (B, H * W, C))
        k = jnp.reshape(k, (B, H * W, C))
        w_ = jnp.einsum('bqd,bkd->bqk', q, k)
        w_ = w_ * (int(C) ** (-0.5))
        w_ = jax.nn.softmax(w_, axis=2)

        v = jnp.reshape(v, (B, H * W, C))
        h_ = jnp.einsum('bqk,bkd->bqd', w_, v)
        h_ = jnp.reshape(h_, (B, H, W, C))

        h_ = nn.Conv(channels, [1, 1])(h_)
        
        return x + h_