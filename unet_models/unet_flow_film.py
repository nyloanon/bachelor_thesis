# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================


# ==========================================================================
#  import of libraries
# ==========================================================================

import jax
import jax.numpy as jnp
import equinox as eqx

# ==========================================================================
#  constants
# ==========================================================================

INPUT_CHANNELS = 4  #(density, velocity_x, velocity_y, pressure)
BASE_CHANNELS = 96
FOURIER_DIM = 32
TIME_CHANNELS = FOURIER_DIM * 2
SKIP_CHANNELS = 2*BASE_CHANNELS
GROUPS = 8
MAX_PERIOD = 1000

# ==========================================================================
#  Fourier time embedding
# ==========================================================================

"""
1. Time embedding
"""

"""
1.1 Fourier embedding: frequency values follow DDPM scheme 
"""

def fourier_embedding(t: float):
    
    # calculate frequencies 
    d : int = 2
    freqs: list[float] = [1000**(-k/d) for k in range(FOURIER_DIM)]
    freqs = jnp.array(freqs)

    # sin and cos values

    angles = freqs * t
    sin_vals = jnp.sin(angles)
    cos_vals = jnp.cos(angles)

    emb = jnp.concatenate([
        sin_vals,
        cos_vals
    ])

    return emb



"""
1.2 MLP for time embedding: 
    MLP learns which frequencies are important
"""

class TimeMLP(eqx.Module):

    mlp: list

    def __init__(self, in_dim, hidden_dim, out_dim, key):

        key1, key2 = jax.random.split(key)

        self.mlp = [
            eqx.nn.Linear(in_dim, hidden_dim, key=key1),
            eqx.nn.Linear(hidden_dim, out_dim, key=key2)
        ]

    def __call__(self, t):
        
        x = self.mlp[0](t)
        x = jax.nn.silu(x)
        x = self.mlp[1](x)

        return x


# ==========================================================================
#  U-Net class
# ==========================================================================

""" 
2. U-Net Implementation: 
    First the convolutional block, the down- and up-sampling are implemented as classes. Then the U-Net is implemented
"""

"""
2.1 ConvBlock, Down and Up
"""

class ConvBlock(eqx.Module):

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    film: eqx.nn.Linear
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(self, in_ch, out_ch, t_dim, key):
        key1, key2, key3 = jax.random.split(key, 3)

        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, key=key2)
        self.film = eqx.nn.Linear(t_dim, 2 * out_ch, key=key3)
        self.norm1 = eqx.nn.GroupNorm(GROUPS, out_ch)
        self.norm2 = eqx.nn.GroupNorm(GROUPS, out_ch)
    
    def __call__(self, x, t_emb):
        # first conv 
        h = self.conv1(x)

        # first group norm 
        h = self.norm1(h)

        # first film  conditioning
        film = self.film(t_emb)
        gamma, beta = jnp.split(film, 2, axis=-1)
        gamma = 0.001 *  jnp.tanh(gamma)
        beta = 0.001 * beta
        h = h + gamma[:, None, None] * h + beta[:, None, None] 
        
        # first silu
        h = jax.nn.silu(h)

        # second conv
        h = self.conv2(h)

        # second group norm
        h = self.norm2(h)

        # second silu
        h = jax.nn.silu(h)

        return h

class Down(eqx.Module):
    
    conv: eqx.nn.Conv2d

    def __init__(self, in_ch, out_ch, key):
        self.conv = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, key=key)
    
    def __call__(self, x):
        return jax.nn.silu(self.conv(x))

class Up(eqx.Module):
    
    conv : eqx.nn.Conv2d

    def __init__(self, in_ch, out_ch, key):
        self.conv = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, key=key)

    def __call__(self, x):
        x = jax.image.resize(
            x, 
            (x.shape[0], x.shape[1]*2, x.shape[2]*2), 
            method="bilinear"
            )
        return jax.nn.silu(self.conv(x))
    

class UNet(eqx.Module):
    
    down1: ConvBlock
    down2: ConvBlock
    down3: ConvBlock
    bottleneck: ConvBlock
    up1: ConvBlock
    up2: ConvBlock
    up3: ConvBlock

    downsample1: Down
    downsample2: Down
    downsample3: Down
    upsample1: Up
    upsample2: Up
    upsample3: Up
    
    t_mlp: eqx.Module

    final: eqx.nn.Conv2d

    def __init__(self, key):

        keys = jax.random.split(key, 16)

        # encoder
        self.down1 = ConvBlock(INPUT_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[0])
        self.downsample1 = Down(BASE_CHANNELS, BASE_CHANNELS, keys[1])

        self.down2 = ConvBlock(BASE_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[2])
        self.downsample2 = Down(BASE_CHANNELS, BASE_CHANNELS, keys[3])

        self.down3 = ConvBlock(BASE_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[14])
        self.downsample3 = Down(BASE_CHANNELS, BASE_CHANNELS, keys[15])

        # bottleneck
        self.bottleneck = ConvBlock(BASE_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[4])

        # decoder
        self.upsample1 = Up(BASE_CHANNELS, BASE_CHANNELS, keys[5])
        self.up1 = ConvBlock(SKIP_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[6])

        self.upsample2 = Up(BASE_CHANNELS, BASE_CHANNELS, keys[7])
        self.up2 = ConvBlock(SKIP_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[8])

        self.upsample3 = Up(BASE_CHANNELS, BASE_CHANNELS, keys[9])
        self.up3 = ConvBlock(SKIP_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[10])

        # time 
        self.t_mlp = TimeMLP(TIME_CHANNELS, TIME_CHANNELS*2, TIME_CHANNELS, key=keys[11])

        # output
        self.final = eqx.nn.Conv2d(BASE_CHANNELS, INPUT_CHANNELS, 1, key=keys[13])

    def __call__(self, x, t):
        #----------- time ----------------
        t_global = self.t_mlp(fourier_embedding(t)) # (64,)

        # ---------- down -----------------
        x1 = self.down1(x, t_global)
        x2 = self.downsample1(x1) # (64, 128, 128)
        
        x3 = self.down2(x2, t_global)
        x4 = self.downsample2(x3) # (64, 64, 64)

        x5 = self.down3(x4, t_global)
        x6 = self.downsample3(x5) #(64, 32, 32)

        # ---------- bottleneck ------------
        x7 = self.bottleneck(x6, t_global) # (64, 32, 32)
        # ---------- up --------------------
        x = self.upsample1(x7) #(64, 32, 32)

        # first skip connection
        x = jnp.concatenate([x, x5], axis=0) # (128, 32, 32)
        x = self.up1(x, t_global) # (64, 64, 64)
        x = self.upsample2(x)
        
        # second skip connection
        x = jnp.concatenate([x, x3], axis=0) # (128, 128, 128)
        x = self.up2(x, t_global) # (64, 128, 128)
        x = self.upsample3(x)
    
        # third skip connection
        x = jnp.concatenate([x, x1], axis=0) # (128, 256, 256)
        x = self.up3(x, t_global) # (64, 256, 256)
        
        # ----------- output ----------------
        return self.final(x)

# ==========================================================================
#  saving, loading and creation of models
# ==========================================================================

def create_model(key):
    model = UNet(key)
    return model

def save_model(model: UNet, filepath: str):
    eqx.tree_serialise_leaves(filepath, model)

def load_model(filepath: str, key):
    model = UNet(key)
    return eqx.tree_deserialise_leaves(filepath, model)