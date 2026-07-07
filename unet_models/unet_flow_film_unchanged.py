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

INPUT_CHANNELS = 1  #(density, velocity_x, velocity_y, pressure)
BASE_CHANNELS = 64
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
    
    down10: ConvBlock
    down11: ConvBlock
    down20: ConvBlock
    down21: ConvBlock
    down30: ConvBlock
    down31: ConvBlock
    bottleneck: ConvBlock
    up10: ConvBlock
    up11: ConvBlock
    up20: ConvBlock
    up21: ConvBlock
    up30: ConvBlock
    up31: ConvBlock

    downsample1: Down
    downsample2: Down
    downsample3: Down
    upsample1: Up
    upsample2: Up
    upsample3: Up
    
    t_mlp: eqx.Module

    final: eqx.nn.Conv2d

    def __init__(self, key):

        keys = jax.random.split(key, 34)

        # encoder
        self.down10 = ConvBlock(INPUT_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[0])
        self.down11 = ConvBlock(BASE_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[24])
        self.downsample1 = Down(BASE_CHANNELS, int(BASE_CHANNELS*1.5), keys[1])

        self.down20 = ConvBlock(int(BASE_CHANNELS*1.5), int(BASE_CHANNELS*1.5), TIME_CHANNELS, keys[2])
        self.down21 = ConvBlock(int(BASE_CHANNELS*1.5), int(BASE_CHANNELS*1.5), TIME_CHANNELS, keys[25])
        self.downsample2 = Down(int(BASE_CHANNELS*1.5), BASE_CHANNELS*3, keys[3])

        self.down30 = ConvBlock(BASE_CHANNELS*3, BASE_CHANNELS*3, TIME_CHANNELS, keys[14])
        self.down31 = ConvBlock(BASE_CHANNELS*3, BASE_CHANNELS*3, TIME_CHANNELS, keys[26])
        self.downsample3 = Down(BASE_CHANNELS*3, int(BASE_CHANNELS*4.5), keys[15])


        # bottleneck
        self.bottleneck = ConvBlock(int(BASE_CHANNELS*4.5), int(BASE_CHANNELS*4.5), TIME_CHANNELS, keys[4])

        # decoder
        self.upsample1 = Up(int(BASE_CHANNELS*4.5), int(BASE_CHANNELS*4.5), keys[5])
        self.up10 = ConvBlock(int(SKIP_CHANNELS*3.75), BASE_CHANNELS*3, TIME_CHANNELS, keys[6])
        self.up11 = ConvBlock(BASE_CHANNELS*3, BASE_CHANNELS*3, TIME_CHANNELS, keys[29])

        self.upsample2 = Up(BASE_CHANNELS*3, BASE_CHANNELS*3, keys[7])
        self.up20 = ConvBlock(int(SKIP_CHANNELS*2.25), int(BASE_CHANNELS*1.5), TIME_CHANNELS, keys[8])
        self.up21 = ConvBlock(int(BASE_CHANNELS*1.5), int(BASE_CHANNELS*1.5), TIME_CHANNELS, keys[30])

        self.upsample3 = Up(int(BASE_CHANNELS*1.5), int(BASE_CHANNELS*1.5), keys[9])
        self.up30 = ConvBlock(int(SKIP_CHANNELS*1.25), BASE_CHANNELS, TIME_CHANNELS, keys[10])
        self.up31 = ConvBlock(BASE_CHANNELS, BASE_CHANNELS, TIME_CHANNELS, keys[31])

        # time 
        self.t_mlp = TimeMLP(TIME_CHANNELS, TIME_CHANNELS*2, TIME_CHANNELS, key=keys[11])

        # output
        self.final = eqx.nn.Conv2d(BASE_CHANNELS, INPUT_CHANNELS, 1, key=keys[13])

    def __call__(self, x, t):
        #----------- time ----------------
        t_global = self.t_mlp(fourier_embedding(t)) # (64,)

        # ---------- down -----------------
        x1 = self.down10(x, t_global)
        x1_ = self.down11(x1, t_global) # (64, 256, 256)
        x2 = self.downsample1(x1_) 
        
        x3 = self.down20(x2, t_global)
        x3_ = self.down21(x3, t_global) # (96, 128, 128)
        x4 = self.downsample2(x3_) 

        x5 = self.down30(x4, t_global)
        x5_ = self.down31(x5, t_global) # (192, 64, 64)
        x6 = self.downsample3(x5_) 


        # ---------- bottleneck ------------
        x7 = self.bottleneck(x6, t_global) # (288, 32, 32)
        # ---------- up --------------------
        x = self.upsample1(x7) 

        # first skip connection
        x = jnp.concatenate([x, x5], axis=0) # (480, 64, 64)
        x = self.up10(x, t_global) # (192, 64, 64)
        x = self.up11(x, t_global)
        
        x = self.upsample2(x)
        
        # second skip connection
        x = jnp.concatenate([x, x3], axis=0) # (288, 128, 128)
        x = self.up20(x, t_global) # (96, 128, 128)
        x = self.up21(x, t_global)
        
        x = self.upsample3(x)
    
        # third skip connection
        x = jnp.concatenate([x, x1], axis=0) # (160, 256, 256)
        x = self.up30(x, t_global) # (64, 256, 256)
        x = self.up31(x, t_global)
        

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
