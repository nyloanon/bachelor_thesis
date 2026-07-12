# ==== GPU / XLA memory config (must precede any jax import) ====
import os as _os
# allocation of GPU memory on demand rather than grabbing ca. 75% up front
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Rectified-flow U-Net for KHI field generation
#
#  Predicts the rectified-flow velocity v(x_t, t) for the full 4-channel
#  field (density, velocity_x, velocity_y, pressure).
#
#  Design notes
#  ------------
#  * DDPM-style residual blocks (GroupNorm + SiLU + 3x3 conv) with FiLM
#    time conditioning (per-channel scale & shift). FiLM is applied at full
#    strength -- time information is essential for a flow model.
#  * Channel widths grow with depth; the architecture is parameterised by a
#    list of widths so there is no fragile hand-tuned channel arithmetic.
# ==========================================================================

# ==========================================================================
#  import of libraries
# ==========================================================================

import jax
import jax.numpy as jnp
import equinox as eqx

# ==========================================================================
#  constants
# ==========================================================================

INPUT_CHANNELS = 8  #(x_rf and current concatenated --> (B, 8, 256, 256))
OUTPUT_CHANNELS = 4
BASE_CHANNELS = 32
WIDTHS = (BASE_CHANNELS, BASE_CHANNELS*2, BASE_CHANNELS*4, BASE_CHANNELS*6)
BOTTLENECK = BASE_CHANNELS*8
FOURIER_DIM = 32
EMB_CHANNELS = FOURIER_DIM * 2
EMB_DIM = 256
N_COND = 3
COND_DIM = N_COND * EMB_DIM
GROUPS = 8
MAX_PERIOD = 1000.0

# Gradient checkpointing (rematerialization): recompute ResBlock activations in
# the backward pass instead of storing them. Cuts peak activation memory ~2-4x
# for ~30% extra compute, so training fits even on a partly-occupied GPU.
# Disable with the environment variable KHI_CHECKPOINT=0.
USE_CHECKPOINT = _os.environ.get("KHI_CHECKPOINT", "1") != "0"

# ==========================================================================
#  Fourier time embedding
# ==========================================================================

"""
1. Rectified Flow time, physical dt and mach number embedding
"""

"""
1.1 Fourier embedding: frequency values follow DDPM scheme 
"""

def fourier_embedding(t: float):
    # calculate frequencies and angles
    d : int = 2
    freqs = jnp.array([MAX_PERIOD ** (-k / d) for k in range(FOURIER_DIM)])
    angles = freqs * t

    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])
    

"""
1.2 MLP for Rectified FLow time, physical dt and mach number embedding: 
    MLP learns which frequencies are important
"""

class EmbMLP(eqx.Module):

    mlp: list

    def __init__(self, in_dim, hidden_dim, out_dim, key):

        key1, key2 = jax.random.split(key)

        self.mlp = [
            eqx.nn.Linear(in_dim, hidden_dim, key=key1),
            eqx.nn.Linear(hidden_dim, out_dim, key=key2)
        ]

    def __call__(self, t):
        x = jax.nn.silu(self.mlp[0](t))
        
        return self.mlp[1](x)


# ==========================================================================
#  Classes for U-Net 
# ==========================================================================

""" 
2. U-Net Implementation: 
    First the Residual block, the down- and up-sampling are implemented as classes.
"""

"""
2.1 ResBlock, Down and Up
"""

class ResBlock(eqx.Module):

    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    film: eqx.nn.Linear
    skip: eqx.Module # Conv2d (1x1) or Identity depending on in_dim and out_dim  

    def __init__(self, in_ch, out_ch, cond_dim, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        self.norm1 = eqx.nn.GroupNorm(GROUPS, in_ch)
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, key=key1)
        self.norm2 = eqx.nn.GroupNorm(GROUPS, out_ch)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, key=key2)
        # FiLM shifts and scales in pairs per-channel
        self.film = eqx.nn.Linear(cond_dim, 2 * out_ch, key=key3)
        if in_ch == out_ch:
            self.skip = eqx.nn.Identity()
        else:
            self.skip = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=1, key=key4)

    def __call__(self, x, cond):
        h = self.conv1(jax.nn.silu(self.norm1(x)))

        # FiLM 
        scale, shift = jnp.split(self.film(cond), 2, axis=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, None, None]) + shift[:, None, None]
        h = self.conv2(jax.nn.silu(h))

        return h + self.skip(x)

class Downsample(eqx.Module):
    
    conv: eqx.nn.Conv2d

    def __init__(self, ch, key):
        self.conv = eqx.nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1, key=key)
    
    def __call__(self, x):
        return self.conv(x)

class Upsample(eqx.Module):

    conv : eqx.nn.Conv2d

    def __init__(self, ch, key):
        self.conv = eqx.nn.Conv2d(ch, ch, kernel_size=3, padding=1, key=key)

    def __call__(self, x):
        x = jax.image.resize(
            x, 
            (x.shape[0], x.shape[1] * 2, x.shape[2] * 2), 
            method="bilinear"
            )
        return self.conv(x)

# ==========================================================================
#  checkpointed block runner
# ==========================================================================

def _run_block(block, x, cond):
    return block(x, cond)

# Rematerialized variant: forward activations inside the block are recomputed
# during the backward pass rather than stored. filter_checkpoint treats the
# block's weight arrays as differentiable inputs, so gradients still flow.
_run_block_ckpt = eqx.filter_checkpoint(_run_block)


def run_block(block, x, cond):
    if USE_CHECKPOINT:
        return _run_block_ckpt(block, x, cond)
    return _run_block(block, x, cond)


# ==========================================================================
#  U-Net architecture 
# ==========================================================================    

class UNet(eqx.Module):
    
    in_conv: eqx.nn.Conv2d
    down_blocks: list
    downsamples: list
    mid1: ResBlock
    mid2: ResBlock
    up_blocks: list
    upsamples: list
    t_rf_mlp: EmbMLP
    dt_mlp: EmbMLP
    mach_mlp: EmbMLP
    out_norm: eqx.nn.GroupNorm
    out_conv: eqx.nn.Conv2d

    def __init__(self, key):

        keys = iter(jax.random.split(key, 66))

        widths = list(WIDTHS)
        self.in_conv = eqx.nn.Conv2d(INPUT_CHANNELS, widths[0], kernel_size=3, padding=1, key=next(keys))
        self.t_rf_mlp = EmbMLP(EMB_CHANNELS, EMB_DIM, EMB_DIM, key=next(keys))
        self.dt_mlp = EmbMLP(EMB_CHANNELS, EMB_DIM, EMB_DIM, key=next(keys))
        self.mach_mlp = EmbMLP(EMB_CHANNELS, EMB_DIM, EMB_DIM, key=next(keys))
        
        #---- encoder -----
        self.down_blocks = []
        self.downsamples = []
        for i in range(len(widths)):
            in_ch = widths[i - 1] if i > 0 else widths[0]
            self.down_blocks.append(ResBlock(in_ch, widths[i], COND_DIM, next(keys)))
            self.downsamples.append(Downsample(widths[i], next(keys)))
        
        #----- bottleneck -------
        self.mid1 = ResBlock(widths[-1], BOTTLENECK, COND_DIM, next(keys))
        self.mid2 = ResBlock(BOTTLENECK, BOTTLENECK, COND_DIM, next(keys))

        #------ decoder -------
        self.up_blocks = []
        self.upsamples = []
        prev = BOTTLENECK
        for i in reversed(range(len(widths))):
            self.upsamples.append(Upsample(prev, next(keys)))
            # skip connections with width[i] layer of encoder
            self.up_blocks.append(ResBlock(prev + widths[i], widths[i], COND_DIM, next(keys)))
            prev = widths[i]

        #------ output refinement ---------
        self.out_norm = eqx.nn.GroupNorm(GROUPS, widths[0])
        self.out_conv = eqx.nn.Conv2d(widths[0], OUTPUT_CHANNELS, kernel_size=1, key=next(keys))

    def __call__(self, x_t_rf, current, t_rf, dt, mach):
        #----------- time ----------------
        x = jnp.concatenate(
            [
                x_t_rf,
                current
            ],
            axis=0
        )
        
        t_rf_emb = self.t_rf_mlp(fourier_embedding(t_rf))
        dt_emb = self.dt_mlp(fourier_embedding(dt))
        mach_emb = self.mach_mlp(fourier_embedding(mach))
        
        cond = jnp.concatenate(
            [
                t_rf_emb,
                dt_emb,
                mach_emb,
            ],
            axis=-1,
        )

        #----------- input layer ---------
        h = self.in_conv(x)

        # ---------- down -----------------
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = run_block(block, h, cond)
            skips.append(h)
            h = down(h)

        # ---------- bottleneck ------------
        h = run_block(self.mid1, h, cond)
        h = run_block(self.mid2, h, cond)

        # ---------- up --------------------
        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            h = jnp.concatenate([h, skip], axis=0)
            h = run_block(block, h, cond)

        # ----------- output ----------------
        h = jax.nn.silu(self.out_norm(h))
        return self.out_conv(h)


# ==========================================================================
#  saving, loading and creation of models
# ==========================================================================

def create_model(key):
    return UNet(key)

def save_model(model: UNet, filepath: str):
    eqx.tree_serialise_leaves(filepath, model)

def load_model(filepath: str, key):
    model = UNet(key)
    return eqx.tree_deserialise_leaves(filepath, model)
