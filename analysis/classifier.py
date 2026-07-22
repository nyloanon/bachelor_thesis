# ==== GPU / XLA memory config (must precede any jax import) ====
import os
os.environ.setdefault("JAX_ENABLE_X64", "False")

# allocate GPU memory on demand rather than grabbing ~75% up front, and keep
# convolution autotuning on (disabling it makes each step ~50x slower).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Classifier for KHI destinguishing samples from simulations
#
#  Simple CNN implementation learns classification border between samples  
#  and real simulation data. 
# ==========================================================================

import argparse
import glob

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# ==========================================================================
# CNN classifier constants
# ==========================================================================
INPUT_CHANNELS = 4
WIDTHS = (32, 64, 128, 256)
EMB_DIM = (2, 32)
GROUPS = 8


# ==========================================================================
# CNN classifier classes
# ==========================================================================

class ConvBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm

    def __init__(self, in_ch, out_ch, key):
        key1, key2 = jax.random.split(key, 2)
        
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, key=key1)
        self.norm1 = eqx.nn.GroupNorm(GROUPS, out_ch)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, key=key2)
        self.norm2 = eqx.nn.GroupNorm(GROUPS, out_ch)

    def __call__(self, x):
        h = jax.nn.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))

        if x.shape == h.shape:
            h = h + x

        return jax.nn.silu(h)

class Downsample(eqx.Module):
    
    conv: eqx.nn.Conv2d

    def __init__(self, ch, key):
        self.conv = eqx.nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1, key=key)
    
    def __call__(self, x):
        return self.conv(x)


class OutBlock(eqx.Module):
    outblock: list

    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim, key):
        key1, key2, key3 = jax.random.split(key)

        self.outblock = [
            eqx.nn.Linear(in_dim, hidden_dim1, key=key1),
            eqx.nn.Linear(hidden_dim1, hidden_dim2, key=key2),
            eqx.nn.Linear(hidden_dim2, out_dim, key=key3)
        ]
    
    def __call__(self, x):
        x = jax.nn.silu(self.outblock[0](x))
        x = jax.nn.silu(self.outblock[1](x))
        
        return self.outblock[2](x)


class CondMLP(eqx.Module):
    mlp: list

    def __init__(self, in_dim, hidden_dim, out_dim, key):
        key1, key2 = jax.random.split(key)

        self.mlp = [
            eqx.nn.Linear(in_dim, hidden_dim, key=key1),
            eqx.nn.Linear(hidden_dim, out_dim, key=key2)
        ]

    def __call__(self, c):
        x = jax.nn.silu(self.mlp[0](c))
        
        return self.mlp[1](x)


# ==========================================================================
#  CNN classifier
# ==========================================================================

class CNN_Classifier(eqx.Module):
    in_conv: eqx.nn.Conv2d
    down_blocks: list
    downsamples: list
    cond_mlp: CondMLP
    out_block: OutBlock

    def __init__(self, key):
        keys = iter(jax.random.split((key), 11))
        
        widths = list(WIDTHS)
        self.in_conv = eqx.nn.Conv2d(INPUT_CHANNELS, widths[0], kernel_size=3, stride=1, padding=1, key=next(keys))
        self.cond_mlp = CondMLP(EMB_DIM[0], EMB_DIM[1], EMB_DIM[1], key=next(keys))

        # ---- down blocks -------
        self.down_blocks = []
        self.downsamples = []
        
        for i in range(len(widths)):
            in_ch = widths[i-1] if i > 0 else widths[0]
            self.down_blocks.append(ConvBlock(in_ch, widths[i], key=next(keys)))
            self.downsamples.append(Downsample(widths[i], key=next(keys)))
        
        # --- out block --------
        self.out_block = OutBlock(widths[-1] + EMB_DIM[-1], hidden_dim1=128, hidden_dim2=64, out_dim=1, key=next(keys))
    
    def __call__(self, x, t, M):
        # ---- conditioning ------
        cond = jnp.array([t, M])
        cond = self.cond_mlp(cond)

        # --- input layer ------
        h = self.in_conv(x)

        # --- down layer ------
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h)
            h = down(h)
        
        h = h.mean(axis=(1,2))
        
        # --- output layer ----
        h = jnp.concatenate(
            [
                h, 
                cond
            ]
        )

        return self.out_block(h).squeeze()


# ==========================================================================
#  CNN classifier helpers
# ==========================================================================

def create_model(key):
    return CNN_Classifier(key)

def save_model(model: CNN_Classifier, filepath: str):
    eqx.tree_serialise_leaves(filepath, model)

def load_model(filepath: str, key):
    model = CNN_Classifier(key)
    return eqx.tree_deserialise_leaves(filepath, model)