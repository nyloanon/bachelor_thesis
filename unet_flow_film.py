# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================


# imports

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax 
import glob
import matplotlib.pyplot as plt

# saving models
import os
os.makedirs("unet_checkpoints", exist_ok=True)


"""
1. Time embedding
"""

"""
1.1 Fourier embedding: frequency values follow DDPM scheme 
"""

def fourier_embedding(t: float):
    
    # calculate frequencies 
    i : int = 32
    d : int = 2
    freqs: list[float] = [1000**(-k/d) for k in range(i)]
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
        self.norm1 = eqx.nn.GroupNorm(8, out_ch)
        self.norm2 = eqx.nn.GroupNorm(8, out_ch)
    
    def __call__(self, x, t_emb):
        # first conv 
        h = self.conv1(x)

        # first group norm 
        h = self.norm1(h)

        # first film  conditioning
        film = self.film(t_emb)
        gamma, beta = jnp.split(film, 2, axis=-1)
        gamma = 0.1 *  jnp.tanh(gamma)
        beta = 0.1 * beta
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
        self.down1 = ConvBlock(1, 64, 64, keys[0])
        self.downsample1 = Down(64, 64, keys[1])

        self.down2 = ConvBlock(64, 96, 64, keys[2])
        self.downsample2 = Down(96, 96, keys[3])

        self.down3 = ConvBlock(96, 128, 64, keys[14])
        self.downsample3 = Down(128, 128, keys[15])

        # bottleneck
        self.bottleneck = ConvBlock(128, 128, 64, keys[4])

        # decoder
        self.upsample1 = Up(128, 128, keys[5])
        self.up1 = ConvBlock(256, 128, 64, keys[6])

        self.upsample2 = Up(128, 128, keys[7])
        self.up2 = ConvBlock(224, 96, 64, keys[8])

        self.upsample3 = Up(96, 96, keys[9])
        self.up3 = ConvBlock(160, 64, 64, keys[10])

        # time 
        self.t_mlp = TimeMLP(64, 128, 64, key=keys[11])

        # output
        self.final = eqx.nn.Conv2d(64, 1, 1, key=keys[13])

    def __call__(self, x, t):
        #----------- time ----------------
        t_global = self.t_mlp(fourier_embedding(t)) # (64,)

        # ---------- down -----------------
        x1 = self.down1(x, t_global)
        x2 = self.downsample1(x1) # (64, 128, 128)

        x3 = self.down2(x2, t_global)
        x4 = self.downsample2(x3) # (96, 64, 64)

        x5 = self.down3(x4, t_global)
        x6 = self.downsample3(x5) #(128, 32, 32)
        # ---------- bottleneck ------------
        x7 = self.bottleneck(x6, t_global) # (128, 32, 32)
        # ---------- up --------------------
        x = self.upsample1(x7) #(128, 32, 32)
        # first skip connection
        x = jnp.concatenate([x, x5], axis=0) # (256, 32, 32)
        x = self.up1(x, t_global) # (128, 64, 64)

        x = self.upsample2(x)
        # second skip connection
        x = jnp.concatenate([x, x3], axis=0) # (224, 64, 64)
        x = self.up2(x, t_global) # (96, 128, 128)

        x = self.upsample3(x)
        # third skip connection
        x = jnp.concatenate([x, x1], axis=0) # (160, 128, 128)
        x = self.up3(x, t_global) # (64, 256, 256)
        # ----------- output ----------------
        return self.final(x)

        
"""### 2. Data generation
Data batches are divided into single samples at the current time point t. The image x_t, time t itself and the target velocity v_target are returned.
"""


def sample_batch(key, data, batch_size):

    keys = jax.random.split(key, batch_size)

    def single_sample(k):
        
        key1, key2, key3 = jax.random.split(k, 3)

        idx = jax.random.randint(key1, (), 0, len(data))
        
        x1 = jax.lax.dynamic_index_in_dim(
            data,
            idx,
            axis=0,
            keepdims=False
        )           # real KHI

        x0 = jax.random.normal(key2, x1.shape)  # noise

        t = jax.random.uniform(key3, ())

        x_t = (1 - t) * x0 + t * x1

        v_target = x1 - x0   # clean rectified flow direction
        
        return x_t, t, v_target
    
    x_t, t, v_target = jax.vmap(single_sample)(keys)

    return x_t, t, v_target

"""### 3. Loss function
For the loss a simple mean squared error is implemented.
"""

def loss_function(model, x_t, t, v_target):

    batched_model = jax.vmap(model, in_axes=(0, 0))

    v_pred = batched_model(x_t, t)

    return jnp.mean((v_pred - v_target) ** 2)

"""### 4. Training
Using the optax library we set up the training of the model.
"""

@eqx.filter_jit
def train_step(model, opt_state, x_t, t, v_target):

  loss, grads = eqx.filter_value_and_grad(loss_function)(model, x_t, t, v_target)

  updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))

  model = eqx.apply_updates(model, updates)

  return model, opt_state, loss

"""### 5. Sampling"""

def sample(model, x, steps):

    dt = 1.0 / steps

    x_result = []

    for i in range(steps):

        t = i / steps

        # velocity at current state
        v1 = model(x, t)

        # Euler prediction
        x_euler = x + dt * v1

        # next time (clamped to training range)
        t_next = min(t + dt, 1.0)

        # velocity at predicted state
        v2 = model(x_euler, t_next)

        # Heun update
        x = x + dt * 0.5 * (v1 + v2)

        if i % 10 == 0:
            x_result.append(x)

    return x_result

# ----------------- data import -------------------------------

files = sorted(glob.glob("data/final_state_*.npy"))
# only load the density data first
data = np.stack([np.load(f)[0] for f in files]) 
# normalize data
mean = data.mean()
std = data.std()
data = (data - mean) / std 
# put data in correct shape 
data = data[:, None, :, :]
# use small data first
small_data = data


# ----------------- training setup ----------------------------
# hyperparameters
batch_size = 16
epochs = 10000

key = jax.random.key(0)

model = UNet(key)

# set learning rate
learning_rate = 3e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# ----------------- fixed validation set ----------------------

val_key = jax.random.key(12345)

x_val, t_val, v_val = sample_batch(
    val_key,
    small_data,
    batch_size=16
)

# ----------------- fixed sample noises -----------------------

noise_key = jax.random.key(999)
fixed_noises = jax.random.split(noise_key, 4)

fixed_noises = [
    jax.random.normal(k, (1, 256, 256))
    for k in fixed_noises
]

# ----------------- training loop -----------------------------

loss_history = []
val_mse_history = []
checkpoint_steps = {10000, 20000, 30000, 40000, 50000}

for step in range(epochs + 1):

    key, subkey = jax.random.split(key)

    x_t, t, v_target = sample_batch(
        subkey,
        small_data,
        batch_size
    )

    model, opt_state, loss = train_step(
        model,
        opt_state,
        x_t,
        t,
        v_target
    )

    # evaluate on fixed validation batch
    if step % 1000 == 0:

        v_pred_val = jax.vmap(
            lambda x, t_: model(x, t_)
        )(x_val, t_val)

        val_mse = jnp.mean(
            (v_pred_val - v_val) ** 2
        )

        val_mse_history.append(val_mse)
        
        loss_history.append(loss)

        print(
            f"step {step}, "
            f"train loss {loss:.4f}, "
            f"val mse {val_mse:.4f}"
        )

        if step in checkpoint_steps:
            eqx.tree_serialise_leaves(
            f"unet_checkpoints/unet_velocity_field_{step}.eqx",
            model
            )
            

# ------------- test ---------------------------
#------------------ load models -----------------------------

def load_model(step, template_model):
    return eqx.tree_deserialise_leaves(
        f"unet_checkpoints/unet_velocity_field_{step}.eqx",
        template_model
    )

# ----------------- plot training history --------------------

plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Training loss")
plt.savefig("loss_history.png")

plt.figure()
plt.plot(
    np.arange(len(val_mse_history)) * 100,
    val_mse_history
)
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("validation MSE")
plt.title("Validation velocity MSE")
plt.savefig("val_mse_history.png")

# ----------------- validation visualization -----------------

x_vis, t_vis, v_target_vis = sample_batch(
    jax.random.key(777),
    small_data,
    batch_size=1
)

v_pred_vis = model(
    x_vis[0],
    t_vis[0]
)

plt.figure()
plt.imshow(v_target_vis[0, 0], cmap="RdBu")
plt.colorbar()
plt.title("Target velocity")
plt.savefig("v_target.png")

plt.figure()
plt.imshow(v_pred_vis[0], cmap="RdBu")
plt.colorbar()
plt.title("Predicted velocity")
plt.savefig("v_pred.png")

err = v_pred_vis - v_target_vis[0]

plt.figure()
plt.imshow(err[0], cmap="RdBu")
plt.colorbar()
plt.title("Velocity error")
plt.savefig("err_vel.png")


# ----------------- fixed sample generation for the different time steps------------------

x_gen_1 = sample(model, fixed_noises[0], 100)
plt.figure()
plt.imshow(x_gen_1[-1][0], cmap="RdBu")
plt.colorbar()
plt.title("test generationv1, step 100")
plt.savefig("test_generationv1_100.png")
plt.close()

plt.figure()
plt.imshow(x_gen_1[5][0], cmap="RdBu")
plt.colorbar()
plt.title("test generationv1, step 60")
plt.savefig("test_generationv1_60.png")
plt.close()

plt.figure()
plt.imshow(fixed_noises[0][0], cmap="RdBu")
plt.colorbar()
plt.title("test noise")
plt.savefig("test_noisev1.png")
plt.close()

x_gen_2 = sample(model, fixed_noises[1], 100)
plt.figure()
plt.imshow(x_gen_2[-1][0], cmap="RdBu")
plt.colorbar()
plt.title("test generationv2, step 100")
plt.savefig("test_generationv2_100.png")
plt.close()

plt.figure()
plt.imshow(x_gen_2[5][0], cmap="RdBu")
plt.colorbar()
plt.title("test generationv2, step 60")
plt.savefig("test_generationv2_60.png")
plt.close()

plt.figure()
plt.imshow(fixed_noises[1][0], cmap="RdBu")
plt.colorbar()
plt.title("test noise")
plt.savefig("test_noisev2.png")
plt.close()