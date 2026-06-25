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
os.makedirs("checkpoints", exist_ok=True)



"""## 1. Primitive Flow implementation

### 1.1.1 MLP Model
Using Equinox library and Jax a velocity field on a numcells x numcells grid is set up.
"""

class Velocity_field_mlp(eqx.Module):

  mlp: eqx.nn.MLP

  def __init__(self, key, dim):

    self.mlp = eqx.nn.MLP(
        in_size = dim + 1, # x + t
        out_size = dim, # v
        width_size = 128,
        depth = 3,
        key = key
    )

  def __call__(self, x, t):

    x_flat = x.reshape(-1)
    t = jnp.array([t])
    x_t = jnp.concatenate([x_flat, t])
    v = self.mlp(x_t)

    return v.reshape(x.shape)

"""### 1.1.2 CNN Model
Using the Equinox library and Jax a CNN velocity field on a 256x256 grid is set up. The activation function used in all hidden layers is the SiLu function: r"$\text{SiLu}(x) = x \cdot \sigma(x)$".
"""

class Velocity_field_cnn(eqx.Module):
    
    # set up 5 convolutional layers
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    conv5: eqx.nn.Conv2d

    def __init__(self, key):

        keys = jax.random.split(key, 5)

        self.conv1 = eqx.nn.Conv2d(
            2, 32, kernel_size=7, padding=3, dilation=1, key=keys[0]
        )
        self.conv2 = eqx.nn.Conv2d(
            32, 64, kernel_size=5, padding=4, dilation=2, key=keys[1]
        )
        self.conv3 = eqx.nn.Conv2d(
            64, 64, kernel_size=3, padding=4, dilation=4, key=keys[2]
        )
        self.conv4 = eqx.nn.Conv2d(
            64, 32, kernel_size=3, padding=2, dilation=2, key=keys[3]
        )
        self.conv5 = eqx.nn.Conv2d(
            32, 1, kernel_size=3, padding=1, dilation=1, key=keys[4]
        )


    def __call__(self, x, t):
        # shape of x: (C, H, W)

        t_emb = jnp.array([t])  # scalar feature
        t_emb = jnp.broadcast_to(t_emb, (1, x.shape[1], x.shape[2]))
        x = jnp.concatenate([x, t_emb], axis=0) # shape of x: (C, H, W)

        x = jax.nn.silu(self.conv1(x))
        x = jax.nn.silu(self.conv2(x))
        x = jax.nn.silu(self.conv3(x))
        x = jax.nn.silu(self.conv4(x))

        x = self.conv5(x)

        return x


"""### 1.2 Data generation
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

        t = jax.random.uniform(key3, (1,))

        x_t = (1 - t) * x0 + t * x1

        v_target = x1 - x0   # clean rectified flow direction
        
        return x_t, t, v_target
    
    x_t, t, v_target = jax.vmap(single_sample)(keys)

    return x_t, t, v_target

"""### 1.3 Loss function
For the loss a simple mean squared error is implemented.
"""

def loss_function(model, x_t, t, v_target):

    batched_model = jax.vmap(model, in_axes=(0, 0))

    v_pred = batched_model(x_t, t)

    return jnp.mean((v_pred - v_target) ** 2)

"""### 1.4 Training
Using the optax library we set up the training of the model.
"""

@eqx.filter_jit
def train_step(model, opt_state, x_t, t, v_target):

  loss, grads = eqx.filter_value_and_grad(loss_function)(model, x_t, t, v_target)

  updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))

  model = eqx.apply_updates(model, updates)

  return model, opt_state, loss

"""### 1.5 Sampling"""

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

        if step % 10 == 0:
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
small_data = data[:1000]


# ----------------- training setup ----------------------------
# hyperparameters
batch_size = 16
epochs = 1000

key = jax.random.key(0)

model = Velocity_field_cnn(key)

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
checkpoint_steps = {5000, 10000, 15000, 20000}

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
            f"checkpoints/velocity_field_{step}.eqx",
            model
            )
            


#------------------ load models -----------------------------

def load_model(step, template_model):
    return eqx.tree_deserialise_leaves(
        f"checkpoints/velocity_field_{step}.eqx",
        template_model
    )

# load a template model to fill
key = jax.random.key(0)
template_model = Velocity_field_cnn(key)
model_vis = load_model(20000, template_model)

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
plt.close()

plt.figure()
plt.imshow(fixed_noises[0][0], cmap='RdBu')
plt.colorbar()
plt.title('test generation')
plt.savefig('noise.png')
plt.close()

plt.figure()
plt.imshow(fixed_noises[1][0], cmap='RdBu')
plt.colorbar()
plt.title('test generation')
plt.savefig('noise.png')
plt.close()

x_gen1 = sample(model, fixed_noises[0], 100)
x_gen2 = sample(model, fixed_noises[1], 100)

plt.figure()
plt.imshow(x_gen1[4][0], cmap='RdBu')
plt.colorbar()
plt.savefig('x_gen_1_9.png')
plt.close()
plt.figure()
plt.imshow(x_gen1[-1][0], cmap='RdBu')
plt.colorbar()
plt.savefig('x_gen_1_10.png')
plt.close()


plt.figure()
plt.imshow(x_gen2[4][0], cmap='RdBu')
plt.colorbar()
plt.savefig('x_gen_2_9.png')
plt.close()
plt.figure()
plt.imshow(x_gen2[-1][0], cmap='RdBu')
plt.colorbar()
plt.savefig('x_gen_2_10.png')
plt.close()