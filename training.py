# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================


# ==========================================================================
#  import of libraries
# ==========================================================================

# model
from unet_models import unet_flow_film
import equinox as eqx
import optax 

# math
import numpy as np
import jax
import jax.numpy as jnp

# data
import glob

# visualization
import matplotlib.pyplot as plt

# saving models
import os
os.makedirs("unet_checkpoints", exist_ok=True)


# ==========================================================================
#  functions for training
# ==========================================================================

"""### Generation of Data pairs
1. Data batches are divided into single samples at the current time point t. The image x_t, time t itself and the target velocity v_target are returned.
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

"""### 2. Loss function
For the loss a simple mean squared error is implemented.
"""

def loss_function(model, x_t, t, v_target):

    batched_model = jax.vmap(model, in_axes=(0, 0))

    v_pred = batched_model(x_t, t)

    return jnp.mean((v_pred - v_target) ** 2)

"""### 3. Training step
Using the optax library we set up the training of the model.
"""

@eqx.filter_jit
def train_step(model, opt_state, x_t, t, v_target):

  loss, grads = eqx.filter_value_and_grad(loss_function)(model, x_t, t, v_target)

  updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))

  model = eqx.apply_updates(model, updates)

  return model, opt_state, loss



# ==========================================================================
#  data import
# ==========================================================================

files = sorted(glob.glob("data/final_state_*.npy"))

# only load the density data first
rho = np.stack([np.load(f)[0] for f in files]) 
v_x = np.stack([np.load(f)[1] for f in files]) 
v_y = np.stack([np.load(f)[2] for f in files]) 
p = np.stack([np.load(f)[3] for f in files]) 
print(rho.shape)

# normalize data
rho = (rho - rho.mean()) / rho.std()
v_x = (v_x - v_x.mean()) / v_x.std()
v_y = (v_y - v_y.mean()) / v_y.std()
p = (p - p.mean()) / p.std()

# put data in correct shape 
data = jnp.stack([rho, v_x, v_y, p], axis=1)
print(data.shape)


# ==========================================================================
#  training setup
# ==========================================================================

# hyperparameters
batch_size = 16
epochs = 50000

# set up model
key_model = jax.random.key(10)
model = unet_flow_film.create_model(key_model) 

# set learning rate
learning_rate = 3e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# fixed validation set
val_key = jax.random.key(12345)

x_val, t_val, v_val = sample_batch(
    val_key,
    data,
    batch_size=16
)

# ==========================================================================
#  training loop
# ==========================================================================

key_training = jax.random.key(20)
loss_history = []
val_mse_history = []
checkpoint_steps = {10000, 20000, 30000, 40000, 50000}

for step in range(epochs + 1):

    key, subkey = jax.random.split(key_training)

    x_t, t, v_target = sample_batch(
        subkey,
        data,
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
            filepath = f"unet_checkpoints/unet_velocity_field_{step}.eqx"
            unet_flow_film.save_model(model, filepath)
            

# ==========================================================================
#  plot training history
# ==========================================================================

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