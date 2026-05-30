# imports

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax 
import glob



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

"""### 3.1.2 CNN Model
Using the Equinox library and Jax a CNN velocity field on a 64x64 grid is set up. The activation function used in all hidden layers is the SiLu function: $\text{SiLu}(x) = x \cdot \sigma(x)$.
"""

class Velocity_field_cnn(eqx.Module):
  conv1: eqx.nn.Conv2d
  conv2: eqx.nn.Conv2d
  conv3: eqx.nn.Conv2d
  conv4: eqx.nn.Conv2d
  conv5: eqx.nn.Conv2d

  def __init__(self, key):

    keys = jax.random.split(key, 5)   # number of keys matches number of layers

    self.conv1 = eqx.nn.Conv2d(
        1, 32, kernel_size=3, padding=1, key=keys[0]
    )
    self.conv2 = eqx.nn.Conv2d(
        32, 64, kernel_size=3, padding=1, key=keys[1]
    )
    self.conv3 = eqx.nn.Conv2d(
        64, 64, kernel_size=3, padding=1, key=keys[2]
    )
    self.conv4 = eqx.nn.Conv2d(
        64, 32, kernel_size=3, padding=1, key=keys[3]
    )
    self.conv5 = eqx.nn.Conv2d(
        32, 1, kernel_size=3, padding=1, key=keys[4]
    )

  def __call__(self, x, t):
    # shape of x (1, 64, 64)
    x = x + t

    h = jax.nn.silu(self.conv1(x))
    h = jax.nn.silu(self.conv2(h))
    h = jax.nn.silu(self.conv3(h))
    h = jax.nn.silu(self.conv4(h))

    v = self.conv5(h)

    return v

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
        ) 
        
        x0 = jax.random.normal(key2, x1.shape) # noise of same shape

        t = jax.random.uniform(key3, (1,))
        
        x_t = (1 - t) * x0 + t * x1

        v_target = x1 - x0
        
        return x_t, t, v_target
    
    x_t, t, v_target = jax.vmap(single_sample)(keys)

    return x_t, t, v_target

"""### 1.3 Loss function
For the loss a simple mean squared error is implemented.
"""

def loss_function(model, x_t, t, v_target):
  
  v_pred = model(x_t, t)

  return jnp.mean((v_target - v_prediction) ** 2)

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

def sample(model, x, steps, modelname):

    dt = 1.0 / steps

    if modelname == 1:
      x = x[None, :, :]   # add channel for CNN

    for i in range(steps):

      t = i / steps

      v = model(x, t)   # <-- v_theta(x_t)

      x = x + dt * v

    if modelname == 1:
      return x[0]

    return x


# ----------------- data import -------------------------------

files = sorted(glob.glob("data/final_state_*.npy"))
data = np.stack([np.load(f)[0] for f in files])     # only load the density data first
data = data[:, None, :, :]
print(data.shape)

# ----------------- training loop ----------------------------

batch_size = 4
num_steps = 10

key = jax.random.key(0)

model = Velocity_field_cnn(key)

# set learning_rate
optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


for step in range(num_steps):

    key, subkey = jax.random.split(key)

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

    if step % 100 == 0:
        print(f"step {step}, loss {loss}")

# test
print(x_t.shape)
print(t.shape)
print(v_target.shape)