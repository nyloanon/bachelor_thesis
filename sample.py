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
#  Model import
# ==========================================================================

checkpoint_step = 20000
checkpoint_path = f'unet_checkpoints/unet_velocity_field_{checkpoint_step}.eqx'
key = jax.random.key(0)
model = unet_flow_film.load_model(checkpoint_path, key)

# ==========================================================================
#  data generation/sampling
# ==========================================================================

"""### 1. Sampling
Using the Euler-Heun update (Runge-Kutta second order) we sample our generated image. 
"""

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

    return np.asarray(x_result)


# ==========================================================================
#  data import
# ==========================================================================

files = sorted(glob.glob("data/final_state_*.npy"))

# only load the density data first
data = np.stack([np.load(f)[1] for f in files]) 

# normalize data
mean = data.mean()
std = data.std()
data = (data - mean) / std 

# put data in correct shape 
data = data[:, None, :, :]
small_data = data[:100]


# ==========================================================================
#  display of generation and confirmation images
# ==========================================================================

# fixed noise sample
noise_key = jax.random.key(999)
fixed_noises = jax.random.split(noise_key, 4)

fixed_noises = [
    jax.random.normal(k, (4, 256, 256))
    for k in fixed_noises
]

# sample generation 

x_gen_1 = sample(model, fixed_noises[0], 100)
rho_gen1 = x_gen_1[-1][0]
v_x_gen1 = x_gen_1[-1][1]
v_y_gen1 = x_gen_1[-1][2]
p_gen1 = x_gen_1[-1][3]

# density plot
plt.figure()
plt.imshow(rho_gen1, cmap="RdBu")
plt.colorbar()
plt.title("density test generationv1, step 100")
plt.savefig("rho_test_generationv1_100.png")
plt.close()

plt.figure()
plt.imshow(fixed_noises[0][0], cmap="RdBu")
plt.colorbar()
plt.title("density test noise")
plt.savefig("rho_test_noisev1.png")
plt.close()

# velocity x plot
plt.figure()
plt.imshow(v_x_gen1, cmap="RdBu")
plt.colorbar()
plt.title("velocity_x test generationv1, step 100")
plt.savefig("vx_test_generationv1_100.png")
plt.close()

plt.figure()
plt.imshow(fixed_noises[0][1], cmap="RdBu")
plt.colorbar()
plt.title("velcotiy_x test noise")
plt.savefig("vx_test_noisev1.png")
plt.close()

# visualize real data example of velocity x
v_x_real = small_data[0]
plt.figure()
plt.imshow(v_x_real[0], cmap="RdBu")
plt.colorbar()
plt.title("velcotiy_x test real")
plt.savefig("vx_test_realv1.png")
plt.close()

