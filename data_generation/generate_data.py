# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import os as _os
# numerics
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, uniform

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from jaxtyping import Array, Float, Int

# astronomix
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.option_classes.simulation_config import SnapshotSettings
from astronomix import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    DOUBLE_MINMOD,
    FORWARDS,
    HLL,
    HLLC,
    HYBRID_HLLC,
    MINMOD,
    OSHER,
    PERIODIC_BOUNDARY,
    OPEN_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    FINITE_VOLUME,
    FINITE_DIFFERENCE
)

# model
import equinox as eqx

# training
import optax

from astronomix.variable_registry.registered_variables import StaticIntVector
print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = 256

fixed_timestep = False
scale_time = False
dt_max = 0.1
num_timesteps = 2000

# mach number setup
n_mach = 0.5
rho_back = 1.0
rho_slab = 2.0
p_0 = 2.5

# critical mach number
delta = rho_slab / rho_back 
n_mach_crit = (1.0 + delta ** (-1 / 3)) ** (2 / 3)
print(f"Critical mach number for current setup rho_back = {rho_back}, rho_slab = {rho_slab}, n_mach_crit = {n_mach_crit}. \n")
# calc of back ground propagation speed 
c_s_back = jnp.sqrt(gamma * p_0 / rho_back)
v_total = c_s_back * n_mach
v_a = v_total / 2

# setup simulation config
config = SimulationConfig(
    progress_bar = False,
    dimensionality = 2,
    box_size = box_size,
    num_cells = StaticIntVector(x=num_cells, y=num_cells),
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
    num_timesteps = num_timesteps,
    boundary_settings = BoundarySettings(
        x = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
    ),
    limiter = DOUBLE_MINMOD,
    return_snapshots = False,
    riemann_solver = HYBRID_HLLC,
    solver_mode = FINITE_VOLUME
)

helper_data = get_helper_data(config)

params = SimulationParams(
    t_end = 2.0,
    C_cfl = 0.4
)

registered_variables = get_registered_variables(config)

def produce_plot(final_state, index):
  s = 0.1

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

  # equal aspect ratio
  ax1.set_aspect('equal', 'box')
  ax2.set_aspect('equal', 'box')
  ax3.set_aspect('equal', 'box')

  x = jnp.linspace(0, box_size, num_cells)
  y = jnp.linspace(0, box_size, num_cells)

  ym, xm = jnp.meshgrid(x, y)

  # on the first axis plot the density
  # log scaler
  norm_rho = LogNorm(vmin = jnp.min(final_state[0, :, :]), vmax = jnp.max(final_state[0, :, :]), clip = True)
  norm_p = LogNorm(vmin = jnp.min(final_state[3, :, :]), vmax = jnp.max(final_state[3, :, :]), clip = True)

  # ax1.scatter(xm.flatten(), ym.flatten(), c = final_state[0, :, :].flatten(), s = s, norm = norm_rho, marker = "s", cmap = "jet")
  # ax1.set_title("Density")

  ax1.imshow(final_state[0, :, :].T, norm = norm_rho, cmap = "jet", origin = "lower", extent = [0, box_size, 0, box_size])
  ax1.set_title("Density")

  # on the second axis plot the absolute velocity
  # abs_vel = jnp.sqrt(final_state[1, :, :]**2 + final_state[2, :, :]**2)

  # vel_norm = LogNorm(vmin = jnp.min(abs_vel), vmax = jnp.max(abs_vel), clip = True)

  ax2.imshow(final_state[1, :, :].T, cmap = "jet", origin = "lower", extent = [0, box_size, 0, box_size])
  ax2.set_title("Velocity")

  # on the third axis plot the pressure
  ax3.imshow(final_state[4, :, :].T, norm = norm_p, cmap = "jet", origin = "lower", extent = [0, box_size, 0, box_size])
  ax3.set_title("Pressure")

  plt.savefig(f"final_state_{index}.png")

"""## 2. KHI Init

"""

def random_khi_fourier_modes(
    key,
    X,
    Y,
    amplitude=0.01,
    k_min=1,
    k_max=8,
    shear_layers=(0.25, 0.75),
    width=0.03,
    spectral_slope=1.0,
):
    """
    Random Fourier-mode perturbation for Kelvin-Helmholtz initialization.

    Produces a y-velocity perturbation of the form

        u_y(x, y) = A f(y) sum_k a_k sin(2 pi k x + phi_k)

    where f(y) localizes the perturbation around the shear layers.
    """

    modes = jnp.arange(k_min, k_max + 1)
    num_modes = modes.shape[0]

    key_amp, key_phase = jax.random.split(key)

    # random amplitudes with optional spectral decay
    coeffs = jax.random.normal(key_amp, (num_modes,))
    coeffs = coeffs / modes**spectral_slope

    # normalize so amplitude is controlled by `amplitude`
    coeffs = coeffs / jnp.sqrt(jnp.sum(coeffs**2) + 1e-30)

    phases = jax.random.uniform(
        key_phase,
        (num_modes,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    # shape: (num_modes, nx, ny)
    fourier_sum = jnp.sum(
        coeffs[:, None, None]
        * jnp.sin(2.0 * jnp.pi * modes[:, None, None] * X[None, :, :] + phases[:, None, None]),
        axis=0,
    )

    # localize perturbation around both shear layers
    envelope = jnp.zeros_like(Y)
    for y0 in shear_layers:
        envelope = envelope + jnp.exp(-0.5 * ((Y - y0) / width) ** 2)

    return amplitude * envelope * fourier_sum

"""### 3 Data Generation
In this step we generate the 256x256 KHI data which we train our model with. We went for _ images for a start.
"""

def slab_profile(f_b, f_s, Y, y_center, slab_radius, smoothing_length):
    """Tanh transition from f_b (background) to f_s (stream)."""

    # f(y) = f_b + 0.25 * (f_s - f_b) * (1 + tanh((R_s - (y - y_c)) / σ)) * (1 + tanh((R_s + (y - y_c)) / σ))
    return f_b + 0.25 * (f_s - f_b) * (
        (1 + jnp.tanh((slab_radius - (Y - y_center)) / smoothing_length)) *
        (1 + jnp.tanh((slab_radius + (Y - y_center)) / smoothing_length))
    )

def calc_smoothing_cells(n_mach, min_smoothing_cells = 2, max_smoothing_cells = 12, min_n_mach = 0.5, max_n_mach = 1.7):
    """Calculation of smoothing cells, required to prevent numerical fragments.
    A linear relation between the mach number n_mach and the smoothing_cells required was assumed, motivated by the fact that the mach number increases linearly with fluid velocity.
    """

    k = (max_n_mach - n_mach) / (max_n_mach - min_n_mach)
    smoothing_cells = min_smoothing_cells + (1 - k) * (max_smoothing_cells - min_smoothing_cells)

    return smoothing_cells



# Grid size and configuration
num_cells = config.num_cells
x = jnp.linspace(0, 1, num_cells.x)
y = jnp.linspace(0, 1, num_cells.y)
X, Y = jnp.meshgrid(x, y, indexing="ij")

# Initialize state
rho = rho_back * jnp.ones_like(X)
u_x = v_a * jnp.ones_like(X)

# Slab mask
mask = (Y > 0.25) & (Y < 0.75)

# between y = 0.25 and y = 0.75 set u_x to -0.5 and rho to 2.0
y_center = 0.5
slab_radius = 0.25
smoothing_cells = calc_smoothing_cells(n_mach)
print(smoothing_cells)
smoothing_length = smoothing_cells / num_cells.x 

u_x = slab_profile(v_a, -v_a, Y, y_center, slab_radius, smoothing_length)
rho = slab_profile(rho_back, rho_slab, Y, y_center, slab_radius, smoothing_length)
p = jnp.ones((num_cells.x, num_cells.x)) * p_0

# deterministic setup
# u_y = 0.01 * jnp.sin(2 * jnp.pi * X)

# random initialization
key = PRNGKey(0)
num_sims = 1

# training data (num_sims x matrix of dim num_cells x num_cells)
data = jnp.zeros((num_sims, num_cells.x, num_cells.x))

for i in range(num_sims):
  print("finished iteration", i)
  key, subkey = jax.random.split(key)

  # KHI-suited random Fourier perturbation
  u_y = random_khi_fourier_modes(
      subkey,
      X,
      Y,
      amplitude=0.01,
      k_min=1,
      k_max=8,
      shear_layers=(0.25, 0.75),
      width=0.03,
      spectral_slope=1.0,
  )

  # Initial state
  initial_state = construct_primitive_state(
      config=config,
      registered_variables=registered_variables,
      density=rho,
      velocity_x=u_x,
      velocity_y=u_y,
      gas_pressure=p,
  )

  config = finalize_config(config, initial_state.shape)

  final_state = time_integration(
      initial_state,
      config,
      params,
      registered_variables,
  )

  jnp.save(f"test/final_state_{i}", final_state)
  plt.imshow(final_state[0, :, :])
  plt.savefig(f"test/final_state{i}.png")
