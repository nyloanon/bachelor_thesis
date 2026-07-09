"""Shared slab Kelvin-Helmholtz setup for the Mach-number smoothing studies.

Both ``khi_mach_smoothing.py`` (smoothing-strategy comparison) and
``khi_mach_suppression.py`` (growth-rate / suppression sweep) build the same
physical slab initial condition; the physics, the initial-condition recipes and
the flagship finite-difference configuration live here so the two scripts stay
in sync.

The slab: a dense stream (density contrast ``chi = rho_slab / rho_back``) flows
through a lighter background at uniform pressure, with periodic boundaries along
the flow (x) and open boundaries transverse to it (y). This mirrors the emulator
training-data generator, but ties the interface smoothing length to the
perturbation wavelength rather than to the Mach number (Mandelker et al. 2016 /
Roediger et al. 2013).
"""

# general
from typing import NamedTuple

# third-party
import jax
import jax.numpy as jnp
import numpy as np

# astronomix
from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    KINEMATIC_VISCOSITY,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)


# -------------------------------------------------------------
# ==================== ↓ Physical setup ↓ =====================
# -------------------------------------------------------------

GAMMA = 5.0 / 3.0
BOX_SIZE = 1.0
NUM_CELLS = 256

BACKGROUND_DENSITY = 1.0
SLAB_DENSITY = 2.0
PRESSURE = 2.5

SLAB_CENTER = 0.5
SLAB_RADIUS = 0.25  # slab occupies y in [0.25, 0.75]

# Background sound speed and density contrast.
BACKGROUND_SOUND_SPEED = float(jnp.sqrt(GAMMA * PRESSURE / BACKGROUND_DENSITY))
DENSITY_CONTRAST = SLAB_DENSITY / BACKGROUND_DENSITY

# Atwood-like density factor entering the KH growth time (Roediger et al. 2013).
DELTA = (SLAB_DENSITY + BACKGROUND_DENSITY) ** 2 / (SLAB_DENSITY * BACKGROUND_DENSITY)

# Critical Mach number of the vortex sheet (Mandelker et al. 2016, Eq. 22, with
# the paper's 3/2 exponent). Above it the dominant surface modes are suppressed
# by compressibility. The generator's ``n_mach_crit`` used an exponent of 2/3,
# which is a typo and underestimates the true critical Mach.
CRITICAL_MACH = (1.0 + DENSITY_CONTRAST ** (-1.0 / 3.0)) ** (3.0 / 2.0)

# The dominant (longest) perturbation wavelength; the box holds one k=1 mode.
DOMINANT_WAVELENGTH = BOX_SIZE

# Amplitude of the transverse-velocity perturbation as a fraction of the shear
# velocity (Roediger's v_0 = 0.1 v_s, softened a little here).
PERTURBATION_FRACTION = 0.05

# Deterministic perturbation phases, shared across all runs so that only the
# smoothing/Mach change between panels.
PERTURBATION_KEY = jax.random.PRNGKey(0)


# -------------------------------------------------------------
# ================= ↓ Initialisation recipes ↓ ================
# -------------------------------------------------------------

class Strategy(NamedTuple):
    """One initial-condition recipe compared in the smoothing sweep.

    Args:
        name: Short label used in the figures.
        smoothing_mode: ``"fixed_cells"`` pins the interface thickness to a
            fixed number of grid cells (the naive baselines); ``"mach_ramp"``
            reproduces the generator's Mach-dependent thickness;
            ``"wavelength"`` ties it to the perturbation wavelength (the fix).
        smoothing_cells: Interface thickness in cells for ``"fixed_cells"``.
        smoothing_fraction: Interface thickness as a fraction of the *shortest*
            seeded wavelength for ``"wavelength"`` (Mandelker/Roediger use
            1/102 ... 1/50).
        max_mode: Highest seeded Fourier mode, and the mode that sets the
            wavelength-tied smoothing length.
        perturbation: ``"clean"`` seeds an explicit list of long modes with
            fixed phase (textbook single-mode KHI); ``"random"`` seeds a random
            band-limited spectrum (variety for training data).
        modes: The along-flow wavenumbers excited by the ``"clean"`` recipe.
        amplitude_fraction: Perturbation velocity as a fraction of the shear
            velocity. Kept small so the mode rolls up cleanly before going
            nonlinear, then develops rich small-scale secondary structure.
    """

    name: str
    smoothing_mode: str
    smoothing_cells: float
    smoothing_fraction: float
    max_mode: int
    perturbation: str = "random"
    modes: tuple = (1, 2, 3, 4)
    amplitude_fraction: float = PERTURBATION_FRACTION


# The physically motivated, Mach-independent recipe: a fixed interface thickness
# tied to the perturbation wavelength (sigma = lambda / 64, ~1.5% of lambda, two
# cells at N=256 -- within Roediger's 1-2% range), a clean low-amplitude dominant
# mode that rolls up into detailed billows at low Mach, and no Mach dependence.
FIX_STRATEGY = Strategy(
    name="fix: sigma = lambda / 64 (fixed, 2 cells), clean low-amplitude mode",
    smoothing_mode="wavelength",
    smoothing_cells=0.0,
    smoothing_fraction=1.0 / 64.0,
    max_mode=2,
    perturbation="clean",
    modes=(2,),
    amplitude_fraction=0.005,
)


def mach_ramp_smoothing_cells(
    mach_number,
    min_cells=2.0,
    max_cells=12.0,
    min_mach=0.5,
    max_mach=2.4,
):
    """Reproduce the generator's Mach-dependent smoothing length.

    The interface is thin (detailed KHI) at low Mach and thick (stable slab) at
    high Mach. This is the unphysical band-aid the fix removes.
    """

    ramp = (max_mach - mach_number) / (max_mach - min_mach)
    return min_cells + (1.0 - ramp) * (max_cells - min_cells)


def smoothing_length_for(strategy, mach_number):
    """Return the tanh smoothing length sigma for a strategy and Mach number."""

    grid_spacing = BOX_SIZE / NUM_CELLS

    if strategy.smoothing_mode == "fixed_cells":
        return strategy.smoothing_cells * grid_spacing

    if strategy.smoothing_mode == "mach_ramp":
        return mach_ramp_smoothing_cells(mach_number) * grid_spacing

    if strategy.smoothing_mode == "wavelength":
        # The shortest seeded wavelength; sigma is a fixed fraction of it, hence
        # independent of the Mach number.
        shortest_wavelength = BOX_SIZE / strategy.max_mode
        return strategy.smoothing_fraction * shortest_wavelength

    raise ValueError(f"unknown smoothing mode {strategy.smoothing_mode!r}")


def slab_profile(background_value, slab_value, y_coordinate, smoothing_length):
    """Smooth top-hat: ``background_value`` outside the slab, ``slab_value`` inside.

    A product of two tanh transitions places smoothed interfaces at the two slab
    edges ``SLAB_CENTER +/- SLAB_RADIUS``.
    """

    lower_edge = 1.0 + jnp.tanh(
        (SLAB_RADIUS - (y_coordinate - SLAB_CENTER)) / smoothing_length
    )
    upper_edge = 1.0 + jnp.tanh(
        (SLAB_RADIUS + (y_coordinate - SLAB_CENTER)) / smoothing_length
    )
    return background_value + 0.25 * (slab_value - background_value) * lower_edge * upper_edge


def transverse_velocity_perturbation(
    key,
    x_coordinate,
    y_coordinate,
    amplitude,
    max_mode,
    envelope_width,
):
    """Band-limited multi-mode transverse-velocity perturbation at both shear layers.

    The perturbation is a sum of the first ``max_mode`` sinusoids along the flow,
    localised at the two slab edges by Gaussian envelopes. Band-limiting to long
    wavelengths keeps all injected energy at well-resolved scales.

    Args:
        key: PRNG key; only the mode phases and relative weights are random, so
            that the perturbation shape is shared across Mach numbers.
        x_coordinate: Flow-direction coordinate field.
        y_coordinate: Transverse coordinate field.
        amplitude: Peak perturbation velocity (scaled with the shear velocity by
            the caller).
        max_mode: Highest Fourier mode included (k = 1 ... max_mode).
        envelope_width: Gaussian width localising the perturbation at each edge.

    Returns:
        The transverse-velocity perturbation field.
    """

    modes = jnp.arange(1, max_mode + 1)

    key_weights, key_phases = jax.random.split(key)

    # Mildly red spectrum, normalised so the summed amplitude is controlled.
    weights = jax.random.normal(key_weights, (modes.shape[0],)) / modes
    weights = weights / jnp.sqrt(jnp.sum(weights**2) + 1e-30)

    phases = jax.random.uniform(
        key_phases,
        (modes.shape[0],),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    wave = jnp.sum(
        weights[:, None, None]
        * jnp.sin(
            2.0 * jnp.pi * modes[:, None, None] * x_coordinate[None, :, :]
            + phases[:, None, None]
        ),
        axis=0,
    )

    # Localise around both slab edges.
    envelope = jnp.exp(
        -0.5 * ((y_coordinate - (SLAB_CENTER - SLAB_RADIUS)) / envelope_width) ** 2
    ) + jnp.exp(
        -0.5 * ((y_coordinate - (SLAB_CENTER + SLAB_RADIUS)) / envelope_width) ** 2
    )

    return amplitude * envelope * wave


def kelvin_helmholtz_time(mach_number):
    """The linear KH growth time for the dominant mode at a given Mach number."""

    relative_velocity = mach_number * BACKGROUND_SOUND_SPEED
    return float(jnp.sqrt(DELTA)) * DOMINANT_WAVELENGTH / relative_velocity


def build_config(
    return_snapshots=False,
    num_snapshots=10,
    snapshot_settings=None,
    diffusion=False,
    num_cells=NUM_CELLS,
):
    """Flagship finite-difference (WENO) configuration for the slab problem.

    Args:
        return_snapshots: Store intermediate snapshots.
        num_snapshots: Number of snapshots when ``return_snapshots`` is set.
        snapshot_settings: Which per-snapshot diagnostics to record.
        diffusion: Enable the Navier-Stokes viscous source term (finite
            difference only), used to set a finite Reynolds number so the rolls
            stay laminar instead of breaking up into grid-scale turbulence.
        num_cells: Grid resolution per axis.
    """

    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        progress_bar=False,
        dimensionality=2,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        num_timesteps=50000,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        diffusion=diffusion,
        viscosity_type=KINEMATIC_VISCOSITY,
        return_snapshots=return_snapshots,
        num_snapshots=num_snapshots,
        snapshot_settings=snapshot_settings,
    )


def build_initial_state(strategy, mach_number, config, registered_variables, helper_data):
    """Build the slab primitive state for a strategy and Mach number.

    Returns:
        The initial primitive state and the smoothing length used.
    """

    cell_centers = helper_data.geometric_centers
    x_coordinate = cell_centers[:, :, 0]
    y_coordinate = cell_centers[:, :, 1]

    # The two streams move at +/- v_shear with total relative velocity
    # v_rel = n_mach * c_background; the slab is the dense, receding stream.
    relative_velocity = mach_number * BACKGROUND_SOUND_SPEED
    shear_velocity = relative_velocity / 2.0

    smoothing_length = smoothing_length_for(strategy, mach_number)

    density = slab_profile(
        BACKGROUND_DENSITY,
        SLAB_DENSITY,
        y_coordinate,
        smoothing_length,
    )
    velocity_x = slab_profile(
        shear_velocity,
        -shear_velocity,
        y_coordinate,
        smoothing_length,
    )

    # Perturbation amplitude scales with the shear velocity (Roediger); the
    # envelope is a few smoothing lengths wide but never narrower than the
    # dominant-wavelength shear-layer scale.
    amplitude = strategy.amplitude_fraction * shear_velocity
    envelope_width = max(3.0 * smoothing_length, 0.02)

    if strategy.perturbation == "clean":
        velocity_y = clean_mode_perturbation(
            x_coordinate,
            y_coordinate,
            amplitude=amplitude,
            modes=strategy.modes,
            envelope_width=envelope_width,
        )
    else:
        velocity_y = transverse_velocity_perturbation(
            PERTURBATION_KEY,
            x_coordinate,
            y_coordinate,
            amplitude=amplitude,
            max_mode=strategy.max_mode,
            envelope_width=envelope_width,
        )

    gas_pressure = jnp.full_like(density, PRESSURE)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        gas_pressure=gas_pressure,
    )

    return initial_state, smoothing_length


def sampled_perturbation_params(seed):
    """Draw a clean-but-varied perturbation recipe for one training sample.

    A dominant long mode (k = 2 or 3) plus a small admixture of a neighbouring
    mode, each at a random phase, with a low amplitude. This keeps the rolls
    clean while varying their wavelength, position and asymmetry between samples.

    Args:
        seed: Integer seed; the same seed always yields the same recipe.

    Returns:
        A dict with ``modes``, ``weights``, ``phases`` and ``amplitude_fraction``.
    """

    rng = np.random.default_rng(seed)
    dominant = int(rng.choice([2, 3]))
    secondary = max(1, dominant + int(rng.choice([-1, 1])))
    return {
        "modes": (dominant, secondary),
        "weights": (1.0, float(rng.uniform(0.15, 0.4))),
        "phases": (
            float(rng.uniform(0.0, 2.0 * np.pi)),
            float(rng.uniform(0.0, 2.0 * np.pi)),
        ),
        "amplitude_fraction": float(rng.uniform(0.004, 0.007)),
    }


def build_sample_initial_state(
    seed,
    mach_number,
    config,
    registered_variables,
    helper_data,
):
    """Slab initial state with a seeded clean-but-varied perturbation.

    Uses the fixed, wavelength-tied smoothing length of the reference recipe; the
    perturbation mode content, phases and amplitude are drawn from ``seed`` (the
    same seed is used at every Mach number, so within one sample only the Mach
    number changes).
    """

    recipe = sampled_perturbation_params(seed)

    cell_centers = helper_data.geometric_centers
    x_coordinate = cell_centers[:, :, 0]
    y_coordinate = cell_centers[:, :, 1]

    shear_velocity = mach_number * BACKGROUND_SOUND_SPEED / 2.0
    smoothing_length = smoothing_length_for(FIX_STRATEGY, mach_number)

    density = slab_profile(BACKGROUND_DENSITY, SLAB_DENSITY, y_coordinate, smoothing_length)
    velocity_x = slab_profile(shear_velocity, -shear_velocity, y_coordinate, smoothing_length)
    velocity_y = clean_mode_perturbation(
        x_coordinate,
        y_coordinate,
        amplitude=recipe["amplitude_fraction"] * shear_velocity,
        modes=recipe["modes"],
        envelope_width=max(3.0 * smoothing_length, 0.02),
        weights=recipe["weights"],
        phases=recipe["phases"],
    )
    gas_pressure = jnp.full_like(density, PRESSURE)

    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        gas_pressure=gas_pressure,
    )


def build_params(t_end, viscosity=0.0):
    """Finite-difference simulation parameters for the slab problem.

    Args:
        t_end: Integration end time.
        viscosity: Kinematic viscosity (used only when the config enables
            diffusion); sets the Reynolds number of the shear layer.
    """

    return SimulationParams(
        t_end=t_end,
        C_cfl=1.5,
        gamma=GAMMA,
        viscosity=viscosity,
    )


def clean_mode_perturbation(
    x_coordinate,
    y_coordinate,
    amplitude,
    modes,
    envelope_width,
    weights=None,
    phases=None,
):
    """Transverse-velocity perturbation from a few clean modes.

    Unlike the random broadband seeding, this excites only an explicit list of
    long-wavelength modes, producing regular, laminar Kelvin-Helmholtz rolls at
    the two shear layers -- the textbook single-mode KHI rather than a mixture of
    competing billows. With default (zero) phases and equal weights it is
    deterministic; per-mode ``weights`` and ``phases`` allow controlled variety
    (a dominant mode plus a small admixture at a random phase) for training data.

    Args:
        x_coordinate: Flow-direction coordinate field.
        y_coordinate: Transverse coordinate field.
        amplitude: Peak perturbation velocity.
        modes: Iterable of integer along-flow wavenumbers to excite.
        envelope_width: Gaussian width localising the perturbation at each edge.
        weights: Per-mode relative weights (default: all ones).
        phases: Per-mode phases in radians (default: all zero).

    Returns:
        The transverse-velocity perturbation field.
    """

    if weights is None:
        weights = [1.0] * len(modes)
    if phases is None:
        phases = [0.0] * len(modes)

    wave = jnp.zeros_like(x_coordinate)
    for mode, weight, phase in zip(modes, weights, phases):
        wave = wave + weight * jnp.sin(2.0 * jnp.pi * mode * x_coordinate + phase)
    # Normalise by the summed weight so the peak amplitude stays controlled.
    wave = wave / sum(abs(weight) for weight in weights)

    envelope = jnp.exp(
        -0.5 * ((y_coordinate - (SLAB_CENTER - SLAB_RADIUS)) / envelope_width) ** 2
    ) + jnp.exp(
        -0.5 * ((y_coordinate - (SLAB_CENTER + SLAB_RADIUS)) / envelope_width) ** 2
    )

    return amplitude * envelope * wave
