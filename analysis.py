# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax.numpy as jnp
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================================
#  Physical analysis
# ==========================================================================

# ==========================================================================
#  Binning mode constants
# ==========================================================================

LOG_BINNING = 0       # Logarithmic bins — constant dk/k, smoothest at high k
INTEGER_BINNING = 1   # Integer mode shells — dk = 2*pi/L, good statistics per bin
PHYSICAL_BINNING = 2  # Physical wavenumber — dk = 1, finest resolution (default)


# ==========================================================================
#  Shared helpers
# ==========================================================================

def _wavenumber_bins2d(Nx, Ny, binning=PHYSICAL_BINNING):
    """
    Compute shell-binning indices for spectral accumulation.

    Args:
        Nx, Ny: Grid dimensions.
        binning: One of LOG_BINNING, INTEGER_BINNING, PHYSICAL_BINNING.

    Returns:
        k_idx: Flat int32 bin indices, shape (Nx*Ny).
        n_bins: Number of bins.
        k_centers: Physical wavenumber bin centers, shape (n_bins,).
    """
    freq_x = jnp.fft.fftfreq(Nx, d=1.0 / Nx)  # integer mode numbers
    freq_y = jnp.fft.fftfreq(Ny, d=1.0 / Ny)
    FX, FY = jnp.meshgrid(freq_x, freq_y, indexing="ij")

    m_mag = jnp.sqrt(FX**2 + FY**2)  # integer mode magnitude
    max_mode = math.sqrt((Nx // 2) ** 2 + (Ny // 2) ** 2)

    if binning == LOG_BINNING:
        k_min = 1.0
        k_max = max_mode
        n_bins = max(Nx, Ny) // 4
        log_edges = jnp.logspace(
            jnp.log10(k_min * 0.5), jnp.log10(k_max + 0.5), n_bins + 1
        )
        k_idx = jnp.clip(
            jnp.digitize(m_mag.ravel(), log_edges) - 1, 0, n_bins - 1
        )
        # Geometric mean of edges, converted to physical wavenumber
        k_centers = 2.0 * jnp.pi * jnp.sqrt(log_edges[:-1] * log_edges[1:])
        return k_idx, n_bins, k_centers

    elif binning == INTEGER_BINNING:
        k_idx = m_mag.astype(jnp.int32).ravel()
        n_bins = int(max_mode) + 2
        k_centers = (jnp.arange(n_bins) + 0.5) * 2.0 * jnp.pi
        return k_idx, n_bins, k_centers

    else:  # PHYSICAL_BINNING
        k_phys = 2.0 * jnp.pi * m_mag
        k_idx = k_phys.astype(jnp.int32).ravel()
        n_bins = int(2.0 * math.pi * max_mode) + 2
        k_centers = jnp.arange(n_bins) + 0.5
        return k_idx, n_bins, k_centers


# ==========================================================================
#  Generic spectrum functions
# ==========================================================================

def vector_field_energy_spectrum(fx, fy, energy_coeff=1.0,
                                binning=PHYSICAL_BINNING):
    """
    Energy spectrum E(k) of a vector field (fx, fy).

    Satisfies: sum(E(k)) == mean(c * |f|^2) over the domain
    (shell-summed convention).

    Args:
        fx, fy: Field components, each shaped (Nx, Ny).
        energy_coeff: Scalar multiplier c (default 1.0).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        k_centers: Physical wavenumber bin centers.
        Ek: Energy spectrum per bin.

    Based on: https://qiauil.github.io/blog/2026/tke_spectrum/ and https://github.com/leo1200/astronomix/blob/main/astronomix/analysis_helpers/energy_spectrum.py
    """
    Nx, Ny = fx.shape
    N_total = float(Nx * Ny)

    fx_hat = jnp.fft.fftn(fx)
    fy_hat = jnp.fft.fftn(fy)

    energy_fft = energy_coeff * (
        jnp.abs(fx_hat) ** 2 + jnp.abs(fy_hat) ** 2
        ) / N_total**2

    k_idx, n_bins, k_centers = _wavenumber_bins2d(Nx, Ny, binning)
    Ek = jnp.zeros(n_bins).at[k_idx].add(energy_fft.ravel())
    return k_centers, Ek

# -------------------------------------------------------------------------
# HD physics wrapper
# -------------------------------------------------------------------------

def get_kinetic_energy_spectrum(vx, vy, rho, binning=PHYSICAL_BINNING):
    """
    Kinetic energy spectrum with density weighting.

    Uses w = sqrt(rho) * u so that sum(E_k) == mean(0.5 * rho * |u|^2).

    Args:
        vx, vy, vz: Velocity components, each (Nx, Ny, Nz).
        rho: Density field, (Nx, Ny, Nz).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        k_centers, Ek: Wavenumber bin centers and kinetic energy spectrum.
    """
    rho_sqrt = jnp.sqrt(rho)
    return vector_field_energy_spectrum(
        rho_sqrt * vx, rho_sqrt * vy,
        energy_coeff=0.5, binning=binning,
    )


def rho_power_spectrum(rho):
    """
    Compute the power spectrum P(k) for density.
    rho.shape = (n_data, 256, 256)
    return a list containing the power spectrum for all data
    """

    rho = rho - rho.mean(axis=(-2, -1), keepdims=True)
    n_data, N, _ = rho.shape

    # FFT
    rho_hat = jnp.fft.fft2(rho, axes=(-2, -1))
    power = jnp.abs(rho_hat)**2 

    # k grid 
    k_idx, n_bins, k_centers = _wavenumber_bins2d(N, N, binning=PHYSICAL_BINNING)
    k_idx = jnp.reshape(k_idx, shape=(N,N))
    k_idx_flat = k_idx.ravel()
    
    # allocate output
    Pk_all = []

    for j in range(n_data):
        p = power[j].ravel()

        # sum of power per bin
        Pk = jnp.bincount(
            k_idx_flat,
            weights=p,
            length=n_bins
        )

        # number of modes per bin (normalization)
        counts = jnp.bincount(
            k_idx_flat,
            length=n_bins
        )

        # avoid division by zero
        Pk = jnp.where(counts > 0, Pk / counts, 0.0)

        Pk_all.append(Pk)

    return k_centers, jnp.stack(Pk_all)


def density_distribution(rho_gen, rho_real):
    """
    Create and plot a density distribution for comparison of p(rho_real) and p(rho_gen).
    rho_gen.shape = (n_data, 256, 256)
    rho_real.shape = (n_data, 256, 256)
    """

    # flatten data for seaborn distplot
    rho_gen = rho_gen.flatten()
    rho_real = rho_real.flatten()

    fig, ax = plt.subplots()
    sns.kdeplot(rho_real, ax=ax, label="real data")
    sns.kdeplot(rho_gen, ax=ax, color="r", label="generated data")
    plt.xlabel("$\\rho$")
    plt.ylabel("$P(\\rho)$")
    plt.legend()
    plt.savefig("sns_distplot.png")
    
# ==========================================================================
#  Statistical analysis
# ==========================================================================

def calculate_fid(x, y):    
    mu_x, sigma_x = jnp.mean(x, axis=0), np.cov(x)
    mu_y, sigma_y = jnp.mean(y, axis=0), np.cov(y)

    covmean = jnp.sqrtm(sigma_x @ sigma_y)
    # check if covmean is complex and if so set all values to real
    if np.iscomplexobject(covmean):
        covmean = covmean.real
    
    d = (mu_x - mu_y)**2 + jnp.linalg.trace(sigma_x + sigma_y - 2*jnp.sqrt(covmean))

    return d


# ==========================================================================
#  Data import 
# ==========================================================================

files = sorted(glob.glob("unet_generations/test_generation_*.npy"))

generations = np.stack([np.load(f)for f in files]) 
print(generations.shape)

k_centers, Pk_all = rho_power_spectrum(generations)
plt.plot(k_centers, Pk_all[0])
plt.savefig('power_spec_test.png')
