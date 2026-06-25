# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from scipy.fft import fft2, ifft2
import numpy as np

def fourier_energy_spectrum2d(u):
    """
    u: velocity vector u = [[ux], [uy]]
    """
    # safety check
    if len(u) != 2 or len(u[0]) != len(u[1]):
        raise ValueError f"Velocity vector u is not two dimensional or dimension of two components ux and uy do not match! \n Shapes: {u[0].shape}, {u[1].shape}."

    # fourier transform the components u
    # fourier coeff
    v = fft2(u)
    v = np.asarray(v)

    # fourier energy
    e = 0.5 * v ** 2
    print(e)

    # check if total fourier energy is the same as real kinetic energy
    e_tot = sum(e)
    E_tot = sum(0.5 * np.asarray(u)**2)
    if e_tot != E_tot:
        return e, False
    
    return e, True


# test
u = [[0.1, 2.1], [0.1]]
e, truth = fourier_energy_spectrum2d(u)

    