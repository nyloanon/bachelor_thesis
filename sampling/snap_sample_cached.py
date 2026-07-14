# ==== GPU / XLA memory config (must precede any jax import) ====
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Sampling from the trained rectified-flow KHI model
#
#  Integrates the learned velocity field from Gaussian noise (t=0) to a KHI
#  final state (t=1) with a Heun (RK2) integrator, denormalises with the
#  saved per-channel statistics, and writes the full 4-channel field.
# ==========================================================================

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from models/unet_models import snap_conditioned_unet_flow_film_cached as model_lib

CHANNELS = ["density", "velocity_x", "velocity_y", "pressure"]


# ==========================================================================
#  Heun (RK2) rectified-flow sampler
# ==========================================================================

def sample(model, x, steps=100):
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        v1 = model(x, t)
        x_euler = x + dt * v1
        t_next = min(t + dt, 1.0)
        v2 = model(x_euler, t_next)
        x = x + dt * 0.5 * (v1 + v2)
    return x


def batched_sample(model, noises, t_fracs, machs, steps=100):
    """
    noises (N, 4, H, W), t_fracs (N,), machs (N,) --> (N, 4, H, W)
    """
    fn = jax.vmap(lambda z, tf, m: sample(model, z, tf, m, steps))
    return np.asarray(fn(noises, t_fracs, machs))

# ==========================================================================
#  main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="snap_conditioned_unet_checkpoints/unet_ema_final.eqx")
    parser.add_argument("--ckpt-dir", type=str, default="snap_conditioned_unet_checkpoints")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--mach", type=float, nargs="+", default=[0.6, 1.0, 1.6], help="Mach numbers to condition on (rows)")
    parser.add_argument("--time fraction", type=float, nargs="+", default=[0.2, 0.6, 1.0], help="physical time fraction t/t_KH to condition on (columns)")
    parser.add_argument("--channel", type=int, default=0, help=" channel to display in the grid (0=density)")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="snap_conditioned_unet_generations")
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load model + normalisation stats ----
    key = jax.random.key(0)
    model = model_lib.load_model(args.ckpt, key)

    stats = np.load(os.path.join(args.ckpt_dir, "norm_stats.npz"))
    channel_mean = stats["channel_mean"].reshape(4, 1, 1)
    channel_std = stats["channel_std"].reshape(4, 1, 1)
    tf_min, tf_max = float(stats["tf_min"]), float(stats["tf_max"])
    mach_mean, mach_std = float(stats["mach_mean"]), float(stats["mach_std"])

    def norm_tf(tf):
        return (tf - tf_min) / (tf_max - tf_min + 1e-6)

    def norm_mach(m):
        return (m - mach_mean) / mach_std

    machs = list(args.mach)
    tfracs = list(args.time_fraction)
    n_rows, n_cols = len(machs), len(tfracs)

    # ---- generate ----
    # one shared noise per column so differences across a row are purely the
    # conditioning (same latent, different Mach) -- makes the effect legible.

    base_noise = jax.random.normal(jax.random.key(args.seed), (n_cols, 4, 256, 256))

    # build the flattened (mach, t_frac) grid 
    grid_noise, grid_tf, grid_mach = [], [], []

    for r, m in enumerate(machs):

        for c, tf in enumerate(tfracs):

            grid_noise.append(base_noise[c])
            grid_tf.append(norm_tf(tf))
            grid_mach.append(norm_mach(m))

    grid_noise = jnp.asarray(np.stack(grid_noise))
    grid_tf = jnp.asarray(np.array(grid_tf, dtype=np.float32))
    grid_mach = jnp.asarray(np.array(grid_mach, dtype=np.float32))

    gen = batched_sample(model, grid_noise, grid_tf, grid_mach, args.steps)  # (R*C,4,H,W)
    gen = gen * channel_std[None] + channel_mean[None]                       # denormalise
    gen = gen.reshape(n_rows, n_cols, 4, 256, 256)

    np.save(os.path.join(args.out_dir, "grid_generations.npy"), gen)
    print(f"generated grid {gen.shape} -> {args.out_dir}/grid_generations.npy")
    ch = args.channel
    name = CHANNELS[ch]



    # shared colour scale across the whole grid for the chosen channel
    vmin = float(gen[:, :, ch].min())
    vmax = float(gen[:, :, ch].max())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for r in range(n_rows):

        for c in range(n_cols):
            ax = axes[r][c]
            im = ax.imshow(gen[r, c, ch].T, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])

            if r == 0:
                ax.set_title(f"t/t_KH = {tfracs[c]:.2f}")

            if c == 0:
                ax.set_ylabel(f"Mach = {machs[r]:.2f}", fontsize=12)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(f"Conditioned KHI generation — {name} (rows: Mach, cols: time fraction)", fontsize=14)

    out = os.path.join(args.out_dir, f"conditioning_grid_{name}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")



    # per-(mach,time) mean density, a quick numeric check that conditioning bites
    print("\nmean density by (Mach, t/t_KH):")
    header = "        " + "".join(f"tf={tf:<6.2f}" for tf in tfracs)
    print(header)

    for r, m in enumerate(machs):
        row = f"M={m:<5.2f} " + "".join(f"{gen[r, c, 0].mean():<9.3f}" for c in range(n_cols))
        print(row)

if __name__ == "__main__":

    main()