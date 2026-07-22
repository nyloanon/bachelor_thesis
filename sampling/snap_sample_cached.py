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
from timeit import default_timer as timer

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from models.unet_models import snap_condtioned_unet_flow_film_cached as model_lib

CHANNELS = ["density", "velocity_x", "velocity_y", "pressure"]


# ==========================================================================
#  Heun (RK2) rectified-flow sampler
# ==========================================================================

def sample(model, x, t_frac, mach, steps=100):
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        v1 = model(x, t, t_frac, mach)
        x_euler = x + dt * v1
        t_next = min(t + dt, 1.0)
        v2 = model(x_euler, t_next, t_frac, mach)
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
    parser.add_argument("--ckpt", type=str, default="snap_conditioned_unet_checkpoints/conditioned_unet_ema_final.eqx")
    parser.add_argument("--ckpt-dir", type=str, default="snap_conditioned_unet_checkpoints")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--mach", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], help="Mach numbers to condition on (rows)")
    parser.add_argument("--time_fraction", type=float, nargs="+", default=[1.0], help="Physical time fraction t/t_KH to condition on (columns)")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3], help="Channels to display in the grid (0=density, 1=velocity_x, 2=velocity_y, 3=pressure)")
    parser.add_argument("--num_samples", type=int, default=10)
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

    # ---- timed generation ----
    # one shared noise per column so differences across a row are purely the
    # conditioning (same latent, different Mach) -- makes the effect legible.

    base_noise = jax.random.normal(jax.random.key(args.seed), (args.num_samples,n_cols, 4, 256, 256))

    # build the flattened (mach, t_frac) grid 
    mach_ranges = np.array_split(machs, 4)
    ranges = ["1", "2", "3", "4"]

    for i, mach_range in enumerate(mach_ranges):
        grid_noise, grid_tf, grid_mach = [], [], []
        n_rows_chunk = len(mach_range)
        range_ = ranges[i]

        for s in range(args.num_samples):
            for m in mach_range:
                for c, tf in enumerate(tfracs):
                    grid_noise.append(base_noise[s, c])
                    grid_tf.append(norm_tf(tf))
                    grid_mach.append(norm_mach(m))

        grid_noise = jnp.asarray(np.stack(grid_noise))
        grid_tf = jnp.asarray(np.array(grid_tf, dtype=np.float32))
        grid_mach = jnp.asarray(np.array(grid_mach, dtype=np.float32))

        start = timer()

        gen = batched_sample(model, grid_noise, grid_tf, grid_mach, args.steps)  # (R*C,4,H,W)
        gen = gen * channel_std[None] + channel_mean[None]                       # denormalise
        gen = gen.reshape(args.num_samples, n_rows_chunk, n_cols, 4, 256, 256)
        jax.block_until_ready(gen)
        gen = np.asarray(gen)
        
        np.savez(
                os.path.join(args.out_dir, f"grid_generations_{range_}.npz"),
                states=gen,
                mach_values=np.asarray(mach_range),
                time_fraction_values=np.asarray(tfracs)
            )

        time_gen = timer() - start

        print(f"grid generations {gen.shape} -> {args.out_dir}/grid_generations_{range_}.npz. Generation time = {time_gen:.1f} s")
        channels = args.channels

            
        for ch in channels:

            # shared colour scale across the whole grid for the chosen channel
            vmin = float(gen[:, :, :, ch].min())
            vmax = float(gen[:, :, :, ch].max())

            fig, axes = plt.subplots(n_rows_chunk, args.num_samples, figsize=(3.2 * args.num_samples, 3.2 * n_rows_chunk), squeeze=False)
            name = CHANNELS[ch]

            for r in range(n_rows_chunk):

                for c in range(args.num_samples):
                    ax = axes[r][c]
                    im = ax.imshow(gen[c, r, 0, ch].T, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([])

                    if r == 0:
                        ax.set_title(f"Sample {c+1}")

                    if c == 0:
                        ax.set_ylabel(f"Mach = {mach_range[r]:.2f}", fontsize=12)

            fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
            fig.suptitle(f"Conditioned KHI generation — {name} (rows: Mach, cols: samples)\n"
            f"t/t_KH = {tfracs[0]:.2f}"
            , fontsize=14)

            out = os.path.join(args.out_dir, f"conditioning_grid_{range_}_{name}.png")
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":

    main()