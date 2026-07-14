# ==== GPU / XLA memory config (must precede any jax import) ====
import os
os.environ.setdefault("JAX_ENABLE_X64", "False")

# allocate GPU memory on demand rather than grabbing ~75% up front, and keep
# convolution autotuning on (disabling it makes each step ~50x slower).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Rectified-flow training for the 4-channel KHI field sampler
#
#  Learns v(x_t, t) = x1 - x0 where x0 ~ N(0, I), x1 is a (normalised) KHI
#  final state, and x_t = (1 - t) x0 + t x1.  Sampling then integrates this
#  velocity field from noise (t=0) to data (t=1).
# ==========================================================================

import argparse
import glob

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from models/unet_models import snap_conditioned_unet_flow_film_cached as model_lib


# ==========================================================================
#  rectified-flow batch sampling
# ==========================================================================

def sample_snap(cache, sim_idx, t_idx):
    """
    Extract one normalised snapshot from simulation and return physical properties.
    """

    snap = cache["states"][sim_idx][t_idx]
    tf = cache["time_fractions"][sim_idx][t_idx]
    mach = cache["mach"][sim_idx]


    return snap, tf, mach

def cached_batch(key, cache, batch_size):
    """
    Draw batch of rectified-flow training tuples from the in-memory cache.
    """
    keys = jax.random.split(key, batch_size)

    def single_sample(k):

        key_sim, key_time, key_noise, key_rf = jax.random.split(k, 4)

        sim_idx = jax.random.randint(
            key_sim,
            (),
            0,
            cache["states"].shape[0]
        )

        t_idx = jax.random.randint(
            key_time,
            (),
            0,
            cache["states"].shape[1]
        )

        snap, tf, mach = sample_transition(
            cache,
            sim_idx,
            t_idx
        )

        x0 = jax.random.normal(key_noise, target.shape)

        t_rf = jax.random.uniform(key_rf, ())

        x_t = (1.0 - t_rf) * x0 + t_rf * snap

        v_target = snap - x0

        return (
            x_t,
            t_rf,
            tf,
            mach,
            v_target
        )

    return jax.vmap(single_sample)(keys)


def loss_function(model, x_t, current, t_rf, tf, mach, v_target):
    v_pred = jax.vmap(model)(x_t, t_rf, tf, mach)
    return jnp.mean((v_pred - v_target) ** 2)


@eqx.filter_jit
def train_step(model, ema_model, opt_state, x_t, t_rf, tf, mach, v_target, optimizer, ema_decay):
    loss, grads = eqx.filter_value_and_grad(loss_function)(model, x_t, t_rf, tf, mach, v_target)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    # exponential moving average of the weights (used for sampling)
    ema_model = jax.tree_util.tree_map(
        lambda e, m: ema_decay * e + (1.0 - ema_decay) * m if eqx.is_array(e) else e,
        ema_model, model,
    )
    return model, ema_model, opt_state, loss

val_loss = eqx.filter_jit(loss_function)

# ==========================================================================
#  data loading, cache creation and normalization
# ==========================================================================

def compute_stats(files):
    """
    Per-channel mean/std, time-fraction range and Mach mean/std over files.
    """
    channel_sum = np.zeros(4, dtype=np.float64)
    channel_sqsum = np.zeros(4, dtype=np.float64)
    n_pixels = 0
    tf_min = np.inf
    tf_max = -np.inf

    machs = []
    valid_files = []

    for i, filename in enumerate(files):

        if i % 100 == 0:
            print(i, "/", len(files))

        try:
            with np.load(filename) as sample:

                states = sample["states"]          # (T, 4, H, W)

                channel_sum += states.sum(axis=(0,2,3))
                channel_sqsum += (states**2).sum(axis=(0,2,3))

                n_pixels += (
                    states.shape[0]
                    * states.shape[2]
                    * states.shape[3]
                )

                tf = sample["time_fractions"]

                tf_min = min(tf_min, tf.min())
                tf_max = max(tf_max, tf.max())

                machs.append(sample["mach"].item())

                valid_files.append(filename)

        except Exception as e:

            print("\nSkipping corrupted file:")
            print(filename)
            print("Reason:", e)
            print()

    print(
        f"Successfully processed {len(valid_files)} / {len(files)} files"
    )

    if len(valid_files) == 0:
        raise RuntimeError("No valid .npz files found!")

    channel_mean = channel_sum / n_pixels

    channel_std = np.sqrt(
       np.maximum(channel_sqsum / n_pixels - channel_mean**2, 1e-12)
    )

    machs = np.asarray(machs)

    stats = {

        "channel_mean": channel_mean,

        "channel_std": channel_std,

        "tf_min": np.float64(tf_min),

        "tf_max": np.float64(tf_max),

        "mach_mean": np.float64(machs.mean()),

        "mach_std": np.float64(machs.std()),

    }

    return stats, valid_files

def get_stats(directory, files):

    """Load norm_stats.npz from directory, or compute (and save) it if missing."""

    stat_file = os.path.join(directory, "norm_stats.npz")

    if os.path.exists(stat_file):

        print("loading existing statistics")

        data = np.load(stat_file)

        stats = {k: data[k] for k in

                 ("channel_mean", "channel_std", "tf_min", "tf_max", "mach_mean", "mach_std")}

        return stats, files
    
    print("computing normalisation statistics ...")

    stats, valid_files = compute_stats(files)
    np.savez(stat_file, **stats)
    print(f"wrote {stat_file}")

    return stats, valid_files

def normalize_states(states, channel_mean, channel_std):
    """
    states: (cache_size, 100, 4, 256, 256)
    channel_mean: (4,)
    channel_std: (4,)
    """

    mean = channel_mean[None, None, :, None, None]
    std = channel_std[None, None, :, None, None]
    eps = 1e-6

    return (states - mean) / (std + eps)

def normalize_time_fraction(tf, tf_min, tf_max):
    eps = 1e-6
    return (tf - tf_min) / (tf_max - tf_min + eps)

def normalize_mach(mach, mach_mean, mach_std):

    return (mach - mach_mean) / mach_std

def load_cache(files, stats, start_idx, cache_size=32):
    """
    Only select the files that need to be cached. If the to be created cache is too large for the current epoch, the cache size is adjusted. 
    Then the state, time and mach information are extracted from the .npz files and collected in lists. The lists are normalized with the according stats dictionary entries.
    From these lists, the cache, a dictionary of JAX arrays is created and returned.

    -----------input--------------
    files: .npz files
    stats: dict
    start_idx: int
    cache_size: int

    -----------output-------------
    cache: dict[jnp.array]
    """
    if start_idx + cache_size > len(files):
        # adjust cache size
        cache_size = len(files) - start_idx 

    cache_files = files[start_idx : start_idx + cache_size]

    # create cache as dict of JAX array 
    states = []
    time_fracs = []
    machs = []

    for filename in cache_files:

        with np.load(filename) as sample:

            states.append(sample["states"])
            time_fracs.append(sample["time_fractions"])
            machs.append(sample["mach"].item())
    
    states = jnp.asarray(np.asarray(states))
    time_fracs = jnp.asarray(np.asarray(time_fracs))
    machs = jnp.asarray(np.asarray(machs))

    states = normalize_states(
        states,
        stats["channel_mean"],
        stats["channel_std"]
    )

    time_fracs = normalize_time_points(
        time_fracs,
        stats["tf_min"],
        stats["tf_max"]
    )

    machs = normalize_mach(
        machs,
        stats["mach_mean"],
        stats["mach_std"]
    )

    cache = {

        "states": states,

        "time_fractions": time_fracs,

        "mach": machs

    }

    return cache

def load_files(data_dir, max_samples=None):
    files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No data found in {data_dir}/final_state_*.npz")

    return files


# ==========================================================================
#  main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="khi_data")
    parser.add_argument("--ckpt-dir", type=str, default="conditioned_unet_checkpoints")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--val-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--steps-per-cache", type=int, default=128)
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- data + normalisation statistics ----
    files = load_files(args.data_dir, args.max_samples)
    print(f"Found {len(files)} simulations.")
    stats, files = get_stats(args.ckpt_dir, files)
    print("channel mean:", np.round(np.asarray(stats["channel_mean"]), 4))
    print("channel std :", np.round(np.asarray(stats["channel_std"]), 4))
    print(f"time fraction range: [{float(stats['tf_min']):.4f}, {float(stats['tf_max']):.4f}]")
    print(f"mach mean/std: {float(stats['mach_mean']):.4f} / {float(stats['mach_std']):.4f}")

    # convert stats to jnp for the (jitted) normalisation used inside load_cache
    stats = {k: jnp.asarray(v) for k, v in stats.items()}

    # ---- train / val split ----
    perm = np.random.RandomState(args.seed).permutation(len(files))
    files = [files[i] for i in perm]

    n_val = max(1, int(len(files) * args.val_frac))
    val_files, train_files = files[:n_val], files[n_val:]
    val_cache = load_cache(val_files, stats, 0, min(args.cache_size, len(val_files)))
    x_t_val, t_rf_val, t_frac_val, mach_val, v_val = cached_batch(
        jax.random.key(12345), val_cache, args.val_batch
    )
    print("validation batch created.")

    # ---- model / optimiser ----
    key = jax.random.key(args.seed)
    key, mkey = jax.random.split(key)
    model = model_lib.create_model(mkey)
    ema_model = model
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"model params: {n_params / 1e6:.2f} M")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ---- training loop (streaming cache over the file list) ----
    global_step = 0
    loss_history, val_history, val_steps = [], [], []
    key_train = jax.random.key(args.seed + 1)
    checkpoint_steps = {args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps}

    while global_step < args.steps:
        perm = np.random.permutation(len(train_files))
        shuffled_files = [train_files[i] for i in perm]

        cache_start = 0

        while cache_start < len(shuffled_files) and global_step < args.steps:
            cache = load_cache(shuffled_files, stats, cache_start, args.cache_size)

            for _ in range(args.steps_per_cache):
                if global_step >= args.steps:
                    break

                key_train, batch_key = jax.random.split(key_train)
                x_t, t_rf, t_frac, mach, v_target = cached_batch(
                    batch_key, cache, args.batch_size
                )

                model, ema_model, opt_state, loss = train_step(
                    model, ema_model, opt_state,
                    x_t, t_rf, t_frac, mach, v_target,
                    optimizer, args.ema_decay,
                )
                global_step += 1

                loss_history.append(float(loss))

                if global_step % 1000 == 0:
                    vm = float(val_loss(ema_model, x_t_val, t_rf_val, t_frac_val, mach_val, v_val))
                    val_history.append(vm)
                    val_steps.append(global_step)
                    print(f"step {global_step:6d} | train {float(loss):.5f} | val(ema) {vm:.5f}",flush=True)

                if global_step in checkpoint_steps:
                    model_lib.save_model(
                        model, os.path.join(args.ckpt_dir, f"conditioned_unet_{global_step}.eqx")
                    )
                    model_lib.save_model(
                        ema_model, os.path.join(args.ckpt_dir, f"conditioned_unet_ema_{global_step}.eqx")
                    )

            cache_start += args.cache_size

    model_lib.save_model(model, os.path.join(args.ckpt_dir, "conditioned_unet_final.eqx"))
    model_lib.save_model(ema_model, os.path.join(args.ckpt_dir, "conditioned_unet_ema_final.eqx"))
    print("saved final checkpoints")

    # ---- loss plot ----
    plt.figure()
    plt.plot(loss_history, alpha=0.4, label="train (per step)")
    if len(loss_history) > 100:
        w = 100
        smooth = np.convolve(loss_history, np.ones(w) / w, mode="valid")
        plt.plot(np.arange(w - 1, len(loss_history)), smooth, label="train (smoothed)")

    plt.plot(val_steps, val_history, "o-", label="val (ema)")
    plt.yscale("log")
    plt.xlabel("step"); plt.ylabel("MSE loss"); plt.legend()
    plt.title("(Mach, time)-conditioned rectified-flow training")
    plt.savefig("loss_history.png", dpi=120)
    print("wrote loss_history.png")

if __name__ == "__main__":
    main()