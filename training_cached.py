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
import time

import argparse
import glob
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from unet_models import unet_flow_film_cached


# ==========================================================================
#  rectified-flow batch sampling
# ==========================================================================

def sample_transition(cache, sim_idx, t_idx):
    """
    Extract one transition from a cached simulation.
    """

    states = cache["states"][sim_idx]
    times = cache["time_points"][sim_idx]
    mach = cache["mach"][sim_idx]

    current = states[t_idx]
    target = states[t_idx + 1]

    dt = times[t_idx + 1] - times[t_idx]

    return current, target, dt, mach

def cached_batch(key, cache, batch_size):

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
            cache["states"].shape[1] - 1
        )

        current, target, dt, mach = sample_transition(
            cache,
            sim_idx,
            t_idx
        )

        x0 = jax.random.normal(key_noise, target.shape)

        t_rf = jax.random.uniform(key_rf, ())

        x_t_rf = (1.0 - t_rf) * x0 + t_rf * target

        v_target = target - x0

        return (
            x_t_rf,
            current,
            t_rf,
            dt,
            mach,
            v_target
        )

    return jax.vmap(single_sample)(keys)


def loss_function(model, x_t_rf, current, t_rf, dt, mach, v_target):
    v_pred = jax.vmap(model)(x_t_rf, current, t_rf, dt, mach)
    return jnp.mean((v_pred - v_target) ** 2)


@eqx.filter_jit
def train_step(model, ema_model, opt_state, x_t_rf, current, t_rf, dt, mach, v_target, optimizer, ema_decay):
    loss, grads = eqx.filter_value_and_grad(loss_function)(model, x_t_rf, current, t_rf, dt, mach, v_target)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    # exponential moving average of the weights (used for sampling)
    ema_model = jax.tree_util.tree_map(
        lambda e, m: ema_decay * e + (1.0 - ema_decay) * m if eqx.is_array(e) else e,
        ema_model, model,
    )
    return model, ema_model, opt_state, loss



# ==========================================================================
#  data loading, cache creation and normalization
# ==========================================================================

def compute_stats(files):

    channel_sum = np.zeros(4, dtype=np.float64)
    channel_sqsum = np.zeros(4, dtype=np.float64)
    n_pixels = 0

    tp_min = np.inf
    tp_max = -np.inf

    machs = []
    valid_files = []

    for i, filename in enumerate(files):

        if i % 100 == 0:
            print(i, "/", len(files))

        try:
            with np.load(filename) as sample:

                states = sample["states"]          # (100,4,256,256)

                channel_sum += states.sum(axis=(0,2,3))
                channel_sqsum += (states**2).sum(axis=(0,2,3))

                n_pixels += (
                    states.shape[0]
                    * states.shape[2]
                    * states.shape[3]
                )

                tp = sample["time_points"]

                tp_min = min(tp_min, tp.min())
                tp_max = max(tp_max, tp.max())

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
        channel_sqsum / n_pixels - channel_mean**2
    )

    machs = np.asarray(machs)

    stats = {

        "channel_mean": jnp.asarray(channel_mean),

        "channel_std": jnp.asarray(channel_std),

        "tp_min": jnp.asarray(tp_min),

        "tp_max": jnp.asarray(tp_max),

        "mach_mean": jnp.asarray(machs.mean()),

        "mach_std": jnp.asarray(machs.std()),

    }

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

def normalize_time_points(time_points, tp_min, tp_max):
    eps = 1e-6
    return (time_points - tp_min) / (tp_max - tp_min + eps)

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
    time_points = []
    machs = []

    for filename in cache_files:

        with np.load(filename) as sample:

            states.append(sample["states"])
            time_points.append(sample["time_points"])
            machs.append(sample["mach"].item())
    
    states = jnp.asarray(np.asarray(states))
    time_points = jnp.asarray(np.asarray(time_points))
    machs = jnp.asarray(np.asarray(machs))

    states = normalize_states(
        states,
        stats["channel_mean"],
        stats["channel_std"]
    )

    time_points = normalize_time_points(
        time_points,
        stats["tp_min"],
        stats["tp_max"]
    )

    machs = normalize_mach(
        machs,
        stats["mach_mean"],
        stats["mach_std"]
    )

    cache = {

        "states": states,

        "time_points": time_points,

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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_size", type=int, default=512,)
    parser.add_argument("--steps_per_cache", type=int, default=128,)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)


    # --- data loading and stat file for normalization -----
    files = load_files(
    args.data_dir,
    args.max_samples
    )

    print(f"Found {len(files)} simulations.")

    start = time.time()

    stat_file = os.path.join(args.ckpt_dir, "norm_stats.npz")

    if os.path.exists(stat_file):
        print("loading existing statistics")
        data = np.load(stat_file)

        stats = {
            "channel_mean": data["channel_mean"],
            "channel_std": data["channel_std"],
            "tp_min": data["tp_min"],
            "tp_max": data["tp_max"],
            "mach_mean": data["mach_mean"],
            "mach_std": data["mach_std"],
        }

    else:
        print("computing statistics")
        stats, valid_files = compute_stats(files)

        print(
            "stats time:",
            time.time()-start,
            "seconds"
        )

        np.savez(
        os.path.join(args.ckpt_dir, "norm_stats.npz"),
        channel_mean=np.array(stats["channel_mean"]),
        channel_std=np.array(stats["channel_std"]),
        tp_min=float(stats["tp_min"]),
        tp_max=float(stats["tp_max"]),
        mach_mean=float(stats["mach_mean"]),
        mach_std=float(stats["mach_std"])
        )

    # ------- validation cache -------------
    perm = np.random.RandomState(args.seed).permutation(len(valid_files))

    valid_files = [valid_files[i] for i in perm]

    n_val = max(1, int(len(files) * args.val_frac))

    val_files = valid_files[:n_val]

    train_files = valid_files[n_val:]

    val_cache = load_cache(
    val_files,
    stats,
    0,
    len(val_files),
    )

    val_key = jax.random.key(12345)

    x_rf_val, current_val, t_rf_val, dt_val, mach_val, v_val = cached_batch(
        val_key,
        val_cache,
        batch_size=64,
    )
    print("val_files created!")
    # ---- model / optimiser ----
    key = jax.random.key(args.seed)
    key, mkey = jax.random.split(key)
    model = unet_flow_film.create_model(mkey)
    ema_model = model
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"model params: {n_params / 1e6:.2f} M")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    perm = np.random.permutation(len(train_files))
    shuffled_files = [train_files[i] for i in perm]

    cache = load_cache(shuffled_files, stats, 0, args.cache_size)
    batch = cached_batch(
    jax.random.key(0),
    cache,
    batch_size=2,
    )
    print("before")
    out = jax.vmap(model)(
        batch[0],  # x_t_rf
        batch[1],  # current
        batch[2],  # t_rf
        batch[3],  # dt
        batch[4],  # mach
    )
    print("after")

    print(out.shape)

    # # ---- training loop -----
    # global_step = 0

    # key_train = jax.random.key(args.seed + 1)

    # checkpoint_steps = {
    #     args.steps // 4,
    #     args.steps // 2,
    #     3 * args.steps // 4,
    #     args.steps,
    # }

    # while global_step < args.steps:

    #     perm = np.random.permutation(len(train_files))
    #     shuffled_files = [train_files[i] for i in perm]

    #     cache_start = 0

    #     while (
    #         cache_start < len(shuffled_files)
    #         and global_step < args.steps
    #     ):

    #         cache = load_cache(
    #             shuffled_files,
    #             stats,
    #             cache_start,
    #             args.cache_size,
    #         )

    #         for _ in range(args.steps_per_cache):

    #             if global_step >= args.steps:
    #                 break

    #             key_train, batch_key = jax.random.split(key_train)

    #             batch = cached_batch(
    #                 batch_key,
    #                 cache,
    #                 args.batch_size
    #             )

    #             (
    #                 x_t_rf,
    #                 current,
    #                 t_rf,
    #                 dt,
    #                 mach,
    #                 v_target
    #             ) = batch

    #             model, ema_model, opt_state, loss = train_step(
    #                 model,
    #                 ema_model,
    #                 opt_state,
    #                 x_t_rf,
    #                 current,
    #                 t_rf,
    #                 dt,
    #                 mach,
    #                 v_target,
    #                 optimizer,
    #                 args.ema_decay
    #             )

    #             global_step += 1

    #             loss_history.append(float(loss))

    #             if global_step % 500 == 0:

    #                 val_mse = float(
    #                     loss_function(
    #                         ema_model,
    #                         x_rf_val,
    #                         current_val,
    #                         t_rf_val,
    #                         dt_val,
    #                         mach_val,
    #                         v_val
    #                     )
    #                 )

    #                 val_history.append(val_mse)
    #                 val_steps.append(global_step)

    #                 print(
    #                     f"step {global_step:6d} | "
    #                     f"train {float(loss):.5f} | "
    #                     f"val {val_mse:.5f}",
    #                     flush=True
    #                 )

    #             if global_step in checkpoint_steps:

    #                 unet_flow_film.save_model(
    #                     model,
    #                     os.path.join(
    #                         args.ckpt_dir,
    #                         f"unet_{global_step}.eqx"
    #                     ),
    #                 )

    #                 unet_flow_film.save_model(
    #                     ema_model,
    #                     os.path.join(
    #                         args.ckpt_dir,
    #                         f"unet_ema_{global_step}.eqx"
    #                     ),
    #                 )

    #         cache_start += args.cache_size


if __name__ == "__main__":
    main()
