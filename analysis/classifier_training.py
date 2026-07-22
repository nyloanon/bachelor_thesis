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
# Dataset loading for CNN classifier
# ==========================================================================

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np

from analysis import classifier as model_lib

# ==========================================================================
# Stats
# ==========================================================================
# ==========================================================================
#  data loading, cache creation and normalization
# ==========================================================================

def compute_joint_stats(real_files, generated_files):
    """
    Compute normalization statistics over real + generated files.
    Skips corrupted files.
    """

    channel_sum = np.zeros(4, dtype=np.float64)
    channel_sqsum = np.zeros(4, dtype=np.float64)
    n_pixels = 0

    tf_min = np.inf
    tf_max = -np.inf

    machs = []

    valid_real_files = []
    valid_gen_files = []


    # =============================================================
    # Real simulations
    # =============================================================

    print("Processing real files...")

    for i, filename in enumerate(real_files):

        if i % 100 == 0:
            print(i, "/", len(real_files))

        try:
            with np.load(filename) as sample:

                states = sample["states"]       # (T,4,H,W)
                tf = sample["time_fractions"]
                mach = sample["mach"].item()


                channel_sum += states.sum(axis=(0,2,3))
                channel_sqsum += (states**2).sum(axis=(0,2,3))

                n_pixels += (
                    states.shape[0]
                    * states.shape[2]
                    * states.shape[3]
                )


                tf_min = min(tf_min, tf.min())
                tf_max = max(tf_max, tf.max())

                machs.append(mach)

                valid_real_files.append(filename)


        except Exception as e:

            print("\nSkipping corrupted real file:")
            print(filename)
            print("Reason:", e)
            print()


    # =============================================================
    # Generated samples
    # =============================================================

    print("Processing generated files...")

    for i, filename in enumerate(generated_files):

        if i % 100 == 0:
            print(i, "/", len(generated_files))

        try:
            with np.load(filename) as data:

                states = data["states"]
                mach_values = data["mach_values"]
                tf_values = data["time_fraction_values"]

                # states:
                # (N_samples, N_mach, N_tf, 4,H,W)

                channel_sum += states.sum(axis=(0,1,2,4,5))
                channel_sqsum += (
                    states**2
                ).sum(axis=(0,1,2,4,5))


                n_pixels += (
                    states.shape[0]
                    * states.shape[1]
                    * states.shape[2]
                    * states.shape[4]
                    * states.shape[5]
                )


                tf_min = min(
                    tf_min,
                    tf_values.min()
                )

                tf_max = max(
                    tf_max,
                    tf_values.max()
                )


                # every generated sample has the same mach/tf grid
                mach_grid = np.repeat(
                    mach_values,
                    states.shape[0] * states.shape[2]
                )

                machs.extend(mach_grid)

                valid_gen_files.append(filename)


        except Exception as e:

            print("\nSkipping corrupted generated file:")
            print(filename)
            print("Reason:", e)
            print()


    print(
        f"Valid real files: {len(valid_real_files)} / {len(real_files)}"
    )

    print(
        f"Valid generated files: {len(valid_gen_files)} / {len(generated_files)}"
    )


    if len(valid_real_files) == 0 and len(valid_gen_files) == 0:
        raise RuntimeError(
            "No valid files found!"
        )


    # =============================================================
    # Final statistics
    # =============================================================

    channel_mean = channel_sum / n_pixels

    channel_std = np.sqrt(
        np.maximum(
            channel_sqsum / n_pixels - channel_mean**2,
            1e-12
        )
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


    return stats, valid_real_files, valid_gen_files

def get_stats(directory, real_files, gen_files):

    """Load classifier_norm_stats.npz from directory, or compute (and save) it if missing."""

    stat_file = os.path.join(directory, "classifier_norm.npz")

    if os.path.exists(stat_file):

        print("loading existing statistics")

        data = np.load(stat_file)

        stats = {k: data[k] for k in

                 ("channel_mean", "channel_std", "tf_min", "tf_max", "mach_mean", "mach_std")}

        return stats, real_files, gen_files

# ==========================================================================
# Normalisation
# ==========================================================================

def normalize_states(states, channel_mean, channel_std):
    eps = 1e-6
    mean = channel_mean[:, None, None]
    std = channel_std[:, None, None]
    return (states - mean) / (std + eps)


def normalize_time_fraction(tf, tf_min, tf_max):
    eps = 1e-6
    return (tf - tf_min) / (tf_max - tf_min + eps)


def normalize_mach(mach, mach_mean, mach_std):
    return (mach - mach_mean) / (mach_std + 1e-6)


# ==========================================================================
# Build classifier dataset
# ==========================================================================

def build_classifier_dataset(
    real_files,
    generated_files,
    stats,
):

    states = []
    tfs = []
    machs = []
    labels = []

    # -------------------------------------------------------------
    # Real simulations
    # -------------------------------------------------------------

    print(f"Loading {len(real_files)} real simulations...")

    for filename in real_files:

        with np.load(filename) as sample:

            sim_states = sample["states"]              # (T,4,H,W)
            sim_tf = sample["time_fractions"]          # (T,)
            mach = sample["mach"].item()

            for i in range(sim_states.shape[0]):

                states.append(sim_states[i])
                tfs.append(sim_tf[i])
                machs.append(mach)
                labels.append(0)

    # -------------------------------------------------------------
    # Generated samples
    # -------------------------------------------------------------
    print("Loading generated samples...")

    for filename in generated_files:

        with np.load(filename) as data:

            gen = data["states"]
            mach_values = data["mach_values"]
            tf_values = data["time_fraction_values"]

            for s in range(gen.shape[0]):
                for m, mach in enumerate(mach_values):
                    for t, tf in enumerate(tf_values):

                        states.append(gen[s, m, t])
                        machs.append(mach)
                        tfs.append(tf)
                        labels.append(1)

    states = np.asarray(states, dtype=np.float32)
    tfs = np.asarray(tfs, dtype=np.float32)
    machs = np.asarray(machs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    print(f"Real samples      : {np.sum(labels == 0)}")
    print(f"Generated samples : {np.sum(labels == 1)}")
    # -------------------------------------------------------------
    # Normalisation
    # -------------------------------------------------------------
    states = normalize_states(
        states,
        stats["channel_mean"],
        stats["channel_std"],
    )

    tfs = normalize_time_fraction(
        tfs,
        stats["tf_min"],
        stats["tf_max"],
    )

    machs = normalize_mach(
        machs,
        stats["mach_mean"],
        stats["mach_std"],
    )

    # -------------------------------------------------------------
    # Shuffle once
    # -------------------------------------------------------------
    perm = np.random.RandomState(seed).permutation(len(states))

    n_val = int(0.1 * len(states))

    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_dataset = {
        k: jnp.asarray(v[train_idx])
        for k, v in dataset.items()
    }

    val_dataset = {
        k: jnp.asarray(v[val_idx])
        for k, v in dataset.items()
    }

    return train_dataset, val_dataset

# ==========================================================================
# Cache creation
# ==========================================================================

def load_cache(dataset, start_idx, cache_size):

    end_idx = min(start_idx + cache_size, len(dataset["states"]))

    cache = {

        "states": dataset["states"][start_idx:end_idx],
        "time_fractions": dataset["time_fractions"][start_idx:end_idx],
        "mach": dataset["mach"][start_idx:end_idx],
        "labels": dataset["labels"][start_idx:end_idx],

    }

    return cache


# ==========================================================================
# Batch sampling
# ==========================================================================

def cached_batch(key, cache, batch_size):

    idx = jax.random.randint(
        key,
        (batch_size,),
        0,
        cache["states"].shape[0],
    )

    return (

        cache["states"][idx],
        cache["time_fractions"][idx],
        cache["mach"][idx],
        cache["labels"][idx],

    )


# ==========================================================================
#  training setup
# ==========================================================================

@eqx.filter_value_and_grad
def loss_fn(model, states, tfs, machs, labels):

    logits = jax.vmap(model)(states, tfs, machs)
    logits = logits.squeeze(-1)

    loss = optax.sigmoid_binary_cross_entropy(
        logits,
        labels.astype(jnp.float32),
    )

    return loss.mean()

@eqx.filter_jit
def train_step(
    model,
    opt_state,
    optimizer,
    states,
    tfs,
    machs,
    labels,
):

    loss, grads = loss_fn(
        model,
        states,
        tfs,
        machs,
        labels,
    )

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        model,
    )

    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss

# ==========================================================================
#  validation accuracy and loss
# ==========================================================================

@eqx.filter_jit
def accuracy(
    model,
    states,
    tfs,
    machs,
    labels,
):

    logits = jax.vmap(model)(
        states,
        tfs,
        machs,
    ).squeeze(-1)


    probs = jax.nn.sigmoid(logits)

    preds = probs > 0.5

    return jnp.mean(
        preds == labels.astype(bool)
    )

def evaluate_accuracy(
    model,
    dataset,
    batch_size,
):

    states = dataset["states"]
    tfs = dataset["time_fractions"]
    machs = dataset["mach"]
    labels = dataset["labels"]


    correct = 0
    total = len(labels)


    for i in range(0, total, batch_size):

        batch = slice(
            i,
            min(i + batch_size, total)
        )


        acc = accuracy(
            model,
            states[batch],
            tfs[batch],
            machs[batch],
            labels[batch],
        )


        correct += float(acc) * len(labels[batch])


    return correct / total

@eqx.filter_jit
def batch_loss(
    model,
    states,
    tfs,
    machs,
    labels,
):

    logits = jax.vmap(model)(
        states,
        tfs,
        machs,
    ).squeeze(-1)


    loss = optax.sigmoid_binary_cross_entropy(
        logits,
        labels.astype(jnp.float32),
    )

    return loss.mean()

def evaluate_loss(
    model,
    dataset,
    batch_size,
):

    states = dataset["states"]
    tfs = dataset["time_fractions"]
    machs = dataset["mach"]
    labels = dataset["labels"]


    total_loss = 0.0
    total = len(labels)


    for i in range(0, total, batch_size):

        batch = slice(
            i,
            min(i + batch_size, total)
        )


        loss = batch_loss(
            model,
            states[batch],
            tfs[batch],
            machs[batch],
            labels[batch],
        )


        total_loss += float(loss) * len(labels[batch])


    return total_loss / total

# ==========================================================================
#  main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_dir", type=str, default="khi_training_data")
    parser.add_argument("--gen_data_dir", type=str, default="snap_conditioned_unet_generations")
    parser.add_argument("--ckpt-dir", type=str, default="analysis/classifier_checkpoints")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--val-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--steps-per-cache", type=int, default=128)
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- data + normalisation statistics ----
    real_files = sorted(glob.glob(os.path.join(args.real_data_dir, "sample_*.npz")))
    gen_files = sorted(glob.glob(os.path.join(args.gen_data_dir, "grid_generations_*.npz")))

    stats, real_files, gen_files = compute_joint_stats(real_files, gen_files)
    np.savez(os.path.join(args.ckpt_dir, "classifier_norm_stats.npz"),**stats)   

    # ---- classifier dataset ----
    train_dataset, val_dataset = build_classifier_dataset(real_files, gen_files, stats)

    # --- model / optimizer -------
    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)
    model = model_lib.create_model(model_key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=args.lr,
            weight_decay=1e-4,
        )
    )

    opt_state = optimizer.init(
        eqx.filter(model, eqx.is_array)
    )

    # --- training loop ----
    best_val = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    steps_recorded = []

    for step in range(args.steps):

        key, batch_key = jax.random.split(key)

        states, tfs, machs, labels = cached_batch(
            batch_key,
            train_dataset,
            args.batch_size,
        )


        model, opt_state, loss = train_step(
            model,
            opt_state,
            optimizer,
            states,
            tfs,
            machs,
            labels,
        )


        if step % 100 == 0:

            val_acc = evaluate_accuracy(
                model,
                val_dataset,
                args.val_batch,
            )

            val_loss = evaluate_loss(
                model,
                val_dataset,
                args.val_batch
            )

            train_losses.append(float(loss))
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            steps_recorded.append(step)

            print(
                f"step {step} "
                f"loss={float(loss):.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

            if val_acc > best_val:

                best_val = val_acc

                eqx.tree_serialise_leaves(
                    os.path.join(
                        args.ckpt_dir,
                        "best_classifier.eqx"
                    ),
                    model,
                )

    model = eqx.tree_deserialise_leaves(
                os.path.join(
                    args.ckpt_dir,
                    "best_classifier.eqx"
                ),
                model,
            )
    save_model(model, args.ckpt_dir)

    np.savez(
        os.path.join(args.ckpt_dir, "classifier_training_history.npz"),
        steps=np.asarray(steps_recorded),
        train_loss=np.asarray(train_losses),
        val_loss=np.asarray(val_losses),
        val_acc=np.asarray(val_accs),
    )

if __name__ == "__main__":
    main()