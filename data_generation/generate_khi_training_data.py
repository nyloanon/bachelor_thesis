"""Generate slab-KHI training data for the emulator.

Each sample is one finite-difference (WENO) simulation of the smoothed slab at a
Mach number drawn from [mach_min, mach_max], perturbed by a clean, low-amplitude,
seeded dominant mode (see ``_khi_slab_common``). The trajectory is stored at
``--num-timepoints`` snapshots evenly spaced over [0, age * t_KH], so every sample
is a short time series that spans the growth (low Mach) to suppression (high Mach)
behaviour with a single, Mach-independent smoothing length.

The recipe is fixed and physically motivated -- sigma = lambda / 64 (two cells,
~1.5% of the perturbation wavelength) for every Mach number, no Mach-dependent
smoothing. Sample ``j`` is fully determined by ``(--seed, j)``, so runs are
reproducible and can be sharded across GPUs/processes by index range.

Each sample is written to ``<out-dir>/sample_<j>.npz`` with:

    states          float array (num_timepoints, 4, Nx, Ny)
                    channels = [density, velocity_x, velocity_y, pressure]
    channel_names   the four channel labels
    time_points     snapshot times
    time_fractions  snapshot times in units of t_KH
    mach            the Mach number of this sample
    kh_time         the linear KH time
    perturbation_*  the drawn mode content / phases / amplitude
    plus static metadata (chi, gamma, box_size, num_cells, smoothing_length).

A one-off ``dataset_meta.json`` records the run-wide configuration. Existing
sample files are skipped, so the generator is resumable and shardable.

Examples:
    # 200 samples, 100 time points each, Mach in [0.5, 1.8]
    python generate_khi_training_data.py --num-samples 200 --num-timepoints 100

    # shard across two GPUs (run each on its own GPU; autocvd picks a free one)
    python generate_khi_training_data.py --start-index 0   --num-samples 100
    python generate_khi_training_data.py --start-index 100 --num-samples 100
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import json
from pathlib import Path
from timeit import default_timer as timer

# third-party
import numpy as np

# astronomix
from astronomix import (
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import SnapshotSettings

# shared slab setup
from data_generation._khi_slab_common import (
    BOX_SIZE,
    DENSITY_CONTRAST,
    FIX_STRATEGY,
    GAMMA,
    build_config,
    build_params,
    build_sample_initial_state,
    kelvin_helmholtz_time,
    sampled_perturbation_params,
    smoothing_length_for,
)

CHANNEL_NAMES = ("density", "velocity_x", "velocity_y", "pressure")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate slab-KHI training-data trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples", type=int, default=64,
                        help="number of samples to generate this invocation")
    parser.add_argument("--start-index", type=int, default=0,
                        help="index of the first sample (for sharding across processes)")
    parser.add_argument("--num-timepoints", type=int, default=100,
                        help="snapshots stored per trajectory, spanning [0, age * t_KH]")
    parser.add_argument("--mach-min", type=float, default=0.5)
    parser.add_argument("--mach-max", type=float, default=1.8)
    parser.add_argument("--mach", type=float, default=None,
                        help="fix the Mach number for every sample (overrides the range)")
    parser.add_argument("--num-cells", type=int, default=256)
    parser.add_argument("--age", type=float, default=1.2,
                        help="trajectory length in units of the KH time t_KH")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; sample j is determined by (seed, j)")
    parser.add_argument("--out-dir", type=str, default="data/khi_training")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def sample_mach(rng, args):
    """The Mach number for one sample: fixed if requested, else uniform in range."""

    if args.mach is not None:
        return float(args.mach)
    return float(rng.uniform(args.mach_min, args.mach_max))


def write_dataset_meta(args, out_dir, smoothing_length):
    """Record the run-wide configuration once, for provenance."""

    meta = {
        "description": "Slab Kelvin-Helmholtz training trajectories (finite-difference WENO).",
        "channel_names": list(CHANNEL_NAMES),
        "num_timepoints": args.num_timepoints,
        "mach_min": args.mach_min,
        "mach_max": args.mach_max,
        "fixed_mach": args.mach,
        "num_cells": args.num_cells,
        "age_in_kh_times": args.age,
        "base_seed": args.seed,
        "dtype": args.dtype,
        "density_contrast": DENSITY_CONTRAST,
        "gamma": GAMMA,
        "box_size": BOX_SIZE,
        "smoothing_length": smoothing_length,
        "smoothing_recipe": FIX_STRATEGY.name,
    }
    with open(out_dir / "dataset_meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)


def main():
    args = parse_args()
    np_dtype = np.float32 if args.dtype == "float32" else np.float64

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The state shape is identical for every sample, so build and finalise the
    # config (and JIT-warm the integrator) once.
    config = build_config(
        num_cells=args.num_cells,
        return_snapshots=True,
        num_snapshots=args.num_timepoints,
        snapshot_settings=SnapshotSettings(return_states=True),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    channel_indices = [
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.pressure_index,
    ]

    probe_state = build_sample_initial_state(
        0, args.mach_min, config, registered_variables, helper_data
    )
    config = finalize_config(config, probe_state.shape)

    smoothing_length = smoothing_length_for(FIX_STRATEGY, args.mach_min)
    write_dataset_meta(args, out_dir, smoothing_length)

    approx_mb = (
        args.num_timepoints * len(CHANNEL_NAMES) * args.num_cells**2
        * np.dtype(np_dtype).itemsize / 1e6
    )
    print(
        f"Generating samples [{args.start_index}, "
        f"{args.start_index + args.num_samples}) into {out_dir} "
        f"(~{approx_mb:.0f} MB each, {args.dtype})"
    )

    for offset in range(args.num_samples):
        index = args.start_index + offset
        out_path = out_dir / f"sample_{index:06d}.npz"
        if out_path.exists():
            print(f"  [{index}] exists, skipping")
            continue

        # Sample j is fully determined by (seed, j): draw the Mach number and a
        # perturbation seed from one per-sample stream.
        rng = np.random.default_rng([args.seed, index])
        mach = sample_mach(rng, args)
        perturbation_seed = int(rng.integers(1 << 31))
        recipe = sampled_perturbation_params(perturbation_seed)

        kh_time = kelvin_helmholtz_time(mach)
        t_end = args.age * kh_time

        initial_state = build_sample_initial_state(
            perturbation_seed, mach, config, registered_variables, helper_data
        )

        start = timer()
        snapshot_data = time_integration(
            initial_state,
            config,
            build_params(t_end),
            registered_variables,
        )

        # np.asarray forces the device -> host transfer (and thus completion).
        states = np.asarray(snapshot_data.states)[:, channel_indices, :, :].astype(np_dtype)
        time_points = np.asarray(snapshot_data.time_points)

        np.savez_compressed(
            out_path,
            states=states,
            channel_names=np.asarray(CHANNEL_NAMES),
            time_points=time_points.astype(np.float64),
            time_fractions=(time_points / kh_time).astype(np.float64),
            mach=np.float64(mach),
            kh_time=np.float64(kh_time),
            perturbation_modes=np.asarray(recipe["modes"]),
            perturbation_weights=np.asarray(recipe["weights"]),
            perturbation_phases=np.asarray(recipe["phases"]),
            perturbation_amplitude_fraction=np.float64(recipe["amplitude_fraction"]),
            density_contrast=np.float64(DENSITY_CONTRAST),
            gamma=np.float64(GAMMA),
            box_size=np.float64(BOX_SIZE),
            num_cells=np.int64(args.num_cells),
            smoothing_length=np.float64(smoothing_length),
        )
        print(
            f"  [{index}] M={mach:.3f} modes={recipe['modes']} "
            f"-> {out_path.name} ({timer() - start:.1f}s)"
        )

    print("Done.")


if __name__ == "__main__":
    main()