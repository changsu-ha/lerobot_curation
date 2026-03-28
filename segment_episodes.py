#!/usr/bin/env python3
"""
Sample episodes from a LeRobot 2.0 dataset, compute FK, and segment each
episode into task phases:

    approach → grasp → move → insertion → place → move_to_ready

Segmentation is driven by gripper-state anchors (grasp / place events)
and end-effector velocity / position signals.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d, median_filter

from joint_to_cartesian import (
    build_joint_index_map,
    compute_fk_trajectory,
    extract_joint_array,
    extract_timestamps,
    load_episodes,
    load_info,
    load_robot_model,
    resolve_dataset_path,
    save_fk_results,
)

# Phase colours for visualisation
PHASE_COLORS = {
    "approach": "#4CAF50",       # green
    "grasp": "#FF9800",          # orange
    "move": "#2196F3",           # blue
    "insertion": "#9C27B0",      # purple
    "place": "#F44336",          # red
    "move_to_ready": "#607D8B",  # grey-blue
    "unknown": "#BDBDBD",        # light grey
}

PHASE_ORDER = [
    "approach", "grasp", "move", "insertion", "place", "move_to_ready",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GripperConfig:
    thresh_closed: float = 0.01
    thresh_open: float = 0.04
    dwell_frames: int = 6
    active_hand: str = "right"
    right_index: int = 20
    left_index: int = 32


@dataclass
class VelocityConfig:
    thresh_moving: float = 0.005
    thresh_stationary: float = 0.002
    dwell_frames: int = 10


@dataclass
class PreprocessConfig:
    gripper_median_kernel: int = 5
    pos_smooth_sigma: float = 3.0
    vel_smooth_sigma: float = 5.0


@dataclass
class SegmentConfig:
    gripper: GripperConfig = field(default_factory=GripperConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    phases: list[str] = field(default_factory=lambda: list(PHASE_ORDER))

    @classmethod
    def from_yaml(cls, path: str) -> "SegmentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg = cls()
        if "gripper" in d:
            cfg.gripper = GripperConfig(**d["gripper"])
        if "velocity" in d:
            cfg.velocity = VelocityConfig(**d["velocity"])
        if "preprocessing" in d:
            cfg.preprocessing = PreprocessConfig(**d["preprocessing"])
        if "phases" in d:
            cfg.phases = d["phases"]
        return cfg


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------
def preprocess_gripper(state_array: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """Extract and smooth the active gripper aperture signal."""
    idx = (cfg.gripper.right_index if cfg.gripper.active_hand == "right"
           else cfg.gripper.left_index)
    raw = state_array[:, idx].copy()
    return median_filter(raw, size=cfg.preprocessing.gripper_median_kernel)


def compute_ee_speed(positions: np.ndarray, timestamps: np.ndarray,
                     cfg: SegmentConfig) -> np.ndarray:
    """Compute smoothed end-effector speed (scalar, m/s)."""
    pos_smooth = gaussian_filter1d(
        positions, sigma=cfg.preprocessing.pos_smooth_sigma, axis=0,
    )
    dt = np.diff(timestamps)
    dt[dt == 0] = 1e-6  # avoid division by zero
    vel = np.diff(pos_smooth, axis=0) / dt[:, None]
    vel = np.vstack([vel, vel[-1:]])  # replicate last frame
    speed = np.linalg.norm(vel, axis=1)
    return gaussian_filter1d(speed, sigma=cfg.preprocessing.vel_smooth_sigma)


# ---------------------------------------------------------------------------
# Gripper anchor detection
# ---------------------------------------------------------------------------
def _dwell_check(signal: np.ndarray, start: int, n_frames: int,
                 condition_fn) -> bool:
    """Return True if condition_fn(signal[i]) is True for n_frames from start."""
    end = min(start + n_frames, len(signal))
    if end - start < n_frames:
        return False
    return all(condition_fn(signal[i]) for i in range(start, end))


def detect_gripper_anchors(gripper: np.ndarray, cfg: SegmentConfig):
    """
    Detect T_grasp (close event) and T_place (open event) using a
    hysteresis state-machine with dwell confirmation.

    Returns
    -------
    t_grasp : int | None
    t_place : int | None
    """
    tc = cfg.gripper.thresh_closed
    to = cfg.gripper.thresh_open
    dwell = cfg.gripper.dwell_frames

    state = "open"
    t_grasp = None
    t_place = None

    for t in range(len(gripper)):
        if state == "open" and gripper[t] < tc:
            if _dwell_check(gripper, t, dwell, lambda v: v < tc):
                t_grasp = t
                state = "closed"
        elif state == "closed" and gripper[t] > to:
            if _dwell_check(gripper, t, dwell, lambda v: v > to):
                t_place = t
                state = "open"
                break  # first open event after grasp

    return t_grasp, t_place


# ---------------------------------------------------------------------------
# Phase labelling
# ---------------------------------------------------------------------------
def label_phases(
    timestamps: np.ndarray,
    ee_pos: np.ndarray,
    speed: np.ndarray,
    gripper: np.ndarray,
    cfg: SegmentConfig,
) -> tuple[np.ndarray, dict]:
    """
    Assign a phase label to every timestep.

    Strategy
    --------
    1. Find gripper anchors (T_grasp, T_place) — strongest signals.
    2. Use velocity to distinguish movement vs stationary within each segment.
    3. Fallback to velocity-only when anchors are missing.

    Returns
    -------
    labels : np.ndarray of str, shape (N,)
    info   : dict with anchor indices and diagnostics
    """
    N = len(timestamps)
    labels = np.full(N, "unknown", dtype=object)

    t_grasp, t_place = detect_gripper_anchors(gripper, cfg)

    # Fallback: if no gripper anchor found, use velocity-based heuristic
    if t_grasp is None:
        t_grasp = _fallback_grasp(speed, cfg)
    if t_place is None and t_grasp is not None:
        t_place = _fallback_place(speed, cfg, t_grasp)

    vm = cfg.velocity.thresh_moving
    vs = cfg.velocity.thresh_stationary

    if t_grasp is not None and t_place is not None:
        # Segment 1: [0, t_grasp) → approach
        labels[0:t_grasp] = "approach"

        # Segment 2: [t_grasp, t_grasp + grasp_end) → grasp
        # Grasp phase = from gripper close until robot starts moving again
        grasp_end = t_grasp
        for t in range(t_grasp, t_place):
            if speed[t] > vm:
                if _speed_dwell(speed, t, cfg.velocity.dwell_frames, vm):
                    grasp_end = t
                    break
        else:
            grasp_end = t_place
        labels[t_grasp:grasp_end] = "grasp"

        # Segment 3: [grasp_end, insertion_start) → move
        # Segment 4: [insertion_start, t_place) → insertion
        # Insertion = last stationary period before t_place
        insertion_start = _find_last_stationary_start(
            speed, grasp_end, t_place, vs, cfg.velocity.dwell_frames,
        )
        if insertion_start is not None:
            labels[grasp_end:insertion_start] = "move"
            labels[insertion_start:t_place] = "insertion"
        else:
            labels[grasp_end:t_place] = "move"

        # Segment 5: [t_place, place_end) → place
        place_end = t_place
        for t in range(t_place, N):
            if speed[t] > vm:
                if _speed_dwell(speed, t, cfg.velocity.dwell_frames, vm):
                    place_end = t
                    break
        else:
            place_end = N
        labels[t_place:place_end] = "place"

        # Segment 6: [place_end, N) → move_to_ready
        if place_end < N:
            labels[place_end:N] = "move_to_ready"

    elif t_grasp is not None:
        # Only grasp detected, no place
        labels[0:t_grasp] = "approach"
        labels[t_grasp:N] = "grasp"
    else:
        # No anchors at all — label everything unknown
        pass

    info = {
        "t_grasp": t_grasp,
        "t_place": t_place,
    }
    return labels, info


def _speed_dwell(speed, start, n_frames, thresh_moving):
    """Check if speed stays above thresh_moving for n_frames."""
    end = min(start + n_frames, len(speed))
    return all(speed[i] > thresh_moving for i in range(start, end))


def _find_last_stationary_start(speed, seg_start, seg_end, thresh_stat,
                                dwell_frames):
    """Find the start of the last stationary region within [seg_start, seg_end)."""
    last_start = None
    t = seg_start
    while t < seg_end:
        if speed[t] < thresh_stat:
            candidate = t
            count = 0
            while t < seg_end and speed[t] < thresh_stat:
                count += 1
                t += 1
            if count >= dwell_frames:
                last_start = candidate
        else:
            t += 1
    return last_start


def _fallback_grasp(speed, cfg):
    """Fallback: find the first long stationary period (likely grasp)."""
    dwell = cfg.velocity.dwell_frames * 2  # require longer dwell for fallback
    vs = cfg.velocity.thresh_stationary
    N = len(speed)
    third = N // 3

    t = 0
    while t < third:
        if speed[t] < vs:
            start = t
            while t < third and speed[t] < vs:
                t += 1
            if t - start >= dwell:
                return start
        else:
            t += 1
    return None


def _fallback_place(speed, cfg, t_grasp):
    """Fallback: find the last long stationary period after t_grasp."""
    dwell = cfg.velocity.dwell_frames * 2
    vs = cfg.velocity.thresh_stationary
    N = len(speed)

    last = None
    t = t_grasp + 1
    while t < N:
        if speed[t] < vs:
            start = t
            while t < N and speed[t] < vs:
                t += 1
            if t - start >= dwell:
                last = start
        else:
            t += 1
    return last


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_episode_ids(dataset_dir: Path, n_samples: int) -> list[int]:
    """Uniformly sample episode IDs from the dataset."""
    data_dir = dataset_dir / "data"
    all_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    all_ids = [int(f.stem.split("_")[-1]) for f in all_files]

    if n_samples >= len(all_ids):
        return all_ids

    # Uniform sampling
    indices = np.linspace(0, len(all_ids) - 1, n_samples, dtype=int)
    return [all_ids[i] for i in indices]


# ---------------------------------------------------------------------------
# Compute phase boundary table
# ---------------------------------------------------------------------------
def compute_boundaries(labels: np.ndarray, timestamps: np.ndarray) -> list[dict]:
    """
    Extract contiguous phase segments from the label array.

    Returns list of dicts: [{phase, start_idx, end_idx, start_time, end_time}, ...]
    """
    boundaries = []
    if len(labels) == 0:
        return boundaries

    current_phase = labels[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != current_phase:
            boundaries.append({
                "phase": str(current_phase),
                "start_idx": int(start_idx),
                "end_idx": int(i - 1),
                "start_time": float(timestamps[start_idx]),
                "end_time": float(timestamps[i - 1]),
            })
            current_phase = labels[i]
            start_idx = i

    boundaries.append({
        "phase": str(current_phase),
        "start_idx": int(start_idx),
        "end_idx": int(len(labels) - 1),
        "start_time": float(timestamps[start_idx]),
        "end_time": float(timestamps[-1]),
    })
    return boundaries


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_segmented_episode(
    timestamps: np.ndarray,
    state_pos: np.ndarray,
    state_rpy: np.ndarray,
    action_pos: np.ndarray,
    action_rpy: np.ndarray,
    speed: np.ndarray,
    gripper: np.ndarray,
    labels: np.ndarray,
    ee_name: str,
    episode_id: int,
    save_path: Path | None = None,
):
    """Plot Cartesian poses + speed + gripper with phase-coloured background."""
    fig, axes = plt.subplots(9, 1, figsize=(16, 28), sharex=True)

    # Row 0-2: position x, y, z
    pos_labels = ["x [m]", "y [m]", "z [m]"]
    for i in range(3):
        ax = axes[i]
        ax.plot(timestamps, state_pos[:, i], label="state", color="tab:blue", lw=1.0)
        ax.plot(timestamps, action_pos[:, i], label="action", color="tab:red", lw=1.0, alpha=0.7)
        ax.set_ylabel(pos_labels[i], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    # Row 3-5: orientation roll, pitch, yaw
    rpy_labels = ["roll [rad]", "pitch [rad]", "yaw [rad]"]
    for i in range(3):
        ax = axes[3 + i]
        ax.plot(timestamps, state_rpy[:, i], label="state", color="tab:blue", lw=1.0)
        ax.plot(timestamps, action_rpy[:, i], label="action", color="tab:red", lw=1.0, alpha=0.7)
        ax.set_ylabel(rpy_labels[i], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    # Row 6: speed
    axes[6].plot(timestamps, speed, color="tab:green", lw=1.0)
    axes[6].set_ylabel("EE speed [m/s]", fontsize=10)

    # Row 7: gripper
    axes[7].plot(timestamps, gripper, color="tab:orange", lw=1.0)
    axes[7].set_ylabel("Gripper aperture", fontsize=10)

    # Row 8: phase labels as coloured bars
    ax_phase = axes[8]
    boundaries = compute_boundaries(labels, timestamps)
    for seg in boundaries:
        color = PHASE_COLORS.get(seg["phase"], PHASE_COLORS["unknown"])
        ax_phase.axvspan(seg["start_time"], seg["end_time"],
                         alpha=0.7, color=color, label=seg["phase"])
    ax_phase.set_ylabel("Phase", fontsize=10)
    ax_phase.set_yticks([])
    # Deduplicated legend
    handles, lbls = ax_phase.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax_phase.legend(by_label.values(), by_label.keys(),
                    loc="upper right", fontsize=8, ncol=3)

    # Phase background on all subplots
    for ax in axes[:8]:
        for seg in boundaries:
            color = PHASE_COLORS.get(seg["phase"], PHASE_COLORS["unknown"])
            ax.axvspan(seg["start_time"], seg["end_time"],
                       alpha=0.08, color=color)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]", fontsize=11)
    fig.suptitle(
        f"{ee_name} Segmented Trajectory — Episode {episode_id}",
        fontsize=14,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Plot saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_summary(all_stats: list[dict], save_path: Path | None = None):
    """Bar chart of average phase durations across sampled episodes."""
    phase_durations = {p: [] for p in PHASE_ORDER}
    for stats in all_stats:
        for seg in stats["boundaries"]:
            p = seg["phase"]
            if p in phase_durations:
                dur = seg["end_time"] - seg["start_time"]
                phase_durations[p].append(dur)

    fig, ax = plt.subplots(figsize=(10, 5))
    phases_present = [p for p in PHASE_ORDER if phase_durations[p]]
    means = [np.mean(phase_durations[p]) for p in phases_present]
    stds = [np.std(phase_durations[p]) for p in phases_present]
    colors = [PHASE_COLORS[p] for p in phases_present]

    bars = ax.bar(phases_present, means, yerr=stds, color=colors,
                  edgecolor="black", capsize=5, alpha=0.85)
    ax.set_ylabel("Duration [s]", fontsize=11)
    ax.set_title(f"Average Phase Duration ({len(all_stats)} episodes)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.2f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Summary plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sample episodes, compute FK, and segment into task phases"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace repo ID or local dataset path")
    parser.add_argument("--urdf", type=str, required=True,
                        help="Path to RB-Y1 URDF file")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="Number of episodes to sample")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to segmentation config YAML (default: built-in)")
    parser.add_argument("--save-dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--local", action="store_true",
                        help="Treat --dataset as a local path")
    parser.add_argument("--ee-frame", type=str, default="ee_right",
                        help="End-effector frame for segmentation (default: ee_right)")
    args = parser.parse_args()

    # --- Load config ---
    if args.config:
        cfg = SegmentConfig.from_yaml(args.config)
        print(f"Loaded config: {args.config}")
    else:
        cfg = SegmentConfig()
        print("Using default segmentation config")

    # --- Load URDF ---
    print(f"Loading URDF: {args.urdf}")
    model, data = load_robot_model(args.urdf)
    name_to_q_idx = build_joint_index_map(model)

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset}")
    dataset_dir = resolve_dataset_path(args.dataset, local=args.local)
    info = load_info(dataset_dir)
    fps = info.get("fps", 30)
    print(f"  FPS: {fps}")

    # --- Sample episodes ---
    sampled_ids = sample_episode_ids(dataset_dir, args.n_samples)
    print(f"  Sampled {len(sampled_ids)} episodes: {sampled_ids}")

    episodes = load_episodes(dataset_dir, sampled_ids)
    print(f"  Loaded {len(episodes)} episodes")

    save_dir = Path(args.save_dir)
    ee_frames = ["ee_right", "ee_left"]
    all_stats = []

    # --- Process each episode ---
    for ep_id in sorted(episodes.keys()):
        print(f"\n{'='*60}")
        print(f"Episode {ep_id}")
        print(f"{'='*60}")
        df = episodes[ep_id]
        timestamps = extract_timestamps(df, fps)
        state_array = extract_joint_array(df, "observation.state")
        action_array = extract_joint_array(df, "action")
        print(f"  Frames: {len(timestamps)}, shape: {state_array.shape}")

        # Preprocess gripper signal
        gripper = preprocess_gripper(state_array, cfg)

        # FK for both end-effectors
        fk_results = {}
        for ee_name in ee_frames:
            print(f"  Computing FK for {ee_name}...")
            s_pos, s_rot, s_rpy = compute_fk_trajectory(
                model, data, name_to_q_idx, state_array, ee_name,
            )
            a_pos, a_rot, a_rpy = compute_fk_trajectory(
                model, data, name_to_q_idx, action_array, ee_name,
            )
            fk_results[ee_name] = {
                "state_pos": s_pos, "state_rot": s_rot, "state_rpy": s_rpy,
                "action_pos": a_pos, "action_rot": a_rot, "action_rpy": a_rpy,
            }

            # Save FK results
            save_fk_results(
                save_dir / "fk_results", ep_id, ee_name, timestamps,
                s_pos, s_rot, s_rpy, a_pos, a_rot, a_rpy,
            )

        # --- Segmentation (based on active ee_frame) ---
        fk = fk_results[args.ee_frame]
        speed = compute_ee_speed(fk["state_pos"], timestamps, cfg)

        labels, seg_info = label_phases(timestamps, fk["state_pos"], speed,
                                        gripper, cfg)
        boundaries = compute_boundaries(labels, timestamps)

        print(f"\n  Segmentation (anchor ee: {args.ee_frame}):")
        print(f"    T_grasp: {seg_info['t_grasp']}")
        print(f"    T_place: {seg_info['t_place']}")
        for seg in boundaries:
            dur = seg["end_time"] - seg["start_time"]
            print(f"    {seg['phase']:15s}  "
                  f"[{seg['start_idx']:5d} - {seg['end_idx']:5d}]  "
                  f"{seg['start_time']:.3f}s - {seg['end_time']:.3f}s  "
                  f"({dur:.3f}s)")

        ep_stats = {
            "episode_id": ep_id,
            "t_grasp": seg_info["t_grasp"],
            "t_place": seg_info["t_place"],
            "boundaries": boundaries,
        }
        all_stats.append(ep_stats)

        # --- Visualise each end-effector ---
        for ee_name in ee_frames:
            fk_ee = fk_results[ee_name]
            speed_ee = compute_ee_speed(fk_ee["state_pos"], timestamps, cfg)
            plot_path = (save_dir / "segment_plots"
                         / f"episode_{ep_id:06d}_{ee_name}_segmented.png")
            plot_segmented_episode(
                timestamps,
                fk_ee["state_pos"], fk_ee["state_rpy"],
                fk_ee["action_pos"], fk_ee["action_rpy"],
                speed_ee, gripper, labels,
                ee_name, ep_id, save_path=plot_path,
            )

    # --- Save segmentation results ---
    seg_results_path = save_dir / "segmentation_results.json"
    with open(seg_results_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSegmentation results saved: {seg_results_path}")

    # --- Summary across all episodes ---
    if all_stats:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        phase_durations = {p: [] for p in PHASE_ORDER}
        for stats in all_stats:
            for seg in stats["boundaries"]:
                p = seg["phase"]
                if p in phase_durations:
                    dur = seg["end_time"] - seg["start_time"]
                    phase_durations[p].append(dur)

        for p in PHASE_ORDER:
            durs = phase_durations[p]
            if durs:
                print(f"  {p:15s}  mean={np.mean(durs):.3f}s  "
                      f"std={np.std(durs):.3f}s  "
                      f"min={np.min(durs):.3f}s  max={np.max(durs):.3f}s  "
                      f"(n={len(durs)})")
            else:
                print(f"  {p:15s}  (not detected)")

        summary_path = save_dir / "segment_plots" / "phase_duration_summary.png"
        plot_summary(all_stats, save_path=summary_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
