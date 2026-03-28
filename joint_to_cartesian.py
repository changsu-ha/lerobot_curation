#!/usr/bin/env python3
"""
Convert LeRobot 2.0 dataset joint states/commands to Cartesian 6-DOF poses
using Pinocchio forward kinematics and RB-Y1 URDF, then plot with matplotlib.

Dataset observation.state / action layout (44 values):
  [0:6]   torso_0 ~ torso_5
  [6:13]  right_arm_0 ~ right_arm_6
  [13:20] left_arm_0 ~ left_arm_6
  [20:32] right_gripper (12 values)
  [32:44] left_gripper (12 values)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pinocchio as pin
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Joint mapping: dataset index → URDF joint name
# ---------------------------------------------------------------------------
DATASET_JOINT_NAMES = (
    [f"torso_{i}" for i in range(6)]
    + [f"right_arm_{i}" for i in range(7)]
    + [f"left_arm_{i}" for i in range(7)]
)
# Indices 0..19 in the 44-dim vector are the 20 joints used for FK.
# Indices 20..31 = right gripper (12), 32..43 = left gripper (12).
# Gripper prismatic joints in URDF (gripper_finger_r1/r2, l1/l2) are mapped
# from the first values of each gripper block.
GRIPPER_MAPPING = {
    "gripper_finger_r1": 20,
    "gripper_finger_r2": 21,
    "gripper_finger_l1": 32,
    "gripper_finger_l2": 33,
}


# ---------------------------------------------------------------------------
# URDF / Pinocchio helpers
# ---------------------------------------------------------------------------
def load_robot_model(urdf_path: str):
    """Load URDF and return pinocchio model + data."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def build_joint_index_map(model):
    """
    Build a mapping from URDF joint name → (joint_id, idx_in_q).

    Pinocchio's model.joints[i].idx_q gives the start index in the q vector
    for joint i. For 1-DOF joints this is a single value.
    """
    name_to_q_idx = {}
    for i in range(1, model.njoints):  # skip universe joint at 0
        jname = model.names[i]
        idx_q = model.joints[i].idx_q
        name_to_q_idx[jname] = idx_q
    return name_to_q_idx


def build_q_from_state(model, name_to_q_idx, joint_values):
    """
    Map the 44-dim dataset vector to Pinocchio's q vector.

    Parameters
    ----------
    model : pinocchio.Model
    name_to_q_idx : dict  (joint_name → index in q)
    joint_values : np.ndarray of shape (44,)

    Returns
    -------
    q : np.ndarray of shape (model.nq,)
    """
    q = pin.neutral(model)

    # Map the first 20 values (torso + arms)
    for ds_idx, jname in enumerate(DATASET_JOINT_NAMES):
        if jname in name_to_q_idx:
            q[name_to_q_idx[jname]] = joint_values[ds_idx]

    # Map gripper prismatic joints (first 2 of each 12-value block)
    for jname, ds_idx in GRIPPER_MAPPING.items():
        if jname in name_to_q_idx and ds_idx < len(joint_values):
            q[name_to_q_idx[jname]] = joint_values[ds_idx]

    return q


def compute_fk(model, data, name_to_q_idx, joint_values, frame_name):
    """
    Compute forward kinematics for a single timestep.

    Returns
    -------
    pos : np.ndarray (3,)
    rot_matrix : np.ndarray (3, 3)  — SO(3) rotation matrix
    rpy : np.ndarray (3,)  — roll, pitch, yaw
    """
    q = build_q_from_state(model, name_to_q_idx, joint_values)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    frame_id = model.getFrameId(frame_name)
    oMf = data.oMf[frame_id]

    pos = oMf.translation.copy()
    rot_matrix = oMf.rotation.copy()
    rpy = pin.rpy.matrixToRpy(rot_matrix)

    return pos, rot_matrix, rpy


def compute_fk_trajectory(model, data, name_to_q_idx, joint_array, frame_name):
    """
    Compute FK for an entire trajectory.

    Parameters
    ----------
    joint_array : np.ndarray of shape (N, 44)
    frame_name : str  ("ee_right" or "ee_left")

    Returns
    -------
    positions : np.ndarray (N, 3)
    rotations : np.ndarray (N, 3, 3)
    rpys : np.ndarray (N, 3)
    """
    n = len(joint_array)
    positions = np.empty((n, 3))
    rotations = np.empty((n, 3, 3))
    rpys = np.empty((n, 3))

    for i in range(n):
        pos, rot, rpy = compute_fk(
            model, data, name_to_q_idx, joint_array[i], frame_name
        )
        positions[i] = pos
        rotations[i] = rot
        rpys[i] = rpy

    return positions, rotations, rpys


# ---------------------------------------------------------------------------
# Dataset loading (LeRobot 2.0 format)
# ---------------------------------------------------------------------------
def resolve_dataset_path(dataset: str, local: bool = False) -> Path:
    """
    Resolve dataset to a local path.
    If *local* is True, treat *dataset* as a local directory.
    Otherwise download from HuggingFace Hub.
    """
    if local or Path(dataset).is_dir():
        return Path(dataset)
    # Download from HF Hub
    path = snapshot_download(repo_id=dataset, repo_type="dataset")
    return Path(path)


def load_info(dataset_dir: Path) -> dict:
    """Load meta/info.json."""
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def load_episodes(dataset_dir: Path, episode_ids: list[int] | None = None):
    """
    Load episode data from parquet chunk files.

    Returns
    -------
    dict[int, pd.DataFrame]  — episode_id → DataFrame with columns
        including 'observation.state', 'action', 'timestamp', etc.
    """
    data_dir = dataset_dir / "data"
    episodes = {}

    # Collect all parquet files across chunks
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    for pf in parquet_files:
        # Extract episode id from filename: episode_000003.parquet → 3
        ep_id = int(pf.stem.split("_")[-1])
        if episode_ids is not None and ep_id not in episode_ids:
            continue
        df = pd.read_parquet(pf)
        episodes[ep_id] = df

    return episodes


def extract_joint_array(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Extract joint values from a DataFrame column.

    Each row's value is a list/array of 44 floats.
    Returns np.ndarray of shape (N, 44).
    """
    values = df[column].tolist()
    return np.array(values, dtype=np.float64)


def extract_timestamps(df: pd.DataFrame, fps: float) -> np.ndarray:
    """
    Build a timestamp array. Use 'timestamp' column if present,
    otherwise generate from index and fps.
    """
    if "timestamp" in df.columns:
        return df["timestamp"].to_numpy(dtype=np.float64)
    return np.arange(len(df), dtype=np.float64) / fps


# ---------------------------------------------------------------------------
# Saving FK results
# ---------------------------------------------------------------------------
def save_fk_results(
    save_dir: Path,
    episode_id: int,
    ee_name: str,
    timestamps: np.ndarray,
    state_pos: np.ndarray,
    state_rot: np.ndarray,
    state_rpy: np.ndarray,
    action_pos: np.ndarray,
    action_rot: np.ndarray,
    action_rpy: np.ndarray,
):
    """Save FK results to a .npz file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"episode_{episode_id:06d}_{ee_name}_fk.npz"
    np.savez(
        fname,
        timestamps=timestamps,
        state_pos=state_pos,
        state_rotation_matrix=state_rot,
        state_rpy=state_rpy,
        action_pos=action_pos,
        action_rotation_matrix=action_rot,
        action_rpy=action_rpy,
    )
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cartesian_poses(
    timestamps: np.ndarray,
    state_poses_pos: np.ndarray,
    state_poses_rpy: np.ndarray,
    action_poses_pos: np.ndarray,
    action_poses_rpy: np.ndarray,
    ee_name: str,
    episode_id: int,
    save_path: Path | None = None,
):
    """
    Plot 6-DOF Cartesian poses (x, y, z, roll, pitch, yaw) over time.

    state and action are overlaid on the same subplots.
    """
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    pos_labels = ["x [m]", "y [m]", "z [m]"]
    rpy_labels = ["roll [rad]", "pitch [rad]", "yaw [rad]"]
    labels = pos_labels + rpy_labels

    state_data = np.hstack([state_poses_pos, state_poses_rpy])   # (N, 6)
    action_data = np.hstack([action_poses_pos, action_poses_rpy]) # (N, 6)

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(timestamps, state_data[:, i], label="state", color="tab:blue", linewidth=1.0)
        ax.plot(timestamps, action_data[:, i], label="action", color="tab:red", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]", fontsize=11)
    fig.suptitle(f"{ee_name} Cartesian Pose — Episode {episode_id}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot joint states/commands to Cartesian poses via FK"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. tony346/rby1_HF_Test) or local path",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to the RB-Y1 URDF file (e.g. models/rby1a/urdf/model.urdf)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode IDs to process (default: all)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output",
        help="Directory for saving plots and FK results (default: ./output)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Treat --dataset as a local directory path",
    )
    args = parser.parse_args()

    # --- Load URDF ---
    print(f"Loading URDF: {args.urdf}")
    model, data = load_robot_model(args.urdf)
    name_to_q_idx = build_joint_index_map(model)

    print(f"  Model joints ({model.njoints - 1}):")
    for jname in DATASET_JOINT_NAMES:
        status = "OK" if jname in name_to_q_idx else "MISSING"
        print(f"    {jname}: {status}")

    # Verify end-effector frames exist
    ee_frames = ["ee_right", "ee_left"]
    for frame in ee_frames:
        fid = model.getFrameId(frame)
        print(f"  Frame '{frame}': id={fid}")

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.dataset}")
    dataset_dir = resolve_dataset_path(args.dataset, local=args.local)
    info = load_info(dataset_dir)
    fps = info.get("fps", 30)
    print(f"  FPS: {fps}")
    print(f"  Codebase version: {info.get('codebase_version', 'unknown')}")

    episodes = load_episodes(dataset_dir, args.episodes)
    print(f"  Loaded {len(episodes)} episode(s): {sorted(episodes.keys())}")

    save_dir = Path(args.save_dir)

    # --- Process each episode ---
    for ep_id in sorted(episodes.keys()):
        print(f"\nProcessing episode {ep_id}...")
        df = episodes[ep_id]
        timestamps = extract_timestamps(df, fps)

        state_array = extract_joint_array(df, "observation.state")
        action_array = extract_joint_array(df, "action")
        print(f"  Frames: {len(timestamps)}, state shape: {state_array.shape}, action shape: {action_array.shape}")

        for ee_name in ee_frames:
            print(f"  Computing FK for {ee_name}...")

            s_pos, s_rot, s_rpy = compute_fk_trajectory(
                model, data, name_to_q_idx, state_array, ee_name
            )
            a_pos, a_rot, a_rpy = compute_fk_trajectory(
                model, data, name_to_q_idx, action_array, ee_name
            )

            # Save FK results (.npz)
            save_fk_results(
                save_dir / "fk_results",
                ep_id, ee_name, timestamps,
                s_pos, s_rot, s_rpy,
                a_pos, a_rot, a_rpy,
            )

            # Plot
            plot_path = save_dir / "plots" / f"episode_{ep_id:06d}_{ee_name}.png"
            plot_cartesian_poses(
                timestamps, s_pos, s_rpy, a_pos, a_rpy,
                ee_name, ep_id, save_path=plot_path,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
