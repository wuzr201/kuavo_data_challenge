#!/usr/bin/env python
"""
Evaluate a trained GR00T N1.5 policy on a LeRobot dataset (open-loop).

Computes MSE/MAE between predicted and ground-truth actions per trajectory,
and optionally saves trajectory comparison plots.

Example:
    python -m kuavo_deploy.eval_on_dataset --checkpoint_path your_checkpoint_path --dataset_path your_dataset_path --traj_ids 0 1 2 --save_plot_dir /tmp/eval_plots
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt

# Apply custom patches before other lerobot imports
import lerobot_patches.custom_patches  # noqa: F401

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.processor import PolicyProcessorPipeline

from kuavo_train.wrapper.policy.gr00t_n1d5.Gr00tN1d5PolicyWrapper import CustomGr00tN1d5PolicyWrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_delta_timestamps(dataset_metadata: LeRobotDatasetMetadata, policy_cfg: Any) -> dict[str, list[float]] | None:
    """Build delta timestamps for observations and actions from policy config."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None
    delta_timestamps = {}
    fps = dataset_metadata.fps
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / fps for i in act_indices]
    return delta_timestamps if delta_timestamps else None


def add_batch_dim(item: dict[str, Any]) -> dict[str, Any]:
    """Add batch dimension (0) to tensor values for a single sample."""
    out = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            if v.dim() == 0:
                out[k] = v.unsqueeze(0)
            elif v.dim() >= 1 and v.size(0) != 1:
                out[k] = v.unsqueeze(0)
            else:
                out[k] = v
        elif isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
            out[k] = torch.from_numpy(v).unsqueeze(0)
        elif isinstance(v, (list, tuple)) and k != "task" and len(v) > 0 and not isinstance(v[0], str):
            out[k] = torch.tensor(np.asarray(v)).unsqueeze(0)
        else:
            out[k] = v
    return out


def extract_gt_actions_for_range(
    dataset: LeRobotDataset,
    from_index: int,
    to_index: int,
    action_key: str = "action",
) -> np.ndarray:
    """Extract ground-truth action array for frame range [from_index, to_index)."""
    actions = []
    end = min(to_index, len(dataset))
    hf = getattr(dataset, "hf_dataset", None) or getattr(dataset, "_dataset", None)
    for idx in range(from_index, end):
        if hf is not None:
            item = hf[idx]
        else:
            item = dataset[idx]
        a = item[action_key]
        if hasattr(a, "numpy"):
            a = a.numpy()
        else:
            a = np.asarray(a)
        if a.ndim >= 2:
            a = a.squeeze(0)
        actions.append(a)
    if not actions:
        return np.array([]).reshape(0, 0)
    return np.stack(actions, axis=0)


def plot_trajectory_results(
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str | Path,
) -> None:
    """Plot and save trajectory comparison (GT vs predicted actions)."""
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]
    indices_to_plot = list(range(min(action_dim, 16)))  # at most 16 subplots
    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logger.warning("No valid indices to plot")
        return

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 2.5 * num_plots))
    if num_plots == 1:
        axes = [axes]
    fig.suptitle(
        f"Trajectory {traj_id} | Action: {', '.join(action_keys)}",
        fontsize=14,
        color="blue",
    )
    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")
        for j in range(0, actual_steps, action_horizon):
            marker = "ro" if j == 0 else "ro"
            label = "inference point" if j == 0 else None
            ax.plot(j, gt_action_across_time[j, action_idx], marker, label=label)
        ax.set_title(f"Dim {action_idx}")
        ax.legend()
    plt.tight_layout()
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)
    plt.close()


def _unnormalize_action_chunk(
    pred_chunk: np.ndarray,
    postprocessor: PolicyProcessorPipeline | None,
    device: torch.device,
) -> np.ndarray:
    """Unnormalize predicted action chunk so it is in same scale as dataset GT (raw)."""
    if postprocessor is None:
        return pred_chunk
    out_list = []
    for t in range(pred_chunk.shape[0]):
        action_t = torch.from_numpy(pred_chunk[t : t + 1].astype(np.float32)).to(device)
        batch = {"action": action_t}
        result = postprocessor(batch)
        a = result["action"] if isinstance(result, dict) else result
        a = a.cpu().numpy().squeeze(0)
        out_list.append(a)
    return np.stack(out_list, axis=0).astype(np.float32)


def evaluate_single_trajectory(
    policy: CustomGr00tN1d5PolicyWrapper,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline | None,
    dataset: LeRobotDataset,
    episodes_from: np.ndarray,
    episodes_to: np.ndarray,
    traj_id: int,
    steps: int = 300,
    action_horizon: int = 32,
    save_plot_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Run open-loop evaluation on one trajectory (episode).
    Returns (MSE, MAE) over the trajectory.
    """
    if traj_id >= len(episodes_from):
        raise ValueError(f"traj_id {traj_id} >= num_episodes {len(episodes_from)}")

    from_idx = int(episodes_from[traj_id])
    to_idx = int(episodes_to[traj_id])
    traj_length = to_idx - from_idx
    actual_steps = min(steps, traj_length)
    if actual_steps <= 0:
        logger.warning(f"Trajectory {traj_id} has length 0, skipping.")
        return float("nan"), float("nan")

    logger.info(
        "Trajectory %d: %d steps (requested %d, episode length %d)",
        traj_id,
        actual_steps,
        steps,
        traj_length,
    )

    policy.reset()
    pred_actions_list = []

    step = 0
    while step < actual_steps:
        dataset_index = from_idx + step
        item = dataset[dataset_index]
        batch = add_batch_dim(item)

        # Move to device if needed (preprocessor may do this)
        if device is not None:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=False)

        batch = preprocessor(batch)

        with torch.no_grad():
            pred_chunk = policy.predict_action_chunk(batch)
        # pred_chunk: (1, n_action_steps, action_dim) — in normalized space
        pred_chunk = pred_chunk.cpu().numpy()[0]
        pred_chunk = _unnormalize_action_chunk(pred_chunk, postprocessor, device)
        chunk_size = pred_chunk.shape[0]
        for j in range(chunk_size):
            if step + j < actual_steps:
                pred_actions_list.append(pred_chunk[j])
        # Advance by actual chunk length so pred count matches trajectory length
        step += chunk_size

    pred_action_across_time = np.array(pred_actions_list, dtype=np.float32)[:actual_steps]
    gt_action_across_time = extract_gt_actions_for_range(
        dataset,
        from_idx,
        from_idx + actual_steps,
    )

    # Align time steps and action dims (model may output more dims or fewer steps)
    gt_len, gt_dim = gt_action_across_time.shape[0], gt_action_across_time.shape[1]
    pred_len, pred_dim = pred_action_across_time.shape[0], pred_action_across_time.shape[1]
    if gt_len != pred_len or gt_dim != pred_dim:
        logger.warning(
            "Shape mismatch: gt %s vs pred %s; trimming to min length and action dims",
            gt_action_across_time.shape,
            pred_action_across_time.shape,
        )
    min_len = min(gt_len, pred_len)
    min_dim = min(gt_dim, pred_dim)
    gt_action_across_time = gt_action_across_time[:min_len, :min_dim]
    pred_action_across_time = pred_action_across_time[:min_len, :min_dim]

    mse = float(np.mean((gt_action_across_time - pred_action_across_time) ** 2))
    mae = float(np.mean(np.abs(gt_action_across_time - pred_action_across_time)))
    logger.info("Trajectory %d — Unnormalized MSE: %.6f, MAE: %.6f", traj_id, mse, mae)

    action_keys = list(dataset.meta.info["features"].get("action", {}).get("names", {}).get("action_names", ["action"]) or ["action"])
    if save_plot_path is not None:
        plot_trajectory_results(
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id,
            action_keys=action_keys,
            action_horizon=action_horizon,
            save_plot_path=save_plot_path,
        )

    return mse, mae


def main(
    checkpoint_path: str | Path,
    dataset_root: str = "~/zxh_data",
    dataset_repo_id: str = "0922-grab_bottle",
    dataset_path: str | None = None,
    traj_ids: list[int] | None = None,
    steps: int = 300,
    action_horizon: int = 32,
    save_plot_dir: str | Path | None = None,
    device: str | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent

    # LeRobot expects root = directory that directly contains meta/info.json and data/
    # (i.e. the dataset folder, e.g. .../0922-grab_bottle), not the parent.
    if dataset_path is not None:
        dataset_dir = Path(os.path.expanduser(dataset_path)).resolve()
        dataset_repo_id = dataset_dir.name
    else:
        dataset_root = Path(os.path.expanduser(dataset_root)).resolve()
        dataset_dir = dataset_root / dataset_repo_id
    if not (dataset_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Dataset meta not found at {dataset_dir / 'meta' / 'info.json'}. "
            "Pass the path to the dataset directory that contains meta/ and data/ (e.g. --dataset_path ~/zxh_data/0922-grab_bottle)."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Load policy from checkpoint (e.g. epochbest)
    logger.info("Loading policy from %s", checkpoint_path)
    policy = CustomGr00tN1d5PolicyWrapper.from_pretrained(checkpoint_path, strict=True)
    policy.to(dev)
    policy.eval()

    # Load preprocessor from run directory (same run that produced epochbest)
    preprocessor_path = run_dir / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor config not found at {preprocessor_path}. "
            "Ensure the run directory contains policy_preprocessor.json (e.g. use run_* as parent of epochbest)."
        )
    logger.info("Loading preprocessor from %s", run_dir)
    preprocessor = PolicyProcessorPipeline.from_pretrained(run_dir, config_filename="policy_preprocessor.json")
    postprocessor_path = run_dir / "policy_postprocessor.json"
    postprocessor = None
    if postprocessor_path.exists():
        postprocessor = PolicyProcessorPipeline.from_pretrained(run_dir, config_filename="policy_postprocessor.json")
        logger.info("Loading postprocessor from %s (pred will be unnormalized for plot/MSE)", run_dir)
    else:
        logger.warning("No policy_postprocessor.json found; pred is in normalized space, plot/MSE may look off vs raw GT.")

    # Use policy's chunk size so pred steps match trajectory length (avoid gt 160 vs pred 80)
    action_horizon = getattr(policy.config, "n_action_steps", None) or action_horizon
    logger.info("Using action_horizon (n_action_steps): %d", action_horizon)

    # Dataset metadata and delta_timestamps (root = dataset dir containing meta/ and data/)
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=str(dataset_dir))
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy.config)

    # Build dataset (same as training: delta_timestamps, no image_transforms for eval)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=str(dataset_dir),
        delta_timestamps=delta_timestamps,
        image_transforms=None,
    )
    episodes_from = np.asarray(dataset.meta.episodes["dataset_from_index"])
    episodes_to = np.asarray(dataset.meta.episodes["dataset_to_index"])

    if traj_ids is None:
        traj_ids = list(range(min(5, len(episodes_from))))
    logger.info("Dataset: %d episodes, %d frames. Evaluating trajectories: %s", len(episodes_from), len(dataset), traj_ids)

    all_mse, all_mae = [], []
    for tid in traj_ids:
        if tid >= len(episodes_from):
            logger.warning("Skipping trajectory id %d (>= %d)", tid, len(episodes_from))
            continue
        save_path = None
        if save_plot_dir is not None:
            save_path = Path(save_plot_dir) / f"traj_{tid}.png"
        mse, mae = evaluate_single_trajectory(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            episodes_from=episodes_from,
            episodes_to=episodes_to,
            traj_id=tid,
            steps=steps,
            action_horizon=action_horizon,
            save_plot_path=save_path,
            device=dev,
        )
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.nanmean(all_mse)
        avg_mae = np.nanmean(all_mae)
        logger.info("Average over evaluated trajectories — MSE: %.6f, MAE: %.6f", avg_mse, avg_mae)
    else:
        logger.info("No trajectories were evaluated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GR00T N1.5 policy on LeRobot dataset (open-loop).")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="your_checkpoint_path",
        help="Path to policy checkpoint directory (e.g. .../epochbest).",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Full path to dataset dir (contains meta/ and data/). Overrides dataset_root + dataset_repo_id.")
    parser.add_argument("--dataset_root", type=str, default="your_dataset_root", help="Parent dir for dataset (used with --dataset_repo_id if --dataset_path not set).")
    parser.add_argument("--dataset_repo_id", type=str, default="your_dataset_repo_id", help="Dataset folder name under dataset_root (ignored if --dataset_path set).")
    parser.add_argument("--traj_ids", type=int, nargs="*", default=None, help="Trajectory (episode) IDs to evaluate. Default: first 5.")
    parser.add_argument("--steps", type=int, default=300, help="Max steps per trajectory (capped by episode length).")
    parser.add_argument("--action_horizon", type=int, default=32, help="Action chunk size (must match policy config).")
    parser.add_argument("--save_plot_dir", type=str, default=None, help="Directory to save trajectory comparison plots.")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda, cpu). Default: cuda if available.")
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint_path,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        dataset_path=args.dataset_path,
        traj_ids=args.traj_ids,
        steps=args.steps,
        action_horizon=args.action_horizon,
        save_plot_dir=args.save_plot_dir,
        device=args.device,
    )
