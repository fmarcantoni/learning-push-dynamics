import os
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def load_data(config) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from files based on configuration"""
    base_path = config.data["base_path"]
    x_path = os.path.join(base_path, config.data["train_x"])
    y_path = os.path.join(base_path, config.data["train_y"])

    x_data = np.load(x_path)
    y_data = np.load(y_path)
    return x_data, y_data

def split_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets with a fixed random seed."""
    rng = np.random.default_rng(seed)
    n = len(x_data)
    indices = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return x_data[train_idx], x_data[test_idx], y_data[train_idx], y_data[test_idx]


def prepare_dataloader(
    x_data: np.ndarray, y_data: np.ndarray, config, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """Prepare DataLoader from numpy arrays using configuration
    shuffle can be True, False, or from config (use config.data["shuffle"])
    """
    x_tensor = torch.FloatTensor(x_data)
    y_tensor = torch.FloatTensor(y_data)
    dataset = TensorDataset(x_tensor, y_tensor)
    if shuffle is False:
        dataloader = DataLoader(
            dataset, batch_size=config.data["batch_size"], shuffle=False
        )
    elif shuffle is True:
        dataloader = DataLoader(
            dataset, batch_size=config.data["batch_size"], shuffle=True
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=config.data["batch_size"], shuffle=config.data["shuffle"]
        )

    return dataloader


def evaluate_planner(
    planner,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate all three models on the given dataloader.
    Returns dict with keys:
        physics_mse, nn_mse, hybrid_mse,
        physics_med, nn_med, hybrid_med
    """
    planner.forward_model.eval()
    planner.hybrid_model.eval()

    total = {
        "physics_mse": 0.0, "nn_mse": 0.0, "hybrid_mse": 0.0,
        "physics_med": 0.0, "nn_med": 0.0, "hybrid_med": 0.0,
    }
    n_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            bs = x_batch.size(0)

            physics_pred = planner.physics.compute_motion(x_batch).to(device)
            nn_pred = planner.forward_model(x_batch)
            hybrid_pred = planner.hybrid_model(x_batch)

            for name, pred in [("physics", physics_pred),
                                ("nn", nn_pred),
                                ("hybrid", hybrid_pred)]:
                # MSE loss
                total[f"{name}_mse"] += planner.forward_model.loss(
                    pred, y_batch
                ).item() * bs
                # MED accuracy
                total[f"{name}_med"] += planner.forward_model.accuracy(
                    pred, y_batch
                ) * bs

            n_samples += bs

    return {k: v / n_samples for k, v in total.items()}

def save_checkpoint(
    planner, epoch: int, forward_loss: float, config
) -> None:
    """Save model checkpoint"""
    os.makedirs(config.training["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(
        config.training["checkpoint_dir"], f"model_epoch_{epoch}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "forward_model_state_dict": planner.forward_model.state_dict(),
            "forward_optimizer_state_dict": planner.forward_optimizer.state_dict(),
            "forward_loss": forward_loss,
        },
        checkpoint_path,
    )

def load_checkpoint(planner, checkpoint_path: str) -> Tuple[int, float, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    planner.forward_model.load_state_dict(checkpoint["forward_model_state_dict"])
    planner.forward_optimizer.load_state_dict(
        checkpoint["forward_optimizer_state_dict"]
    )
    return checkpoint["epoch"], checkpoint["forward_loss"], checkpoint["backward_loss"]

def plot_training_curves(
    nn_losses, hybrid_losses, physics_losses,
    save_path="results/training_curves.png",
):
    """"
    Plot training loss curves for NN, Hybrid, and Physics models.
    Saves the plot to the specified path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(nn_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_physics = float(np.mean(physics_losses))
    avg_nn = float(np.mean(nn_losses))
    avg_hybrid = float(np.mean(hybrid_losses))
    ax.plot(epochs, nn_losses, label=f"NN Model (avg MSE = {avg_nn:.4f})",     color="steelblue",  linewidth=2)
    ax.plot(epochs, hybrid_losses, label=f"Hybrid Model (avg MSE = {avg_hybrid:.4f})", color="darkorange", linewidth=2)
    ax.axhline(
        y=avg_physics, color="seagreen", linestyle="--", linewidth=2,
        label=f"Physics Model (avg MSE = {avg_physics:.4f})",
    )
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MSE Loss", fontsize=13)
    ax.set_title("Training Loss Curves", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_predictions(
    y_gt, physics_pred, nn_pred, hybrid_pred,
    n_samples=50, save_path="results/predictions.png",
):
    """"
    Plot predicted vs ground truth values for x, y, and θ dimensions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = ["x (m)", "y (m)", "θ (rad)"]
    models = {
        "Physics": (physics_pred, "seagreen"),
        "NN": (nn_pred, "steelblue"),
        "Hybrid":  (hybrid_pred, "darkorange"),
    }
    n = min(n_samples, len(y_gt))
    idx = np.arange(n)
    gt = y_gt[:n]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Predictions vs Ground Truth (Test Set)",
                 fontsize=16, fontweight="bold")
    for col, (model_name, (pred, color)) in enumerate(models.items()):
        p = pred[:n]
        for row, dim_label in enumerate(labels):
            ax = axes[row][col]
            ax.scatter(idx, gt[:, row], label="Ground Truth",
                       color="black", s=20, alpha=0.7, zorder=3)
            ax.scatter(idx, p[:, row],  label=model_name,
                       color=color,   s=20, alpha=0.7, zorder=2)
            ax.set_title(f"{model_name} – {dim_label}", fontsize=11)
            ax.set_xlabel("Sample index")
            ax.set_ylabel(dim_label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_xy_trajectories(
    y_gt, physics_pred, nn_pred, hybrid_pred,
    n_samples=100, save_path="results/xy_trajectories.png",
):
    """"
    Plot XY trajectories of predicted vs ground truth values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n  = min(n_samples, len(y_gt))
    gt = y_gt[:n]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("XY Displacement: Predicted vs Ground Truth (Test Set)",
                 fontsize=14, fontweight="bold")
    data = [
        ("Physics", physics_pred[:n], "seagreen"),
        ("NN",      nn_pred[:n],      "steelblue"),
        ("Hybrid",  hybrid_pred[:n],  "darkorange"),
    ]
    for ax, (name, pred, color) in zip(axes, data):
        ax.scatter(gt[:, 0], gt[:, 1], c="black", s=20,
                   alpha=0.6, label="Ground Truth", zorder=3)
        ax.scatter(pred[:, 0], pred[:, 1], c=color, s=20,
                   alpha=0.6, label=name, zorder=2)
        for i in range(n):
            ax.plot(
                [gt[i, 0], pred[i, 0]],
                [gt[i, 1], pred[i, 1]],
                c=color, alpha=0.2, linewidth=0.7,
            )
        ax.set_title(f"{name} Model", fontsize=12)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")