import argparse
import torch
from lib.models import PushPlanner
from helpers.utils import (
    load_data,
    split_data,
    prepare_dataloader,
    evaluate_planner,
    save_checkpoint,
    load_checkpoint,
    plot_training_curves,
    plot_predictions,
    plot_xy_trajectories
)
from helpers.config import load_config
from tqdm import tqdm
from colorama import init, Fore, Style
import numpy as np

# Initialize colorama
init()

def print_header(text: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def print_success(text: str):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")


def print_info(text: str):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")


def print_error(text: str):
    print(f"{Fore.RED}{text}{Style.RESET_ALL}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train push planning model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    device = config.get_device()
    print_info(f"Using device: {device}")

    # Load data
    print_header("Loading and Splitting Data")
    x_data, y_data = load_data(config)
    print_info(f"Full dataset — x: {x_data.shape}  y: {y_data.shape}")

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = split_data(
        x_data, y_data,
        test_ratio=config.data.get("test_ratio", 0.2),
        seed=config.data.get("seed", 42),
    )
    print_info(f"Train samples : {len(x_train)}")
    print_info(f"Test  samples : {len(x_test)}")

    # DataLoaders — train loader shuffled, test loader unshuffled
    train_loader = prepare_dataloader(x_train, y_train, config)
    test_loader = prepare_dataloader(x_test, y_test, config, shuffle=False)

    # Call Physics Push Planner
    planner = PushPlanner(config.model, config.physics_sampling, device)

    start_epoch = 0
    if args.checkpoint is not None:
        print_header(f"Resuming from checkpoint: {args.checkpoint}")
        start_epoch, _ = load_checkpoint(planner, args.checkpoint)
        print_info(f"Resumed from epoch {start_epoch}")

    print_header("Starting Training")
    eval_freq = config.training["eval_frequency"]
    num_epochs = config.training["num_epochs"]
    pbar = tqdm(range(num_epochs), desc="Training Progress")

    # Implement training loop
    for epoch in pbar:
        nn_loss, hybrid_loss, physics_loss = planner.train_epoch(train_loader)
        pbar.set_postfix(
            NN=f"{nn_loss:.5f}",
            Hybrid=f"{hybrid_loss:.5f}",
            Physics=f"{physics_loss:.5f}",
        )
        if epoch % eval_freq == 0:
            print_info(
                f"Epoch {epoch} | "
                f"Physics: {physics_loss:.6f} | "
                f"NN: {nn_loss:.6f} | "
                f"Hybrid: {hybrid_loss:.6f}"
            )

    print_success("\nTraining completed!")
    save_checkpoint(planner, num_epochs, planner.nn_losses[-1], config)

    # Plotting training curves
    print_header("Plotting Training Curves")
    plot_training_curves(planner.nn_losses, planner.hybrid_losses, planner.physics_losses)

    # Evaluate all three models using evaluate_planner
    print_header("Evaluation")
    metrics = evaluate_planner(planner, test_loader, device)

    print_info(f"\n{'Model':<10} {'MSE':>12} {'MED (m)':>14}")
    print_info("-" * 38)
    for name in ("physics", "nn", "hybrid"):
        mse = metrics[f"{name}_mse"]
        med = metrics[f"{name}_med"]
        print_info(f"{name.capitalize():<10} {mse:>12.6f} {med:>14.6f}")

    # Prediction plots on Test Set
    print_header("Generating Prediction Plots (Test Set)")
    planner.forward_model.eval()
    planner.hybrid_model.eval()

    n_eval = min(200, len(x_test))
    x_sample = torch.FloatTensor(x_test[:n_eval]).to(device)
    y_gt_np = y_test[:n_eval]

    with torch.no_grad():
        physics_pred = planner.physics.compute_motion(x_sample).cpu().detach().numpy()
        nn_pred = planner.forward_model(x_sample).cpu().detach().numpy()
        hybrid_pred = planner.hybrid_model(x_sample).cpu().detach().numpy()

    plot_predictions(y_gt_np, physics_pred, nn_pred, hybrid_pred)
    plot_xy_trajectories(y_gt_np, physics_pred, nn_pred, hybrid_pred)

    # Push planning on random test samples
    print_header("Push Planning Demo on Test Samples")
    n_plan = min(3, len(x_test))

    for i in range(n_plan):
        target_np = y_test[i]
        target = torch.FloatTensor(target_np).unsqueeze(0).to(device)
        print_info(f"\n--- Test sample {i} | Target: {target_np} ---")

        results = planner.plan_push(target)

        for model_name, optimal_push in results.items():
            with torch.no_grad():
                if model_name == "physics":
                    achieved = planner.physics.compute_motion(optimal_push).to(device)
                elif model_name == "nn":
                    achieved = planner.forward_model(optimal_push)
                else:
                    achieved = planner.hybrid_model(optimal_push)

                target_cpu = torch.FloatTensor(target_np).unsqueeze(0)
                achieved_cpu = achieved.cpu()

                mse = planner.hybrid_model.loss(achieved_cpu, target_cpu).item()
                med = planner.hybrid_model.accuracy(achieved_cpu, target_cpu)

            print_info(
                f"  [{model_name.upper():<7}] "
                f"push: {optimal_push.cpu().numpy()} | "
                f"achieved: {achieved.cpu().detach().numpy()} | "
                f"MSE: {mse:.6f} | MED: {med:.6f}"
            )

    print_success("\nAll done! Results saved to ./results/")

if __name__ == "__main__":
    main()