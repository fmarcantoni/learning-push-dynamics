import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
from .physics import PushPhysics


class NNModel(nn.Module):
    """Base neural network architecture"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
    
    # Implement forward function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    # Implement loss function
    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)
    
    # Implement accuracy function
    def accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            diff = predictions - targets
            dist = torch.sqrt((diff ** 2).sum(dim=-1))
            return dist.mean().item()


class NNPhysicsModel(NNModel):
    """Neural network with physics knowledge"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        physics: PushPhysics,
    ):
        super().__init__(input_dim + output_dim, output_dim, hidden_dims)
        self.physics = physics
        self.requires_grad = True

    # Implement forward function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            physics_pred = self.physics.compute_motion(x)

        combined = torch.cat([x, physics_pred], dim=1)
        return super().forward(combined)


class PushPlanner:
    """High-level push planning and training"""

    def __init__(
        self, model_config: Dict[str, Any], physics_sampling_config: Dict[str, Any], device
    ):
        self.model_config = model_config
        self.physics_sampling_config = physics_sampling_config
        self.device = device

        # Extract model and physics configurations
        network_cfg = model_config["network"]
        physics_cfg = model_config["physics"]
        optimizer_cfg = model_config["optimizer"]

        input_dim = network_cfg["input_dim"]
        output_dim = network_cfg["task_dim"]
        hidden_dims = network_cfg["hidden_dims"]
        lr = optimizer_cfg["learning_rate"]

        # Initialize models and move them to device

        # --- Physics model
        self.physics = PushPhysics.from_config(physics_cfg)

        # --- Neural network models
        self.forward_model = NNModel(
            input_dim, output_dim, hidden_dims
        ).to(self.device)
        
        self.hybrid_model = NNPhysicsModel(
            input_dim, output_dim, hidden_dims, self.physics
        ).to(self.device)

        # Setup optimizers
        self.forward_optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=lr
        )
        self.hybrid_optimizer = torch.optim.Adam(
            self.hybrid_model.parameters(), lr=lr
        )

        # Loss and accuracy history (tracked per epoch)
        self.nn_losses:      List[float] = []
        self.hybrid_losses:  List[float] = []
        self.physics_losses: List[float] = []
        self.nn_accuracy:      List[float] = []
        self.hybrid_accuracy:  List[float] = []
        self.physics_accuracy: List[float] = []

    # Implement train_epoch function
    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        """Train NN and Hybrid for one epoch; record physics as reference."""
        self.forward_model.train()
        self.hybrid_model.train()

        total_nn_loss      = 0.0
        total_hybrid_loss  = 0.0
        total_physics_loss = 0.0
        total_nn_acc       = 0.0
        total_hybrid_acc   = 0.0
        total_physics_acc  = 0.0
        total_samples      = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            bs = x_batch.size(0)

            # Physics — no gradient, reference only
            with torch.no_grad():
                physics_pred = self.physics.compute_motion(x_batch).to(self.device)
                physics_loss = self.forward_model.loss(physics_pred, y_batch)
                physics_acc  = self.forward_model.accuracy(physics_pred, y_batch)

            # NN model
            self.forward_optimizer.zero_grad()
            nn_pred  = self.forward_model(x_batch)
            nn_loss  = self.forward_model.loss(nn_pred, y_batch)
            nn_loss.backward()
            self.forward_optimizer.step()
            nn_acc = self.forward_model.accuracy(nn_pred, y_batch)

            # Hybrid model
            self.hybrid_optimizer.zero_grad()
            hybrid_pred = self.hybrid_model(x_batch)
            hybrid_loss = self.hybrid_model.loss(hybrid_pred, y_batch)
            hybrid_loss.backward()
            self.hybrid_optimizer.step()
            hybrid_acc = self.hybrid_model.accuracy(hybrid_pred, y_batch)

            total_nn_loss      += nn_loss.item()      * bs
            total_hybrid_loss  += hybrid_loss.item()  * bs
            total_physics_loss += physics_loss.item() * bs
            total_nn_acc       += nn_acc              * bs
            total_hybrid_acc   += hybrid_acc          * bs
            total_physics_acc  += physics_acc         * bs
            total_samples      += bs

        avg_nn_loss      = total_nn_loss      / total_samples
        avg_hybrid_loss  = total_hybrid_loss  / total_samples
        avg_phys_loss    = total_physics_loss / total_samples
        avg_nn_acc       = total_nn_acc       / total_samples
        avg_hybrid_acc   = total_hybrid_acc   / total_samples
        avg_phys_acc     = total_physics_acc  / total_samples

        self.nn_losses.append(avg_nn_loss)
        self.hybrid_losses.append(avg_hybrid_loss)
        self.physics_losses.append(avg_phys_loss)
        self.nn_accuracy.append(avg_nn_acc)
        self.hybrid_accuracy.append(avg_hybrid_acc)
        self.physics_accuracy.append(avg_phys_acc)

        return avg_nn_loss, avg_hybrid_loss, avg_phys_loss

    # Implement optimize_push function
    def optimize_push(
        self,
        target_state: torch.Tensor,
        model_name: str = None,
    ) -> dict:
        """
        Find push parameters that drive a model's prediction toward target_state.

        Args:
            target_state: [1, 3] desired final state (on self.device)
            model_name:   "hybrid" | "nn" | "physics" | None (runs all three)

        Returns:
            dict mapping model name -> optimised [1, 3] push tensor
        """
        if model_name is None:
            return {
                name: self.optimize_push(target_state, model_name=name)
                for name in ("physics", "nn", "hybrid")
            }

        cfg = self.physics_sampling_config

        if model_name == "nn":
            model = self.forward_model
            model.eval()
            use_physics_only = False
        elif model_name == "physics":
            use_physics_only = True
        else:
            model = self.hybrid_model
            model.eval()
            use_physics_only = False

        push = torch.zeros((1, 3), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([push], lr=cfg["learning_rate"])

        for _ in range(cfg["max_iterations"]):
            optimizer.zero_grad()
            if use_physics_only:
                pred = self.physics.compute_motion(push).to(self.device)
            else:
                pred = model(push)
            loss = nn.MSELoss()(pred, target_state)
            loss.backward()
            optimizer.step()

        return push.detach()

    # Implement plan_push function
    def plan_push(
        self, target_state: torch.Tensor, model_name: str = None
    ) -> dict:
        """Plan push parameters. If model_name is None, runs all three."""
        return self.optimize_push(target_state, model_name=model_name)


class PushNetFactory:
    """Factory for creating different types of push networks"""

    @staticmethod
    def create(config: Dict[str, Any]) -> nn.Module:
        network_config = config["network"]
        physics_config = config["physics"]
        model_type = network_config["type"]
        hidden_dims = network_config["hidden_dims"]

        if model_type == "NNModel":
            return NNModel(
                network_config["input_dim"], network_config["task_dim"], hidden_dims
            )
        elif model_type == "PhysicsModel":
            return PushPhysics.from_config(physics_config)
        else:
            physics = PushPhysics.from_config(physics_config)
            return NNPhysicsModel(
                network_config["input_dim"],
                network_config["task_dim"],
                hidden_dims,
                physics,
            )
