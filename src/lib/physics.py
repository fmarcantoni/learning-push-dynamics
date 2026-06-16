import torch
import numpy as np
from typing import Dict, Any

class PushPhysics:
    """Physics engine for push interactions"""

    def __init__(
        self, mass: float = 0.1, size: float = 0.1, inertia_factor: float = 1 / 12
    ):
        # Object properties
        self.mass = float(mass)
        self.size = float(size)
        self.inertia_factor = float(inertia_factor)
        self.inertia = self.inertia_factor * self.mass * (self.size**2)

        # Default simulation parameters
        self._push_duration = 3.0
        self._simulation_steps = 25

    @classmethod
    def from_config(cls, physics_config: Dict[str, Any]) -> "PushPhysics":
        """Create PushPhysics instance from config dictionary"""
        # Extract object properties from config and set simulation parameters
        mass = physics_config.get("mass", 0.1)
        size = physics_config.get("size", 0.1)
        inertia_factor = physics_config.get("inertia_factor", 1 / 12)
        push_duration = physics_config.get("push_duration", 3.0)
        simulation_steps = physics_config.get("simulation_steps", 25)

        instance = cls(mass=mass, size=size, inertia_factor=inertia_factor)
        instance._push_duration = push_duration
        instance._simulation_steps = simulation_steps

        return instance

    def compute_motion(
        self, push_params: torch.Tensor, duration: float = None, steps: int = None
    ) -> torch.Tensor:
        """
        Compute object motion given push parameters

        Args:
            push_params: [batch_size, 3] tensor of [rotation, side, distance]
            duration: Duration of push in seconds (optional)
            steps: Number of simulation steps (optional)

        Returns:
            [batch_size, 3] tensor of [x, y, theta] final states
        """
        # Define motion duration and steps
        T = duration if duration is not None else self._push_duration
        N = steps if steps is not None else self._simulation_steps
        dt = T / N

        # Extract push parameters (rotation, side, distance)
        theta0 = push_params[:, 0]   # initial push rotation/orientation
        d = push_params[:, 1]   # contact point distance
        D = push_params[:, 2]   # total push distance

        batch_size = push_params.shape[0]
        device = push_params.device
        dtype = push_params.dtype

        # Compute velocity profile
        v_max = 2 * D / T

        # Initialize states (x, y, theta)
        x = torch.zeros(batch_size, device=device, dtype=dtype)
        y = torch.zeros(batch_size, device=device, dtype=dtype)   
        theta = theta0.clone()

        # Loop through simulation steps to update states
        I = self.inertia
        m = self.mass

        for i in range(N):
            t_i = torch.tensor(i * dt, device=device, dtype=dtype)
            v_i = v_max * (0.5 * torch.sin(2 * np.pi * t_i / T - np.pi / 2) + 0.5)

            # Angular Motion
            tau = m * v_i * d
            alpha = tau / I
            d_theta = 0.5 * alpha * (dt ** 2)
            theta = theta + d_theta

            # Linear Motion
            dx = -v_i * torch.cos(theta) * dt
            dy = -v_i * torch.sin(theta) * dt
            x = x + dx
            y = y + dy

        # Transform local frame motion to global frame
        cos0, sin0 = torch.cos(theta0), torch.sin(theta0)
        x_global = cos0 * x - sin0 * y
        y_global = sin0 * x + cos0 * y

        return torch.stack([x_global, y_global, theta], dim=1)
