import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass
import torch


@dataclass
class Config:
    """Configuration container"""

    data: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    device: Dict[str, Any]
    physics_sampling: Dict[str, Any]  # Added physics sampling parameters

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

            # Ensure all required sections exist
            required_sections = [
                "data",
                "model",
                "training",
                "device",
                "physics_sampling",
            ]
            for section in required_sections:
                if section not in config_dict:
                    config_dict[section] = {}

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        return cls(**config_dict)

    def update(self, override_config: Dict[str, Any]) -> None:
        """Update configuration with override values"""
        for key, value in override_config.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, dict) and isinstance(value, dict):
                    current.update(value)
                else:
                    setattr(self, key, value)

    def get_device(self) -> torch.device:
        """Get torch device based on configuration"""
        if self.device.get("use_cuda", False) and torch.cuda.is_available():
            return torch.device(f"cuda:{self.device.get('cuda_device', 0)}")
        if torch.backends.mps.is_available(): # Added check for MPS (Apple Silicon)
            return torch.device("mps")
        return torch.device("cpu")


def load_config(
    config_path: str = None, override_config: Dict[str, Any] = None
) -> Config:
    """
    Load configuration from file with optional overrides

    Args:
        config_path: Path to YAML config file (default: config/default.yaml)
        override_config: Optional dictionary to override config values

    Returns:
        Config object
    """
    if config_path is None:
        config_path = os.path.join("config", "default.yaml")

    # Load base configuration
    config = Config.from_yaml(config_path)

    # Apply overrides if provided
    if override_config:
        config.update(override_config)

    return config
