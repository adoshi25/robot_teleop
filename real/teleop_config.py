"""Teleop script configuration (loaded from teleop.yaml)."""

from dataclasses import dataclass


@dataclass
class TeleopConfig:
    """Teleop script configuration (loaded from teleop.yaml)."""
    max_horizon: int = 100
