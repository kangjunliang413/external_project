from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg



# -------------------- Velocity Tracking Rewards -------------------- #

def track_lin_vel_xy_apply_force_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    force_offset_gain: float,
    force_event_name: str = "horizontal_force_pulse",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the body-frame velocity target shifted by the applied external force."""
    asset: RigidObject = env.scene[asset_cfg.name]
    applied_force_b = env.event_manager.get_term_cfg(force_event_name).func.applied_external_force_b
    velocity_target = env.command_manager.get_command(command_name)[:, :2] + force_offset_gain * applied_force_b[:, :2]
    lin_vel_error = torch.sum(torch.square(velocity_target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)