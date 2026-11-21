from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

from locolab.utils.terrains.terrain_utils import get_terrain_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HorizontalForcePulse(ManagerTermBase):
    """Apply a random horizontal force pulse to the robot base."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[self._asset_cfg.name]

        if not isinstance(self._asset_cfg.body_ids, list) or len(self._asset_cfg.body_ids) != 1:
            raise ValueError("HorizontalForcePulse requires exactly one resolved base body.")

        self._num_bodies = len(self._asset_cfg.body_ids)
        # Per-environment applied force in the selected base link's local frame.
        self.applied_external_force_b = torch.zeros(env.num_envs, 3, device=env.device)
        self._pulse_time_left = torch.zeros(env.num_envs, device=env.device)
        self._wait_time_left = torch.zeros(env.num_envs, device=env.device)

        # Generate a random wait time.
        self._sample_wait(torch.arange(env.num_envs, device=env.device), cfg.params["wait_range_s"])

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._pulse_time_left[env_ids] = 0.0
        self._sample_wait(env_ids, self.cfg.params["wait_range_s"])
        self._clear_force(env_ids, self.cfg.params["is_global"])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        force_magnitude_range: tuple[float, float],
        pulse_duration_s: float | tuple[float, float],
        wait_range_s: tuple[float, float],
        is_global: bool = False,
    ):
        """Tick once per control step, sampling each new pulse's duration independently.

        ``pulse_duration_s`` accepts a fixed duration or a uniform ``(min, max)`` range in seconds.
        """
        del asset_cfg  # The resolved config is cached in __init__.

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        dt = env.step_dt

        # Continue active pulses. Clear force immediately after the requested duration.
        active_mask = self._pulse_time_left[env_ids] > 0.0
        active_ids = env_ids[active_mask]
        if active_ids.numel() > 0:
            self._pulse_time_left[active_ids] -= dt
            ended_ids = active_ids[self._pulse_time_left[active_ids] <= 0.0]
            if ended_ids.numel() > 0:
                self._clear_force(ended_ids, is_global)
                self._sample_wait(ended_ids, wait_range_s)

        # Count down environments that are waiting for their next push.
        waiting_ids = env_ids[~active_mask]
        if waiting_ids.numel() > 0:
            self._wait_time_left[waiting_ids] -= dt
            start_ids = waiting_ids[self._wait_time_left[waiting_ids] <= 0.0]
            if start_ids.numel() > 0:
                self._sample_pulse_duration(start_ids, pulse_duration_s)
                self._apply_random_horizontal_force(start_ids, force_magnitude_range, is_global)

    def _sample_pulse_duration(self, env_ids: torch.Tensor, pulse_duration_s: float | tuple[float, float]):
        if isinstance(pulse_duration_s, (int, float)):
            low = high = pulse_duration_s
        else:
            low, high = pulse_duration_s
        if not 0.0 < low <= high:
            raise ValueError("pulse_duration_s must satisfy 0 < min <= max (in seconds).")

        if low == high:
            self._pulse_time_left[env_ids] = low
        else:
            self._pulse_time_left[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)

    def _sample_wait(self, env_ids: torch.Tensor, wait_range_s: tuple[float, float]):
        low, high = wait_range_s
        self._wait_time_left[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)

    def _apply_random_horizontal_force(
        self,
        env_ids: torch.Tensor,
        force_magnitude_range: tuple[float, float],
        is_global: bool,
    ):
        magnitude = torch.empty(len(env_ids), 1, device=self.device).uniform_(*force_magnitude_range)
        heading = torch.empty(len(env_ids), 1, device=self.device).uniform_(-torch.pi, torch.pi)

        forces = torch.zeros(len(env_ids), self._num_bodies, 3, device=self.device)
        forces[:, :, 0] = magnitude * torch.cos(heading)
        forces[:, :, 1] = magnitude * torch.sin(heading)
        # forces[:, :, 2] remains exactly zero

        torques = torch.zeros_like(forces)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=self._asset_cfg.body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )
        force_b = forces[:, 0, :]
        if is_global:
            body_quat_w = self._asset.data.body_link_quat_w[env_ids, self._asset_cfg.body_ids[0]]
            force_b = quat_apply_inverse(body_quat_w, force_b)
        self.applied_external_force_b[env_ids] = force_b

    def _clear_force(self, env_ids: torch.Tensor, is_global: bool):
        zeros = torch.zeros(len(env_ids), self._num_bodies, 3, device=self.device)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            body_ids=self._asset_cfg.body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )
        self.applied_external_force_b[env_ids] = 0.0
