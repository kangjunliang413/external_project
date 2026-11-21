from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from locolab.utils.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def horizontal_force_magnitude_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    stages: tuple[tuple[int, tuple[float, float]], ...],
    num_steps_per_iteration: int = 24,
    start_iteration: int = 0,
    event_term_name: str = "horizontal_force_pulse",
) -> dict[str, float]:
    """Update the force event's sampling range using iteration-based curriculum stages.

    Each stage specifies its duration in learning iterations and its force range in N.
    The curriculum manager calls this term on environment resets, including the initial reset.
    Existing pulses finish unchanged; new pulses use the updated range. The final range is
    retained after all stages. Set ``start_iteration`` when resuming training and match
    ``num_steps_per_iteration`` to the runner's rollout length.
    """
    if not isinstance(num_steps_per_iteration, int) or num_steps_per_iteration <= 0:
        raise ValueError("num_steps_per_iteration must be a positive integer.")
    if not isinstance(start_iteration, int) or start_iteration < 0:
        raise ValueError("start_iteration must be a non-negative integer.")
    if not stages:
        raise ValueError("Force curriculum must contain at least one stage.")
    for duration, force_range in stages:
        if not isinstance(duration, int) or duration <= 0:
            raise ValueError("Force curriculum stage durations must be positive integer iterations.")
        if not 0.0 <= force_range[0] <= force_range[1]:
            raise ValueError("Force curriculum ranges must satisfy 0 <= min <= max.")

    iteration = start_iteration + env.common_step_counter // num_steps_per_iteration
    stage_end = 0
    for stage_index, (duration, force_range) in enumerate(stages, start=1):
        stage_end += duration
        if iteration < stage_end:
            break

    event_cfg = env.event_manager.get_term_cfg(event_term_name)
    event_cfg.params["force_magnitude_range"] = tuple(force_range)
    return {"stage": stage_index, "force_min": force_range[0], "force_max": force_range[1]}