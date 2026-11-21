# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Loco Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import locolab.tasks.manager_based.locomotion.velocity.mdp as mdp
import my_locolab_project.tasks.locomotion.velocity.mdp as my_mdp


@configclass
class FlatCurriculumsCfg:
    """Curriculum terms for Go2 on flat terrain."""

    horizontal_force_magnitude = CurrTerm(
        func=my_mdp.horizontal_force_magnitude_levels,
        params={
            "event_term_name": "horizontal_force_pulse",
            # 每项为 (持续 iteration 数, 力范围 N)，超过总时长后保持最后一档。
            "stages": (
                (500, (10.0, 20.0)),
                (500, (20.0, 40.0)),
                (500, (40.0, 60.0)),
                (500, (60.0, 80.0)),
            ),
            # 与 runner.num_steps_per_env 保持一致；恢复训练时设置起始 iteration。
            "num_steps_per_iteration": 24,
            "start_iteration": 0,
        },
    )



@configclass
class RoughCurriculumsCfg:
    """Curriculum terms for rough terrain."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels, params={"log_by_terrain_type": False})
