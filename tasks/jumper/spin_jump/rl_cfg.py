"""PPO hyperparameters for jumper.spin_jump.

`ref_free_jump`'s settings, which were tuned on this robot at this task's 200
Hz and action scale; only the experiment name and the horizon change. A turn
needs a little more episode than the plain rise -- the landing has a spinning
trunk to arrest -- so 3.0 s episodes; and the iteration count is sized for the
laptop GPU this task was first trained on (see `max_iterations` below).
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update. `ref_free_jump`'s, which corrected the
#: baseline's 24 for 200 Hz; a module constant because `env_cfg.py` needs it
#: for the prior's schedule, as there.
NUM_STEPS_PER_ENV = 48


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.spin_jump",
        # `ref_free_jump`'s: the actor carries five frames of history and the
        # actuator force, and this task's actor sees the same.
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=0.3,
        entropy_coef=0.001,
        gamma=0.99**0.25,
        lam=0.95**0.25,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # The prior is gone by PRIOR_END_ITERATIONS (1200, env_cfg.py, the
        # compressed window); the height and the turn are the objective on
        # their own from there. Sized for the laptop GPU this task was first
        # trained on -- a bigger machine wants ref_free_jump's 6000.
        max_iterations=3000,
    )
