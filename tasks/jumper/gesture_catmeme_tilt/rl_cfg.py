"""Hyper-parameters for jumper.gesture_catmeme_tilt.

`jumper.gesture_hello`'s, value for value; the iteration cap sized for this
machine (an RTX 3050 trains one gesture at a time) -- the official gestures'
10,000 was written for the lab's GPUs.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config: `logs/<model>/jumper.gesture_catmeme_tilt/<date-time>`."""
    return jumper_ppo_baseline(
        experiment_name="jumper.gesture_catmeme_tilt",
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=2500,
    )
