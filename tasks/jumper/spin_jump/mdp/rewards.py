"""Rewards for the whirlwind jump.

Everything of `ref_free_jump`'s is imported, not copied -- one reader of the
prior, the height terms and the penalties. What this module adds is the turn,
in the same two halves the height objective has:

- `spin_angle` -- the latched |yaw turn| of the episode's own flight, in units
  of a full revolution, paid every step after the landing, with the same
  second-jump clawback `apex_height` has;
- `flight_spin` -- the dense half: the accumulated turn while the flight is
  its own, so a turning flight pays before the landing does.

Both pay the **absolute** turn, so a left turn and a right turn score alike
and the left-right mirror augmentation maps either onto the other. Linear and
uncapped, like the height: a double turn is worth two single ones, and nothing
in the reward says 360 degrees is the place to stop.
"""

from __future__ import annotations

import math

import torch

from ...ref_free_jump.mdp import rewards as _rfj


# Re-exported for `env_cfg.py`: one definition of each inherited term.
prior_joint_pos = _rfj.prior_joint_pos
prior_joint_vel = _rfj.prior_joint_vel
apex_height = _rfj.apex_height
flight_height = _rfj.flight_height
home_pose = _rfj.home_pose
upright = _rfj.upright
horizontal_drift = _rfj.horizontal_drift
illegal_contact = _rfj.illegal_contact
leg_collision = _rfj.leg_collision

#: The unit of turn the rewards are paid in: one full revolution.
SPIN_TURN = 2.0 * math.pi


def spin_angle(
    env, command_name: str = "jump", turn: float = SPIN_TURN
) -> torch.Tensor:
    """The episode's own latched turn -- how far the trunk yawed between its own
    takeoff and touchdown -- in units of `turn`, paid every step once it has
    landed. The apex term's shape exactly: same paying mask, same clawback on a
    second jump, so an episode that jumped twice is paid nothing for the turn
    either (see `ref_free_jump/mdp/rewards.py::apex_height` for the accounting)."""
    cmd = env.command_manager.get_term(command_name)
    spins = cmd.spin_total.abs() / turn
    paying = (cmd.landed & ~cmd.second_jump).float()
    return spins * (paying - cmd.clawback_pending.float() * cmd.paid_steps)


def flight_spin(
    env, command_name: str = "jump", turn: float = SPIN_TURN
) -> torch.Tensor:
    """The turn accumulated so far over the episode's own flight, in units of
    `turn` -- the spin term's dense half, as `flight_height` is the apex's."""
    cmd = env.command_manager.get_term(command_name)
    return cmd.spin_accum.abs() / turn * cmd.in_flight.float()


__all__ = [
    "apex_height",
    "flight_height",
    "flight_spin",
    "home_pose",
    "horizontal_drift",
    "illegal_contact",
    "leg_collision",
    "prior_joint_pos",
    "prior_joint_vel",
    "spin_angle",
    "upright",
]
