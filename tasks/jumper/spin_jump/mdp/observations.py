"""Critic-only observations for the whirlwind jump.

`ref_free_jump`'s three terms are imported as they stand. What this adds is
one: the turn's bookkeeping, which the spin rewards are computed from and a
value function asked to predict them needs to see -- the same reason
`jump_state` exists there.

`(in_flight, landed, turn so far / 2pi)`: the turn is the accumulated one
while the flight is under way and the latched one after it, absolute in both
places. `mirror_kind="scalar"` is exact for it -- |−s| = |s| under the
left-right mirror -- as it is for every column of the vector.
"""

from __future__ import annotations

import math

import torch

from ...ref_free_jump.mdp import observations as _rfj

jump_state = _rfj.jump_state
prior_target = _rfj.prior_target
feedforward = _rfj.feedforward

_TURN = 2.0 * math.pi


def spin_state(
    env,
    command_name: str = "jump",
    turn: float = _TURN,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """(in_flight, landed, |turn| / `turn`): the flight bookkeeping's flags and
    the turn the spin rewards read, latched once landed.

    `mirror_kind` is read off the params by the symmetry augmentation, not here.
    """
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    spins = torch.where(
        cmd.in_flight, cmd.spin_accum.abs(), cmd.spin_total.abs()
    ) / turn
    return torch.stack(
        [cmd.in_flight.float(), cmd.landed.float(), spins], dim=1
    )


__all__ = ["feedforward", "jump_state", "prior_target", "spin_state"]
