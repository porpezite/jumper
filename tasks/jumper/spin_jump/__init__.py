"""jumper.spin_jump -- the whirlwind jump: a rise, a full mid-air turn, a landing.

Built on `jumper.ref_free_jump` and inheriting its whole scheme: the recording
is fed forward into the action and rewarded as a prior on the joints while
training, both fading to nothing, and the deployed policy needs no recording,
no clock and nothing a robot cannot measure -- it jumps when it starts to run.

What this task adds is the turn. The objective cross-fades in beside the
height: the episode's own flight is bookkept the same way (`mdp/commands.py`
of `ref_free_jump`), and on top of the latched apex the command latches how far
the trunk yawed between its own takeoff and touchdown. The spin rewards pay
the absolute turn -- either direction, so the left-right mirror augmentation
still holds -- in units of a full revolution, linear and uncapped: two turns
are worth two. `horizontal_drift` stays, so the turn is bought in the air and
not by travelling, and `fell_over` stays at the controller's own 50 degrees,
which a yaw about the vertical never approaches.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.spin_jump",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod whirlwind jump: rise, full mid-air yaw turn, landing "
        "(high_jump_flat.npz only while training: a fading feedforward and joint prior)"
    ),
    tags=("locomotion", "jumper", "jump", "spin", "action-prior"),
)
