"""jumper.gesture_lobster_otaku -- slacking on the clock, learned by imitation.

A one-shot gesture on the dance machinery (`tasks/jumper/common/dance/env.py`);
this directory holds the clip and the numbers. Hand-choreographed for the
worker lobster meme skin: a weary slump, two tired sighs, one lazy fin-wave, and clocking out with a perk-up

The clip begins and ends at HOME with a half-second hold, so the controller's
switch-in ramp and the hand-back to walking both start from a stance the
walking policy knows.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_lobster_otaku",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the hand-choreographed slacking on the clock clip "
                "(committed in tasks/jumper/gesture_lobster_otaku/media/)",
    tags=("imitation", "jumper", "gesture", "flat"),
)
