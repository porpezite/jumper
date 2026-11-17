"""jumper.gesture_pickle_shiver -- the sour shiver, learned by imitation.

A one-shot gesture on the dance machinery (`tasks/jumper/common/dance/env.py`);
this directory holds the clip and the numbers. Hand-choreographed for the
sour pickle meme skin: a fast all-over shiver, a sour crouch, a second shudder, and recovery

The clip begins and ends at HOME with a half-second hold, so the controller's
switch-in ramp and the hand-back to walking both start from a stance the
walking policy knows.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_pickle_shiver",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the hand-choreographed the sour shiver clip "
                "(committed in tasks/jumper/gesture_pickle_shiver/media/)",
    tags=("imitation", "jumper", "gesture", "flat"),
)
