"""jumper.gesture_possum_patrol -- the hands-behind patrol, learned by imitation.

A one-shot gesture on the dance machinery (`tasks/jumper/common/dance/env.py`);
this directory holds the clip and the numbers. Hand-choreographed for the
hands-behind possum meme skin: paws folded behind the shell, a side-to-side surveying sway, and a tall finish

The clip begins and ends at HOME with a half-second hold, so the controller's
switch-in ramp and the hand-back to walking both start from a stance the
walking policy knows.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_possum_patrol",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the hand-choreographed the hands-behind patrol clip "
                "(committed in tasks/jumper/gesture_possum_patrol/media/)",
    tags=("imitation", "jumper", "gesture", "flat"),
)
