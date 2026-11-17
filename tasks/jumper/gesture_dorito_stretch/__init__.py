"""jumper.gesture_dorito_stretch -- the dorito stretch, learned by imitation.

A one-shot gesture on the dance machinery (`tasks/jumper/common/dance/env.py`);
this directory holds the clip and the numbers. Hand-choreographed for the
dorito cat meme skin: front paws forward and down, a long arching hold, and a slow release

The clip begins and ends at HOME with a half-second hold, so the controller's
switch-in ramp and the hand-back to walking both start from a stance the
walking policy knows.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_dorito_stretch",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the hand-choreographed the dorito stretch clip "
                "(committed in tasks/jumper/gesture_dorito_stretch/media/)",
    tags=("imitation", "jumper", "gesture", "flat"),
)
