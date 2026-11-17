"""jumper.gesture_horse_prance -- the foal prance, learned by imitation.

A one-shot gesture on the dance machinery (`tasks/jumper/common/dance/env.py`);
this directory holds the clip and the numbers. Hand-choreographed for the
golden horse meme skin: two in-place bounces on extended legs, alternating front-hoof pawing, and a proud tall stance

The clip begins and ends at HOME with a half-second hold, so the controller's
switch-in ramp and the hand-back to walking both start from a stance the
walking policy knows.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_horse_prance",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the hand-choreographed the foal prance clip "
                "(committed in tasks/jumper/gesture_horse_prance/media/)",
    tags=("imitation", "jumper", "gesture", "flat"),
)
