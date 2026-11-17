"""jumper.gesture_catmeme_tilt -- the suspicious tilt, learned by imitation.

A one-shot gesture on the dance machinery (`tasks/jumper/common/dance/env.py`);
this directory holds the clip and the numbers. Hand-choreographed for the
cat meme meme skin: a left lean held, a right lean held, two quick paw taps, and a curious forward peer

The clip begins and ends at HOME with a half-second hold, so the controller's
switch-in ramp and the hand-back to walking both start from a stance the
walking policy knows.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_catmeme_tilt",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the hand-choreographed the suspicious tilt clip "
                "(committed in tasks/jumper/gesture_catmeme_tilt/media/)",
    tags=("imitation", "jumper", "gesture", "flat"),
)
