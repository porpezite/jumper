"""jumper.gesture_capybara_soak -- hot-spring soak, learned by imitation.

A one-shot gesture on the dance machinery (`tasks/jumper/common/dance/env.py`);
this directory holds the clip and the numbers. Hand-choreographed for the
capybara meme skin: a slow sink into a soak, a lazy side-to-side sway, dangling paws, and a long contented hold

The clip begins and ends at HOME with a half-second hold, so the controller's
switch-in ramp and the hand-back to walking both start from a stance the
walking policy knows.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_capybara_soak",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the hand-choreographed hot-spring soak clip "
                "(committed in tasks/jumper/gesture_capybara_soak/media/)",
    tags=("imitation", "jumper", "gesture", "flat"),
)
