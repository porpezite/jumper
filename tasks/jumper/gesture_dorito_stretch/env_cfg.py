"""Environment config for jumper.gesture_dorito_stretch: the the dorito stretch clip.

`jumper.gesture_hello`'s environment, with this clip's measured numbers: at
the 50 Hz control rate the clip is 14.7 s, per-joint position RMS
amplitude 0.056 rad and joint velocity RMS 0.131 rad/s, so the
tracking stds follow jumper.dance's rules (pos 0.44 x RMS, vel 2/3 x RMS).
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

MEDIA = Path(__file__).resolve().parent / "media"

#: Half the 14.7 s clip, jumper.dance_maze's rule for a short clip.
EPISODE_S = 7.35


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    return dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        joint_pos_std=0.025,
        joint_pos_weight=1.0,
        joint_vel_std=0.088,
        joint_vel_weight=0.5,
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        power_weight=-0.02,
        torque_headroom_weight=-2.0,
        joint_limit_weight=-10.0,
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        anchor_height_error=0.04,
        anchor_tilt_error=0.3,
        support_foot_error=0.05,
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
