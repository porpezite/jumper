"""The go command for the whirlwind jump: `ref_free_jump`'s flight bookkeeping
plus the trunk's yaw turn over that flight.

Everything about whose flight counts -- the push gate, the `min_rise` a jump is
known by, the one-jump rule and its clawback -- is `ref_free_jump`'s and is
inherited untouched. What this adds is one number per episode: how far the
trunk yawed between the takeoff of its own flight and the touchdown that ended
it.

**Yaw is read off the root quaternion and unwrapped by difference.** Each step
takes the yaw, wraps the change since the last step into [-pi, pi] and adds it
to a per-episode accumulator while the flight is its own. The wrap is what
makes the turn countable past 180 degrees: the quaternion (and the yaw angle
itself) repeat every revolution, and only the step-to-step difference is
faithful across the seam. A turn and its mirror pay the same -- the latched
value is the absolute one -- so the left-right symmetry augmentation of the
walking tasks keeps holding for everything this task rewards.

**Only in `mdp/commands.py` and only for the critic.** The yaw accumulator is
bookkeeping a simulator reads off the state; the robot has a gyro whose z axis
is the same yaw rate, but the deployed policy is deliberately given none of
this -- like `ref_free_jump`, the actor observes its body and history alone,
jumps when it starts to run and turns because it was trained to.
"""

from __future__ import annotations

import torch

from ...ref_free_jump.mdp.commands import (
    RefFreeJumpCommand,
    RefFreeJumpCommandCfg,
)

_TWO_PI = 2.0 * torch.pi


def _yaw(quat_w: torch.Tensor) -> torch.Tensor:
    """Trunk yaw from the root quaternion (w, x, y, z), in [-pi, pi].

    The same formula `five_foot/tools/twist_check.py` measures with: half the
    angle of the rotation about z, recovered from the quaternion's off-diagonal
    terms, exact for the yaw-pitch-roll order MuJoCo reports.
    """
    w, x, y, z = quat_w.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SpinJumpCommandCfg(RefFreeJumpCommandCfg):
    """`ref_free_jump`'s command, nothing added in the config: the yaw
    bookkeeping is state the command keeps, not a setting it takes."""

    def __post_init__(self):
        self.class_type = SpinJumpCommand

    def build(self, env) -> "SpinJumpCommand":
        return SpinJumpCommand(self, env)


class SpinJumpCommand(RefFreeJumpCommand):
    cfg: SpinJumpCommandCfg

    def __init__(self, cfg: SpinJumpCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        #: Signed yaw change accumulated over the episode's own flight, radians.
        self.spin_accum = torch.zeros(n, device=self.device)
        #: |yaw change| latched at the touchdown of the episode's own jump.
        self.spin_total = torch.zeros(n, device=self.device)
        #: Last step's yaw, for wrapping the difference.
        self._prev_yaw = _yaw(self.robot.data.root_link_quat_w)
        self._spin_deg = torch.zeros(n, device=self.device)
        self.metrics["spin_deg"] = self._spin_deg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.spin_accum[env_ids] = 0.0
        self.spin_total[env_ids] = 0.0
        # The spawn state was just written (events run before commands
        # resample, as `ref_free_jump`'s RSI handoff relies on): start the
        # differencing from where the spawn actually left the trunk.
        self._prev_yaw[env_ids] = _yaw(self.robot.data.root_link_quat_w[env_ids])

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._spin_deg[:] = torch.rad2deg(self.spin_total) * self.landed.float()
        self.metrics["spin_deg"] = self._spin_deg

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # `RefFreeJumpCommand`'s step path is inlined below, threaded with the
        # yaw bookkeeping, so the super call skips it to `JumpMotionCommand`'s
        # (the go clock) -- calling it directly would run the flight
        # bookkeeping twice and count every takeoff twice over.
        super(RefFreeJumpCommand, self)._update_command(env_ids)
        if env_ids is not None:
            # Reset path: the spawn state was just written and nothing has
            # flown; the spawn yaw is picked up in `_resample_command`.
            return

        # The parent's step path, inlined order-for-order, with the yaw
        # bookkeeping threaded through at the points it belongs to. The step's
        # rewards were computed before this update, on the state the last one
        # left: count the apex payment they made, and the clawback, if it was
        # due, is paid now.
        self.paid_steps += (self.landed & ~self.second_jump).float()
        self.clawback_pending.zero_()

        forces = self._env.scene.sensors[self.cfg.contact_sensor_name].data.force
        assert forces is not None
        airborne = (forces.norm(dim=-1) <= self.cfg.force_threshold).all(dim=1)
        rise = self.lowest_rise()

        takeoff = (
            self.can_jump & ~self.in_flight & ~self.landed
            & (self.time_since_go >= 0.0) & airborne
        )
        self.in_flight |= takeoff
        self.apex_rise.copy_(
            torch.where(self.in_flight, torch.maximum(self.apex_rise, rise), self.apex_rise)
        )

        # ── The turn ────────────────────────────────────────────────────────
        # Wrap the yaw change into [-pi, pi] and add it while the flight is the
        # episode's own. Fresh takeoffs are zeroed after the add, so nothing
        # grounded counts and a hop's stale turn dies with it.
        yaw_now = _yaw(self.robot.data.root_link_quat_w)
        d_yaw = (yaw_now - self._prev_yaw + torch.pi) % _TWO_PI - torch.pi
        self.spin_accum += d_yaw * self.in_flight.float()
        self.spin_accum.masked_fill_(takeoff, 0.0)
        self._prev_yaw.copy_(yaw_now)
        # ─────────────────────────────────────────────────────────────────────

        touchdown = self.in_flight & ~airborne
        jumped = touchdown & (self.apex_rise > self.cfg.min_rise)
        self.spin_total.copy_(
            torch.where(jumped, self.spin_accum.abs(), self.spin_total)
        )
        self.in_flight &= ~touchdown
        self.landed |= jumped
        self.can_jump &= ~jumped
        # A hop that never lifted the feet gives the jump back, and its turn.
        self.apex_rise.masked_fill_(touchdown & ~jumped, 0.0)
        self.spin_accum.masked_fill_(touchdown & ~jumped, 0.0)

        # Up again after the jump, by the jump's own gate: the next reward takes
        # back everything the apex has paid, and none follows.
        again = self.landed & ~self.second_jump & airborne & (rise > self.cfg.min_rise)
        self.second_jump |= again
        self.clawback_pending |= again

        q = self.robot.data.joint_pos[:, self._gait_ids]
        home = self.robot.data.default_joint_pos[:, self._gait_ids]
        qd = self.robot.data.joint_vel[:, self._gait_ids]
        back = (
            self.jump_done
            & (((q - home) ** 2).mean(dim=1).sqrt() < self.cfg.home_tolerance)
            & ((qd**2).mean(dim=1) < self.cfg.home_speed_sq)
        )
        self.home_steps.copy_(torch.where(back, self.home_steps + 1.0, torch.zeros_like(self.home_steps)))


__all__ = ["SpinJumpCommand", "SpinJumpCommandCfg"]
