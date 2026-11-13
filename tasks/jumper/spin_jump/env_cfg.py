"""jumper.spin_jump -- the whirlwind jump: rise, full mid-air turn, landing.

`jumper.ref_free_jump`'s environment, kept whole -- the 200 Hz control, kp=20,
the recording fed forward and fading, RSI spawns, the home return, every
penalty -- with one objective added beside the height: the turn. The command
(`mdp/commands.py`) latches how far the trunk yawed over the episode's own
flight; `spin_angle` pays it after the landing and `flight_spin` during it, in
units of a full revolution, both cross-fading in on the prior's own schedule
beside the two height terms.

**On the weights.** The prior's schedule, the physics and every inherited
weight are `ref_free_jump`'s measured or reasoned numbers; the spin pair's
(weights 4.0 and 2.0, one revolution as the unit) are reasoned by analogy with
the height pair the objective joins -- the apex pays 4 x rise/0.10 m per step
and the turn 4 x revolutions, so a recording-height jump (0.12 m) with a full
turn pays them evenly, and neither is worth giving the other up for. If
training buys height and not turns, or turns and no rise, the pair to move is
these two, and only these two.

**The episode runs 3.0 s**, `ref_free_jump`'s 2.5 plus room for a spinning
trunk to be arrested on the landing before the way home. `horizontal_drift`
stays at -1.0: the turn is bought in the air, and a spin that travels is not a
spin.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..common.constants import (
    ACTUATOR_FORCE_NOISE,
    EFFORT_LIMIT,
    FOOT_GEOMS,
    GAIT_JOINTS,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    PUSH_VELOCITY_RANGE,
    get_jumper_robot_cfg,
)
from ..common.mdp.observations import actuator_force
from ..common.tof import tof_sensor
from ..jump import mdp as jump_mdp
from ..posture.mdp.history import StridedHistory
from . import mdp as spin_mdp
from .mdp.rewards import SPIN_TURN
from .rl_cfg import NUM_STEPS_PER_ENV

# ── Physics: jumper.jump's, for the prior's sake (as ref_free_jump keeps them) ──
_SIM_TIMESTEP = 0.0025
_DECIMATION = 2

#: The gain the recording was made at, and its damping; `ref_free_jump`'s.
_KP = 20.0

# ── The action ─────────────────────────────────────────────────────────────────
#: Absolute targets around HOME, `ref_free_jump`'s scale.
ACTION_SCALE = 0.5

# ── The observation's history ──────────────────────────────────────────────────
#: `ref_free_jump`'s: five frames 20 ms apart at 200 Hz.
OBS_HISTORY = 5
OBS_HISTORY_STRIDE = 4

HISTORY_TERMS = frozenset({"joint_pos", "joint_vel", "actions", "actuator_force"})

# ── The go ─────────────────────────────────────────────────────────────────────
#: The policy jumps when it starts to run; `ref_free_jump`'s finding.
GO_DELAY_RANGE = (0.0, 0.0)

RSI_FRACTION = 0.5
RSI_PHASE_RANGE = (0.0, 0.9)

# ── The prior ──────────────────────────────────────────────────────────────────
#: `ref_free_jump`'s window, weights and kernel widths.
PRIOR_POS_WEIGHT = 3.0
PRIOR_POS_SIGMA_SQ = 0.20
PRIOR_VEL_WEIGHT = 1.0
PRIOR_VEL_SIGMA_SQ = 40.0

#: `ref_free_jump`'s window is 1000/3000 for its 6000-iteration, 4096-env runs.
#: This task's first full runs are on a laptop RTX 3050 (1024 envs, ~12 s an
#: iteration), where that recipe is 20+ hours; the window is compressed to
#: 300/1200 for a 3000-iteration run and holds the same shape: a third holding
#: the recording's jump, two thirds handing it over. A machine that can afford
#: the original window wants it -- more iterations at the feedforward means a
#: surer technique before the objective takes over.
PRIOR_HOLD_ITERATIONS = 300
PRIOR_END_ITERATIONS = 1200

# ── The objective: the rise ────────────────────────────────────────────────────
#: `ref_free_jump`'s height pair, as it stands there.
HEIGHT_SCALE = 0.10
APEX_WEIGHT = 4.0
FLIGHT_WEIGHT = 2.0

# ── The objective: the turn ────────────────────────────────────────────────────
#: One full revolution is the unit; the weights sit beside the height pair's
#: (see the module docstring).
#:
#: **Rebalanced mid-run (2026-11-13, iteration 2000).** At equal weights the
#: first 2000 iterations bought height -- 20 cm, from the recording's 12 -- and
#: no turn at all (3-5 degrees, flat for 500+ iterations): with the height
#: uncapped and the turn capped by what 0.4 s of flight allows, the policy had
#: no reason to look for the asymmetric push a turn needs. The turn now leads
#: 7:4 while the height holds its weight, and the dense half doubles beside it.
#: On a resumed run the curriculum's cross-fade is already past its window, so
#: these values apply from the first resumed iteration.
SPIN_WEIGHT = 7.0
FLIGHT_SPIN_WEIGHT = 4.0

# ── Around the jump ────────────────────────────────────────────────────────────
#: All of these are `ref_free_jump`'s, with its measurements.
HOME_WEIGHT = 1.0
HOME_SIGMA_SQ = 0.05

UPRIGHT_WEIGHT = 1.0
UPRIGHT_SIGMA_SQ = 0.10

DRIFT_WEIGHT = -1.0
DRIFT_SCALE = 0.10

ILLEGAL_CONTACT_WEIGHT = -5.0
LEG_COLLISION_WEIGHT = -5.0

ACTION_RATE_PER_RAD_SQ = -0.2304

#: The controller's own tilt limit, as `ref_free_jump` keeps it: a yaw about
#: the vertical never approaches it, and a policy that could recover from past
#: it would be cut off on the robot.
FELL_OVER_RAD = math.radians(50.0)

HOME_TOLERANCE = 0.05
HOME_SPEED_SQ = 0.5
HOME_HOLD_STEPS = 20

#: The rise, the turn, the landing and the way home: `ref_free_jump`'s 2.5 s
#: measured 1.84 s at the latest; a spinning trunk lands with angular momentum
#: to arrest, so a half-second of margin is added rather than borrowed.
EPISODE_S = 3.0

_COLLISION_GEOMS = r".*_meshcol"


def _stride_history(cfg: ManagerBasedRlEnvCfg) -> None:
    """`ref_free_jump`'s handover, unchanged: see its `env_cfg.py` for the three
    parts that each fail silently."""
    p = StridedHistory.PREFIX
    for group_name, group in cfg.observations.items():
        if group.history_length:
            raise ValueError(
                f"{group_name}: a group-level history_length overwrites every "
                f"term's, and would stack these frames a second time"
            )
        missing = HISTORY_TERMS - set(group.terms)
        if missing:
            raise ValueError(f"{group_name} has no {sorted(missing)} to give history to")
        for name in HISTORY_TERMS:
            term = group.terms[name] = copy.copy(group.terms[name])
            assert not term.history_length, f"{group_name}/{name} is already stacked"
            assert not any(k.startswith(p) for k in term.params or {}), (
                f"{group_name}/{name} has a param colliding with {p!r}"
            )
            term.params = {
                **(term.params or {}),
                p + "func": term.func,
                p + "frames": OBS_HISTORY,
                p + "stride": OBS_HISTORY_STRIDE,
                p + "noise": term.noise if group.enable_corruption else None,
            }
            term.func = StridedHistory
            term.noise = None


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Whirlwind jump: rise, full mid-air turn, landing.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise and pushes off, and every episode starts at HOME
            and jumps on the go, as the deployment will.
    """
    cfg = make_velocity_env_cfg()

    # ── Sim ───────────────────────────────────────────────────────────────────
    cfg.sim.mujoco.timestep = _SIM_TIMESTEP
    cfg.decimation = _DECIMATION
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128

    # ── Scene ─────────────────────────────────────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    robot = get_jumper_robot_cfg(effort_limit=EFFORT_LIMIT, asset=asset)
    for actuator in robot.articulation.actuators:
        actuator.stiffness = _KP
    cfg.scene.entities = {"robot": robot}

    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=_DECIMATION,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="geom", pattern=_COLLISION_GEOMS, entity="robot",
                                 exclude=FOOT_GEOMS),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
    )

    # ── Observations ──────────────────────────────────────────────────────────
    for group in cfg.observations.values():
        for name in ("command", "height_scan", "foot_height", "foot_air_time",
                     "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)
    del cfg.observations["actor"].terms["base_lin_vel"]

    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = dict(term.params or {}) | {"asset_cfg": gait_cfg}

    for group in cfg.observations.values():
        if "base_ang_vel" in group.terms:
            group.terms["base_ang_vel"].noise = Unoise(
                n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE
            )
        if "projected_gravity" in group.terms:
            group.terms["projected_gravity"].noise = Unoise(
                n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE
            )

    force_cfg = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        group.terms["actuator_force"] = ObservationTermCfg(
            func=actuator_force,
            params={"asset_cfg": force_cfg},
            noise=Unoise(n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE),
        )

    # The go clock and what the rewards are computed from, for the critic
    # alone: `ref_free_jump`'s terms plus the turn's bookkeeping.
    cfg.observations["critic"].terms["jump_phase"] = ObservationTermCfg(
        func=jump_mdp.observations.jump_phase, params={"command_name": "jump"}
    )
    cfg.observations["critic"].terms["jump_state"] = ObservationTermCfg(
        func=spin_mdp.observations.jump_state,
        params={"command_name": "jump", "scale": HEIGHT_SCALE, "mirror_kind": "scalar"},
    )
    cfg.observations["critic"].terms["spin_state"] = ObservationTermCfg(
        func=spin_mdp.observations.spin_state,
        params={"command_name": "jump", "turn": SPIN_TURN, "mirror_kind": "scalar"},
    )
    cfg.observations["critic"].terms["prior_target"] = ObservationTermCfg(
        func=spin_mdp.observations.prior_target,
        params={"command_name": "jump", "mirror_kind": "joint"},
    )
    cfg.observations["critic"].terms["feedforward"] = ObservationTermCfg(
        func=spin_mdp.observations.feedforward, params={"mirror_kind": "scalar"},
    )

    # ── Action: absolute targets around HOME, the recording fed forward ───────
    cfg.actions["joint_pos"] = spin_mdp.actions.PriorFeedforwardJointPositionActionCfg(
        entity_name="robot",
        actuator_names=list(GAIT_JOINTS),
        scale={j: ACTION_SCALE for j in GAIT_JOINTS},
        use_default_offset=True,
        command_name="jump",
        hold_step=NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
        end_step=NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS,
    )

    # ── Command: the go, with the turn bookkept ───────────────────────────────
    cfg.commands = {
        "jump": spin_mdp.commands.SpinJumpCommandCfg(
            entity_name="robot",
            motion_file=str(jump_mdp.reference.JUMP_REF_NPZ),
            command_delay_range=GO_DELAY_RANGE,
            rsi_fraction=RSI_FRACTION,
            rsi_phase_range=RSI_PHASE_RANGE,
            home_tolerance=HOME_TOLERANCE,
            home_speed_sq=HOME_SPEED_SQ,
        )
    }

    # ── Rewards ───────────────────────────────────────────────────────────────
    # The four scheduled weights are written as they stand at iteration 0; the
    # curriculum below moves them. Six now: the prior pair out, the height pair
    # and the spin pair in, all on one window.
    cfg.rewards = {
        "prior_joint_pos": RewardTermCfg(
            func=spin_mdp.rewards.prior_joint_pos, weight=PRIOR_POS_WEIGHT,
            params={"asset_cfg": gait_cfg, "command_name": "jump",
                    "sigma_sq": PRIOR_POS_SIGMA_SQ},
        ),
        "prior_joint_vel": RewardTermCfg(
            func=spin_mdp.rewards.prior_joint_vel, weight=PRIOR_VEL_WEIGHT,
            params={"asset_cfg": gait_cfg, "command_name": "jump",
                    "sigma_sq": PRIOR_VEL_SIGMA_SQ},
        ),
        "apex_height": RewardTermCfg(
            func=spin_mdp.rewards.apex_height, weight=0.0,
            params={"command_name": "jump", "scale": HEIGHT_SCALE},
        ),
        "flight_height": RewardTermCfg(
            func=spin_mdp.rewards.flight_height, weight=0.0,
            params={"command_name": "jump", "scale": HEIGHT_SCALE},
        ),
        "spin_angle": RewardTermCfg(
            func=spin_mdp.rewards.spin_angle, weight=0.0,
            params={"command_name": "jump", "turn": SPIN_TURN},
        ),
        "flight_spin": RewardTermCfg(
            func=spin_mdp.rewards.flight_spin, weight=0.0,
            params={"command_name": "jump", "turn": SPIN_TURN},
        ),
        "home_pose": RewardTermCfg(
            func=spin_mdp.rewards.home_pose, weight=HOME_WEIGHT,
            params={"command_name": "jump", "sigma_sq": HOME_SIGMA_SQ,
                    "asset_cfg": gait_cfg},
        ),
        "upright": RewardTermCfg(
            func=spin_mdp.rewards.upright, weight=UPRIGHT_WEIGHT,
            params={"sigma_sq": UPRIGHT_SIGMA_SQ},
        ),
        "horizontal_drift": RewardTermCfg(
            func=spin_mdp.rewards.horizontal_drift, weight=DRIFT_WEIGHT,
            params={"scale": DRIFT_SCALE},
        ),
        "illegal_contact": RewardTermCfg(
            func=spin_mdp.rewards.illegal_contact, weight=ILLEGAL_CONTACT_WEIGHT,
            params={"sensor_name": "body_ground_contact", "force_threshold": 1.0},
        ),
        "leg_collision": RewardTermCfg(
            func=spin_mdp.rewards.leg_collision, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_name": "self_collision", "force_threshold": 1.0},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, weight=ACTION_RATE_PER_RAD_SQ * ACTION_SCALE**2
        ),
    }

    # ── Curriculum: the prior comes down as the height and the turn go up ─────
    def _cross_fade(term: str, start: float, end: float) -> CurriculumTermCfg:
        return CurriculumTermCfg(
            func=spin_mdp.curriculum.anneal_reward_weight,
            params={
                "term_name": term,
                "start_weight": start,
                "end_weight": end,
                "start_step": NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
                "end_step": NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS,
            },
        )

    cfg.curriculum = {
        "prior_joint_pos_weight": _cross_fade("prior_joint_pos", PRIOR_POS_WEIGHT, 0.0),
        "prior_joint_vel_weight": _cross_fade("prior_joint_vel", PRIOR_VEL_WEIGHT, 0.0),
        "apex_height_weight": _cross_fade("apex_height", 0.0, APEX_WEIGHT),
        "flight_height_weight": _cross_fade("flight_height", 0.0, FLIGHT_WEIGHT),
        "spin_angle_weight": _cross_fade("spin_angle", 0.0, SPIN_WEIGHT),
        "flight_spin_weight": _cross_fade("flight_spin", 0.0, FLIGHT_SPIN_WEIGHT),
        "feedforward": CurriculumTermCfg(func=spin_mdp.curriculum.report_feedforward),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "back_home": TerminationTermCfg(
            func=spin_mdp.terminations.back_home, time_out=True,
            params={"command_name": "jump", "hold_steps": HOME_HOLD_STEPS},
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": FELL_OVER_RAD}
        ),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=FOOT_GEOMS
    )
    cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["push_robot"].params["velocity_range"].update(PUSH_VELOCITY_RANGE)
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events = {
        "reset_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default, mode="reset"
        ),
        "reset_from_reference": EventTermCfg(
            func=jump_mdp.events.reset_from_reference_phase,
            mode="reset",
            params={"command_name": "jump", "rsi_fraction": RSI_FRACTION,
                    "phase_range": RSI_PHASE_RANGE},
        ),
        **cfg.events,
    }

    # ── Episode and viewer ────────────────────────────────────────────────────
    cfg.episode_length_s = EPISODE_S
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.9
    cfg.viewer.elevation = -15.0

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events["reset_from_reference"].params["rsi_fraction"] = 0.0
        cfg.commands["jump"].rsi_fraction = 0.0
        cfg.curriculum = {}

    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    # Last, after `play`: the history takes each term's noise with it.
    _stride_history(cfg)
    return cfg


__all__ = ["env_cfg"]
