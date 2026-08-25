"""Evaluate or record an RL-Games blade-swap policy."""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-Insertion-Play-v0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; zero runs until the app closes.")
    parser.add_argument(
        "--episodes", type=int, default=0, help="Stop after N completed episodes; zero disables the gate."
    )
    parser.add_argument(
        "--minimum_success_rate",
        type=float,
        default=None,
        help="Return exit code 2 unless the completed-episode success rate reaches this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--curriculum_stage",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Force an insertion evaluation stage: 0=near, 1=medium, 2=full distance.",
    )
    parser.add_argument(
        "--robustness_level",
        type=int,
        choices=range(5),
        default=None,
        help="Insertion profile: 0=tight/6D, 1=pose, 2=mass, 3=friction, 4=mount wobble.",
    )
    parser.add_argument(
        "--contact_metrics",
        action="store_true",
        help=(
            "Attach a contact sensor to the blade and record peak contact force and impulse per episode. "
            "Off by default because contact reporting costs throughput; training never enables it."
        ),
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help=(
            "Override the task's episode length. For diagnosis only: a skill whose median cycle time equals its "
            "episode length is being cut off, and this says whether the unsatisfied condition was still moving. "
            "Never quote a certification run under an override -- change the task and let PHASE_BUDGET_S read it."
        ),
    )
    parser.add_argument(
        "--latch_mating_compliance",
        action="store_true",
        help=(
            "Engage the form lock straight into the remote-centre mating compliance, which is "
            "the state the chain is in while it seats. For a task whose episode begins at the "
            "mouth there is no rigid transit to soften out of."
        ),
    )
    parser.add_argument(
        "--latch_engage_after_steps",
        type=int,
        default=0,
        help=(
            "Defer form-lock engagement to this control step. Zero engages on the first "
            "qualifying step, which is what every published number was measured under; a task "
            "whose reset writes the module along a stroke needs the pair stepped together first."
        ),
    )
    parser.add_argument(
        "--latch_enabled",
        action="store_true",
        help=(
            "Run a grapple skill with the robot-side form lock engaged, which is how the chain "
            "carries and seats the module. Off by default because that is what every published "
            "skill number was measured under; this exists to measure the difference."
        ),
    )
    parser.add_argument(
        "--latch_rated_torque_nm",
        type=float,
        default=None,
        help=(
            "Override the modelled capture latch's rated torque. Sweeping this is how the interface "
            "specification's latch rating is derived: it is the smallest rating at which extraction certifies."
        ),
    )
    parser.add_argument(
        "--grip_axis_metrics",
        action="store_true",
        help=(
            "Record the capture-attitude error split into the gripper's own closing, third, and approach axes, "
            "instead of only its magnitude. An anti-yaw feature opposes rotation about the closing axis alone, "
            "so this is what says whether tightening one can help. Head-on capture tasks only; off by default "
            "so existing certifications keep their exact column set."
        ),
    )
    parser.add_argument(
        "--pose_noise_scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the reset joint-noise envelope. Values above 1 evaluate the policy outside the distribution "
            "it trained on; the report is flagged out_of_distribution so it is never read as certification."
        ),
    )
    parser.add_argument(
        "--belief_bias_mm",
        type=float,
        default=None,
        help=(
            "Pin the pose-belief error to exactly this magnitude, in millimetres, for one point of the "
            "success-versus-belief-error sweep. Above the trained ceiling this is a capability measurement, "
            "not certification, and the report is flagged out_of_distribution accordingly."
        ),
    )
    parser.add_argument(
        "--force_threshold_n",
        type=float,
        default=None,
        help="Pin the conditioned maximum allowable contact force instead of sampling it per episode.",
    )
    parser.add_argument(
        "--blade_mass_range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override the randomized blade mass range in kg. Outside the trained range this is a stress test.",
    )
    parser.add_argument(
        "--legacy_grip_ball_m",
        type=float,
        default=None,
        help=(
            "Judge the grip as an isotropic ball of this radius about the pin's drawing pose, "
            "the criterion every certification before 2026-08-24 was produced under, instead of "
            "the pin-resolved bounds. For one purpose: running an archived checkpoint under the "
            "criterion it was certified against, so a criterion change and a policy change are "
            "never quoted as one number. 0.030 is the value that was in the task."
        ),
    )
    parser.add_argument(
        "--legacy_unbounded_reset",
        action="store_true",
        help=(
            "Draw the reset's joint noise without bounding the grip error it induces, the way "
            "every certification before 2026-08-24 did. For attribution only: it is what makes "
            "'the reset was generating unwinnable episodes' a measurement rather than a claim."
        ),
    )
    parser.add_argument(
        "--blade_cross_section",
        type=float,
        nargs=2,
        metavar=("WIDTH_M", "HEIGHT_M"),
        default=None,
        help=(
            "Override the module's width and height, leaving its length and mass alone. "
            "The cross-section is what sets how much the rack's own channel constrains the "
            "module while it is still inside it, so this is the knob that says whether a "
            "skill's rate is about the skill or about the clearance it was measured at. "
            "Always out of distribution: no policy trained on a section it was not built for."
        ),
    )
    parser.add_argument(
        "--no_lead_in",
        action="store_true",
        help=(
            "Disable the slot's lead-in flares. A deliberate change to the task, not to the "
            "initial-condition distribution, so it marks the run out of distribution."
        ),
    )
    parser.add_argument(
        "--inspection_view",
        choices=("task", "grasp", "side", "top", "workcell", "array"),
        default="task",
        help=(
            "Presentation-only viewport angle; it does not change policy inputs or physics. "
            "'array' frames the whole parallel environment grid instead of one workcell."
        ),
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=600)
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Where to write the recording; defaults to <checkpoint>/../../videos/play, which successive runs overwrite.",
    )
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/play_report.json"),
        help="Machine-readable checkpoint load/play validation result.",
    )
    parser.add_argument(
        "--episode_metrics",
        type=Path,
        default=None,
        help="Write the raw per-episode terminal metric rows here (.npz) for pooled multi-run statistics.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.steps < 0 or args.episodes < 0:
    parser.error("--steps and --episodes must be non-negative")
if args.pose_noise_scale <= 0.0:
    parser.error("--pose_noise_scale must be positive")
if args.belief_bias_mm is not None and args.belief_bias_mm < 0.0:
    parser.error("--belief_bias_mm must be non-negative")
if args.force_threshold_n is not None and args.force_threshold_n <= 0.0:
    parser.error("--force_threshold_n must be positive")
if args.blade_mass_range is not None and not 0.0 < args.blade_mass_range[0] <= args.blade_mass_range[1]:
    parser.error("--blade_mass_range must be positive and ordered LOW HIGH")
if args.blade_cross_section is not None and min(args.blade_cross_section) <= 0.0:
    parser.error("--blade_cross_section must be positive")
if args.minimum_success_rate is not None and not 0.0 <= args.minimum_success_rate <= 1.0:
    parser.error("--minimum_success_rate must be between 0 and 1")
if "Vision" in args.task or args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.sensors import ContactSensorCfg

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.evaluation import (
    BLADE_MASS_FIELD,
    bucket_success_rates,
    group_rows,
    round_floats,
    summarize_terminal_episodes,
)
from zero_g_blade_swap.tasks.blade_swap.agents import register_rl_games_networks
from zero_g_blade_swap.tasks.blade_swap.mdp.terminal_metrics import InsertionTerminalMetrics


# Tasks that expose cumulative robustness profiles and a mount-stability term.
ROBUST_FAMILY_TASKS = (
    "Insertion-Robust",
    "Insertion-Contact",
    "Insertion-RigidGrasp",
    "Insertion-ForceLimited",
    "Insertion-StrictForceLimited",
    "Insertion-ForceFeedback",
    "Insertion-Uncertain",
    "Insertion-UncertainBlind",
    "Insertion-Vision",
    "Insertion-GuidedSlot",
    "Blade-GrapplePin",
)

# Task labels that share the rigid-grasp PPO configuration and checkpoint tree.
RIGID_GRASP_AGENT_TASKS = ("Insertion-RigidGrasp", "ForceLimited", "ForceFeedback", "GuidedSlot")

# Insertion under a wrong pose belief carries an asymmetric critic, so it has its
# own PPO configuration and its own checkpoint tree.
UNCERTAIN_AGENT_TASKS = ("Insertion-Uncertain", "Insertion-UncertainBlind")


def _checkpoint() -> Path:
    if args.checkpoint is not None:
        path = args.checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    default_experiment = (
        "zero_g_blade_insertion_uncertain"
        if any(label in args.task for label in UNCERTAIN_AGENT_TASKS)
        else "zero_g_blade_insertion_vision"
        if "Insertion-Vision" in args.task
        else "zero_g_blade_insertion_rigid_grasp"
        if any(label in args.task for label in RIGID_GRASP_AGENT_TASKS)
        else "zero_g_blade_insertion_contact"
        if "Insertion-Contact" in args.task or "GrapplePin" in args.task
        else "zero_g_blade_insertion_robust"
        if "Insertion-Robust" in args.task
        else "zero_g_blade_insertion"
    )
    experiment = args.experiment or default_experiment
    candidates = list((Path("logs") / "rl_games" / experiment).glob("**/nn/*.pth"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found; pass --checkpoint explicitly")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _robust_parameter_values(env) -> dict[str, tuple[torch.Tensor, float, float]]:
    """Snapshot per-environment physics values before automatic episode reset."""

    task = env.unwrapped
    level = int(task.cfg.robustness_level)
    values: dict[str, tuple[torch.Tensor, float, float]] = {}
    if level >= 2:
        mass = task.scene["spare_blade"].root_physx_view.get_masses()[:, 0].to(task.device).clone()
        values["mass_kg"] = (mass, 5.0, 15.0)
    if level >= 3:
        material_asset = "blade_slot_left_guide" if getattr(task.cfg, "rigid_grasp", False) else "blade_slot"
        material = task.scene[material_asset].root_physx_view.get_material_properties()
        dynamic_friction = material[:, 0, 1].to(task.device).clone()
        breakaway, viscous = task._rail_stiction["spare_blade"]
        values["dynamic_friction"] = (dynamic_friction, 0.20, 1.50)
        values["breakaway_force_n"] = (breakaway.clone(), 10.0, 120.0)
        values["viscous_drag_ns_per_m"] = (viscous.clone(), 2.0, 25.0)
    return values


def _update_robust_buckets(
    stats: dict[str, list[dict[str, int]]],
    values: dict[str, tuple[torch.Tensor, float, float]],
    dones: torch.Tensor,
    successes: torch.Tensor,
) -> None:
    """Accumulate low/mid/high success counts for completed episodes."""

    completed = dones.to(dtype=torch.bool)
    for name, (value, low, high) in values.items():
        normalized = ((value - low) / (high - low)).clamp(0.0, 1.0 - 1.0e-7)
        bucket_ids = torch.floor(3.0 * normalized).to(dtype=torch.long)
        buckets = stats.setdefault(name, [{"episodes": 0, "successes": 0} for _ in range(3)])
        for index in range(3):
            selected = completed & (bucket_ids == index)
            buckets[index]["episodes"] += int(selected.sum())
            buckets[index]["successes"] += int((successes & selected).sum())


def _summarize_robust_buckets(stats: dict[str, list[dict[str, int]]]) -> dict[str, object]:
    labels = ("low", "mid", "high")
    output: dict[str, object] = {}
    observed_rates: list[float] = []
    for name, buckets in stats.items():
        parameter: dict[str, dict[str, int | float | None]] = {}
        for label, bucket in zip(labels, buckets, strict=True):
            episodes = bucket["episodes"]
            rate = bucket["successes"] / episodes if episodes else None
            parameter[label] = {**bucket, "success_rate": rate}
            if rate is not None:
                observed_rates.append(rate)
        output[name] = parameter
    output["minimum_observed_success_rate"] = min(observed_rates) if observed_rates else None
    output["all_observed_buckets_at_least_80_percent"] = (
        all(rate >= 0.80 for rate in observed_rates) if observed_rates else None
    )
    return output


def _lead_in_present(env_cfg) -> bool | None:
    """Are the slot's lead-in flares collidable? None when the task has none.

    Only ``collision_enabled is False`` disables a collider. ``None`` is Isaac
    Lab's "leave as authored", and these flares are authored collidable.
    """

    flare = getattr(env_cfg.scene, "blade_slot_entry_left_flare", None)
    if flare is None:
        return None
    return flare.spawn.collision_props.collision_enabled is not False


def _disable_lead_in(env_cfg) -> None:
    """Turn every lead-in flare in the scene into a non-collider.

    The mechanism probe this project ran by hand once, made reproducible: two
    fully trained insertion policies both score 0% without these plates, which
    is the measurement ``lead_in_present`` exists to label.
    """

    disabled = []
    for name in dir(env_cfg.scene):
        if "entry_" not in name or "flare" not in name:
            continue
        collision_props = getattr(getattr(getattr(env_cfg.scene, name, None), "spawn", None), "collision_props", None)
        if collision_props is None:
            continue
        collision_props.collision_enabled = False
        disabled.append(name)
    if not disabled:
        raise ValueError("--no_lead_in needs a task whose scene carries lead-in flares")
    print(f"[INFO] Lead-in flares disabled: {', '.join(sorted(disabled))}")


def _apply_stress(env_cfg) -> dict[str, object]:
    """Widen the reset distribution to probe the policy's capability envelope.

    These knobs change only the initial-condition distribution, never the task
    physics, reward, observation, or success criterion, so a stressed run stays
    comparable to certification apart from how hard the starting state is.
    """

    if args.no_lead_in:
        _disable_lead_in(env_cfg)
    # The insert task's reset writes the arm, the module and the fingers together
    # from a bank of stations, so it has no ``reset_arm`` term and no per-stage
    # noise ladder -- one scalar covers every station. Read whichever exists.
    reset_arm = getattr(env_cfg.events, "reset_arm", None)
    reset_stroke = getattr(env_cfg.events, "reset_stroke", None)
    if reset_arm is not None:
        trained_noise = tuple(reset_arm.params["noise_by_stage"])
    elif reset_stroke is not None:
        trained_noise = (float(reset_stroke.params["noise_rad"]),)
    else:
        raise ValueError("this task has neither a reset_arm nor a reset_stroke term")
    trained_mass = (
        tuple(env_cfg.events.blade_mass.params["mass_distribution_params"])
        if getattr(env_cfg.events, "blade_mass", None) is not None
        else None
    )
    trained_bias_ceiling_m = None
    belief_term = getattr(env_cfg.curriculum, "belief_bias", None)
    if belief_term is not None:
        trained_bias_ceiling_m = float(belief_term.params["bias_ceiling_m"])
    stress: dict[str, object] = {
        "pose_noise_scale": args.pose_noise_scale,
        "trained_pose_noise_rad": list(trained_noise),
        "trained_blade_mass_kg": list(trained_mass) if trained_mass else None,
        "blade_mass_range_kg": list(args.blade_mass_range) if args.blade_mass_range else None,
        "trained_belief_bias_ceiling_mm": (
            1_000.0 * trained_bias_ceiling_m if trained_bias_ceiling_m is not None else None
        ),
        "belief_bias_mm": args.belief_bias_mm,
        "force_threshold_n": args.force_threshold_n,
        # Whether the slot's lead-in flares are collidable. Removing them is a
        # deliberate change to the task, not to the initial-condition
        # distribution, so it has to reach the report or a 0% result would be
        # filed as a failed certification instead of a mechanism probe.
        #
        # ``collision_enabled`` is Isaac Lab's TRI-STATE: None means "leave the
        # authored USD alone", which for these flares means collidable, because
        # nothing in assets.py ever sets it. Reading it with ``bool()`` collapsed
        # that None to False and stamped every grapple-pin report
        # ``out_of_distribution: true`` with ``gate.applies: false`` -- a wrong
        # label on correct rates, corrected on 2026-08-18 by re-aggregating the
        # archived rows rather than by re-measuring them. train.py always read
        # the same field correctly; compare its ``collision_enabled is False``.
        "lead_in_present": _lead_in_present(env_cfg),
        "legacy_grip_ball_m": args.legacy_grip_ball_m,
        "legacy_unbounded_reset": bool(args.legacy_unbounded_reset),
        "trained_blade_cross_section_m": None,
        "blade_cross_section_m": list(args.blade_cross_section) if args.blade_cross_section else None,
    }
    blade = getattr(env_cfg.scene, "spare_blade", None)
    if blade is not None and getattr(blade.spawn, "size", None) is not None:
        stress["trained_blade_cross_section_m"] = list(tuple(blade.spawn.size)[1:])
    if args.blade_cross_section is not None:
        if blade is None or getattr(blade.spawn, "size", None) is None:
            raise ValueError("--blade_cross_section needs a task whose module is a sized chassis")
        length = float(tuple(blade.spawn.size)[0])
        blade.spawn.size = (length, float(args.blade_cross_section[0]), float(args.blade_cross_section[1]))
        print(f"[INFO] Module cross-section overridden to {blade.spawn.size} m (length unchanged)")
    if args.belief_bias_mm is not None and belief_term is None:
        raise ValueError("--belief_bias_mm needs a task whose actor observes a pose belief")
    if args.pose_noise_scale != 1.0:
        scaled = tuple(value * args.pose_noise_scale for value in trained_noise)
        if reset_arm is not None:
            reset_arm.params["noise_by_stage"] = scaled
        else:
            reset_stroke.params["noise_rad"] = scaled[0]
        print(f"[INFO] Reset pose noise scaled {args.pose_noise_scale}x to {scaled} rad")
    if args.blade_mass_range is not None:
        if trained_mass is None:
            raise ValueError("--blade_mass_range needs a task that randomizes blade mass (robustness level >= 2)")
        env_cfg.events.blade_mass.params["mass_distribution_params"] = tuple(args.blade_mass_range)
        print(f"[INFO] Blade mass range overridden to {tuple(args.blade_mass_range)} kg")

    outside_mass = bool(
        args.blade_mass_range is not None
        and trained_mass is not None
        and (args.blade_mass_range[0] < trained_mass[0] or args.blade_mass_range[1] > trained_mass[1])
    )
    outside_belief = bool(
        args.belief_bias_mm is not None
        and trained_bias_ceiling_m is not None
        and args.belief_bias_mm > 1_000.0 * trained_bias_ceiling_m + 1.0e-9
    )
    stress["out_of_distribution"] = bool(
        args.pose_noise_scale > 1.0
        or outside_mass
        or outside_belief
        # Every policy trained with the lead-in present, so evaluating without it
        # is out of distribution by construction.
        or stress["lead_in_present"] is False
        # A section the policy did not train on changes what the rack's channel
        # holds, which is task physics rather than an initial condition.
        or args.blade_cross_section is not None
    )
    return stress


def _write_episode_metrics(path: Path, recorder, metadata: dict[str, object]) -> None:
    """Persist raw per-episode rows so several runs can be pooled exactly.

    The field list is stored alongside the rows because a task may record extra
    columns, such as randomized blade mass, that other runs do not have.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rows=np.asarray(recorder.rows, dtype=np.float32),
        fields=np.asarray(recorder.fields),
        metadata=np.asarray(json.dumps(metadata)),
    )


def _effective_mass_range(stress: dict[str, object]) -> tuple[float, float]:
    """The blade mass range actually sampled, trained or overridden."""

    configured = stress.get("blade_mass_range_kg") or stress.get("trained_blade_mass_kg") or (5.0, 15.0)
    return float(configured[0]), float(configured[1])


def _terminal_metrics_report(
    collector: InsertionTerminalMetrics,
    episodes_completed: int,
    mass_range: tuple[float, float],
) -> dict[str, object]:
    """Summarize episodes captured before Isaac Lab's automatic reset."""

    rows = collector.recorder.rows
    fields = collector.recorder.fields
    report: dict[str, object] = {
        "source": "captured_before_auto_reset",
        "recorded_episodes": len(collector.recorder),
        "matches_step_done_count": len(collector.recorder) == episodes_completed,
        "fields": list(fields),
    }
    report.update(summarize_terminal_episodes(rows, fields))
    if BLADE_MASS_FIELD in fields:
        # Bucket over the range actually sampled, otherwise a stress run's
        # heavier blades all collapse into the top band.
        low, high = mass_range
        report["by_blade_mass_bucket"] = bucket_success_rates(rows, BLADE_MASS_FIELD, low, high, fields)
    stages = group_rows(rows, "curriculum_stage", fields)
    if len(stages) > 1:
        # A stage-forced evaluation would only duplicate the pooled block.
        report["by_curriculum_stage"] = {
            str(stage): summarize_terminal_episodes(block, fields, include_successful_metrics=False)
            for stage, block in stages.items()
        }
    else:
        report["curriculum_stages_present"] = sorted(stages)
    return report


def main() -> dict[str, object]:
    env = None
    try:
        checkpoint = _checkpoint()
        rl_device = args.device or "cuda:0"
        if rl_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"GPU playback requested on {rl_device}, but PyTorch cannot access CUDA")
        device_description = (
            torch.cuda.get_device_name(torch.device(rl_device)) if rl_device.startswith("cuda") else "CPU"
        )
        print(f"[INFO] Simulation and policy device: {rl_device} ({device_description})")
        env_cfg = parse_env_cfg(args.task, device=rl_device, num_envs=args.num_envs)
        if args.robustness_level is not None:
            if not any(
                label in args.task for label in ROBUST_FAMILY_TASKS
            ):
                raise ValueError("--robustness_level is valid only for a robust or contact insertion task")
            env_cfg.configure_robustness(args.robustness_level)
            print(f"[INFO] Insertion robustness level: {args.robustness_level}")
        # Stress knobs must be applied after configure_robustness, which rebuilds
        # the event configuration and would otherwise discard them.
        stress = _apply_stress(env_cfg)
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s
            print(f"[INFO] Episode length overridden to {args.episode_length_s} s (diagnosis only)")
        if args.latch_enabled:
            if getattr(env_cfg, "latch_enabled", None) is None:
                raise ValueError("--latch_enabled is valid only for a grapple-pin task")
            # `_configure_latch` sets `events.grapple_latch = None` when the
            # latch is off, and a None cannot be configured back into an event.
            # `configure_robustness` is the supported way to rebuild the event
            # set -- its own docstring says it exists so callers may change
            # things after `parse_env_cfg` -- so flip the flag first and let it
            # reinstall the term.
            env_cfg.latch_enabled = True
            if args.latch_mating_compliance:
                # **The mating compliance is only reachable from the rigid
                # state.** With joint_mode "compliant" the load path is the
                # explicit wrench and the mating joint is never installed, so
                # softening re-anchors a transform that engagement anchored one
                # line earlier and changes nothing -- measured byte-identical.
                # The chain runs the lock "fixed" for exactly this reason: a
                # fixed joint carries the transit, and `soften` disables it and
                # hands the load path to the remote-centre mating joint.
                env_cfg.latch_joint_mode = "fixed"
                # **And the joints do not survive scene replication.** PhysX
                # copies only the first environment's procedurally authored
                # joint, so envs 1..N get the prim and no usable joint --
                # `configure_base_rail` records the same thing and turns
                # replication off for exactly this reason. The skill tasks run
                # replicate_physics=True for throughput, which is why the
                # chain's load path is not reachable from them as they stand.
                env_cfg.scene.replicate_physics = False
                env_cfg.scene.clone_in_fabric = False
                print("[INFO] replicate_physics disabled: latch joints do not replicate")
            env_cfg.configure_robustness(int(env_cfg.robustness_level))
            if getattr(env_cfg.events, "grapple_latch", None) is None:
                raise RuntimeError("--latch_enabled did not reinstall the grapple latch event")
            if args.latch_mating_compliance:
                env_cfg.events.grapple_latch.params["soften_on_engage"] = True
                print("[INFO] Form lock engages straight into the mating compliance")
            if args.latch_engage_after_steps:
                env_cfg.events.grapple_latch.params["engage_after_steps"] = (
                    args.latch_engage_after_steps
                )
                print(
                    f"[INFO] Form lock defers engagement to control step "
                    f"{args.latch_engage_after_steps}"
                )
            print(
                f"[INFO] Form lock ENABLED for this evaluation, joint_mode="
                f"{env_cfg.latch_joint_mode} (diagnostic; the skill's default is off)"
            )
        if args.latch_rated_torque_nm is not None:
            if getattr(env_cfg, "latch_enabled", None) is None:
                raise ValueError("--latch_rated_torque_nm is valid only for a grapple-pin task")
            env_cfg.latch_rated_torque_nm = args.latch_rated_torque_nm
            # configure_robustness already ran in __post_init__, so push the new
            # rating into the event set that is actually installed.
            env_cfg._configure_latch()
            print(f"[INFO] Capture latch rated at {args.latch_rated_torque_nm} N-m")
        if args.legacy_unbounded_reset:
            reset_arm = getattr(env_cfg.events, "reset_arm", None)
            if reset_arm is None or "max_tool_offset_m" not in getattr(reset_arm, "params", {}):
                raise ValueError("--legacy_unbounded_reset needs a task whose reset bounds the grip error")
            reset_arm.params = {**reset_arm.params, "max_tool_offset_m": None}
            print("[INFO] Reset joint noise unbounded (legacy)")
        if args.legacy_grip_ball_m is not None:
            failed = getattr(env_cfg.terminations, "extraction_failed", None)
            if failed is None:
                raise ValueError("--legacy_grip_ball_m is valid only for a task that ends on extraction_failed")
            failed.params = {**failed.params, "grip_position_limit": args.legacy_grip_ball_m}
            print(f"[INFO] Grip judged as a {args.legacy_grip_ball_m * 1000:.1f} mm ball (legacy criterion)")
        if args.grip_axis_metrics:
            if getattr(env_cfg, "tool_target_rot", None) is None:
                raise ValueError("--grip_axis_metrics is valid only for a head-on capture task")
            # Read by InsertionTerminalMetrics when it builds its column set.
            env_cfg.record_grip_axes = True
            print("[INFO] Recording capture attitude per gripper axis")
        if args.contact_metrics and getattr(env_cfg.scene, "blade_contact", None) is not None:
            print("[INFO] Task already reports blade contacts; --contact_metrics is redundant")
        elif args.contact_metrics:
            if "Insertion" not in args.task:
                raise ValueError("--contact_metrics is valid only for an insertion task")
            # PhysX already solves these contacts; this only asks it to report
            # them. The sensor is evaluation-only, so training throughput and
            # the trained policy are untouched.
            env_cfg.scene.spare_blade.spawn.activate_contact_sensors = True
            # Fabric-cloned prims do not carry the contact-report API, so the
            # sensor would resolve zero bodies on every clone but env 0.
            env_cfg.scene.clone_in_fabric = False
            env_cfg.scene.blade_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/SpareBlade",
                update_period=0.0,
                history_length=0,
            )
            print("[INFO] Contact reporting enabled on the blade")
        inspection_views = {
            "grasp": ((0.18, -1.05, 1.02), (0.52, 0.0, 0.72)),
            "side": ((0.52, -1.20, 0.82), (0.58, 0.0, 0.72)),
            "top": ((0.58, -0.05, 1.55), (0.58, 0.0, 0.70)),
            "workcell": ((-0.50, -1.80, 1.25), (0.55, 0.0, 0.72)),
        }
        if args.video and args.num_envs > 1:
            # Fabric-cloned prims do not all reach the RTX renderer, so a
            # multi-environment recording shows empty workcells. Recording is
            # presentation only and uses few environments, so trade the cloning
            # speedup for correct visuals. Physics is unchanged.
            env_cfg.scene.clone_in_fabric = False
            print("[INFO] Disabled Fabric cloning so every recorded environment renders")
        if args.inspection_view != "task" and args.inspection_view != "array":
            env_cfg.viewer.eye, env_cfg.viewer.lookat = inspection_views[args.inspection_view]
            print(f"[INFO] Inspection view: {args.inspection_view}")
        env_cfg.seed = args.seed
        # Every task in this package is now an insertion-family task: they all
        # run on TerminalMetricsManagerBasedRLEnv, carry a curriculum stage, and
        # end on a named success term. The head-on capture scene is included
        # even though its identifier does not say "Insertion".
        agent_task = (
            "Isaac-ZeroG-Blade-Insertion-Uncertain-v0"
            if any(label in args.task for label in UNCERTAIN_AGENT_TASKS)
            else "Isaac-ZeroG-Blade-Insertion-Vision-v0"
            if "Insertion-Vision" in args.task
            else "Isaac-ZeroG-Blade-Insertion-RigidGrasp-v0"
            if any(label in args.task for label in RIGID_GRASP_AGENT_TASKS)
            else "Isaac-ZeroG-Blade-Insertion-Contact-v0"
            if "Insertion-Contact" in args.task or "GrapplePin" in args.task
            else "Isaac-ZeroG-Blade-Insertion-Robust-v0"
            if "Insertion-Robust" in args.task
            else "Isaac-ZeroG-Blade-Insertion-v0"
        )
        agent_cfg = load_cfg_from_registry(agent_task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["seed"] = args.seed
        agent_cfg["params"]["config"]["device"] = rl_device
        agent_cfg["params"]["config"]["device_name"] = rl_device
        options = agent_cfg["params"]["env"]
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.force_threshold_n is not None:
            # Read before the first reset, so the first episode is already
            # sampled at the pinned value rather than one step behind it.
            env.unwrapped._forced_contact_force_threshold_n = float(args.force_threshold_n)
            print(f"[INFO] Contact force threshold pinned to {args.force_threshold_n} N")
        if args.belief_bias_mm is not None:
            for name, term_cfg in zip(
                env.unwrapped.curriculum_manager._term_names,
                env.unwrapped.curriculum_manager._term_cfgs,
                strict=True,
            ):
                if name == "belief_bias":
                    term_cfg.func.force_bias(args.belief_bias_mm / 1_000.0)
            print(f"[INFO] Pose-belief error pinned to {args.belief_bias_mm} mm")
        if args.curriculum_stage is not None:
            insertion_env = env.unwrapped
            insertion_env._insertion_curriculum_stage.fill_(args.curriculum_stage)
            for name, term_cfg in zip(
                insertion_env.curriculum_manager._term_names,
                insertion_env.curriculum_manager._term_cfgs,
                strict=True,
            ):
                if name == "pose_noise":
                    term_cfg.func.force_reset_stage(args.curriculum_stage)
        if args.inspection_view == "array":
            # Frame whatever grid the cloner actually produced instead of
            # assuming a layout; a wrong guess renders empty space.
            origins = env.unwrapped.scene.env_origins
            centre = origins.mean(dim=0).tolist()
            extent = float((origins.max(dim=0).values - origins.min(dim=0).values).max())
            distance = max(extent, 1.0) * 1.15 + 2.5
            env.unwrapped.sim.set_camera_view(
                eye=(centre[0] - 0.45 * distance, centre[1] - 0.95 * distance, 0.62 * distance),
                target=(centre[0], centre[1], centre[2] + 0.72),
            )
            print(f"[INFO] Inspection view: array framing {args.num_envs} environments over {extent:.1f} m")
        task_env = env.unwrapped
        if not hasattr(task_env, "enable_terminal_metrics"):
            raise RuntimeError(
                f"{args.task} is not registered on TerminalMetricsManagerBasedRLEnv; "
                "terminal metrics would be read after the automatic reset"
            )
        terminal_metrics = InsertionTerminalMetrics(task_env)
        task_env.enable_terminal_metrics(terminal_metrics)
        # Each skill ends on its own named success term. Resolve it from the task
        # rather than assuming, so a grasp run is not scored against an insertion
        # predicate it never defines, and fail loudly if none is present rather
        # than silently reporting 0%.
        active_terms = env.unwrapped.termination_manager.active_terms
        success_term_name = next(
            (
                name
                for name in ("insertion_success", "extraction_success", "capture_success")
                if name in active_terms
            ),
            None,
        )
        if success_term_name is None:
            raise RuntimeError(
                f"{args.task} declares no known success termination; active terms are {sorted(active_terms)}"
            )
        print(f"[INFO] Success term: {success_term_name}")
        if args.video:
            video_dir = args.video_dir or (checkpoint.parent.parent / "videos" / "play")
            print(f"[INFO] Recording to {video_dir}")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RlGamesVecEnvWrapper(
            env,
            agent_cfg["params"]["config"]["device"],
            options.get("clip_observations", math.inf),
            options.get("clip_actions", math.inf),
            options.get("obs_groups"),
            options.get("concate_obs_groups", True),
        )
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: env})
        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = str(checkpoint)
        register_rl_games_networks()
        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(str(checkpoint))
        player.reset()

        obs = env.reset()
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
        player.get_batch_size(obs, 1)
        step = 0
        dt = env.unwrapped.step_dt
        # Read the terms the task actually declares. A hardcoded list silently
        # drops any new termination, such as a force-limit abort.
        termination_names = tuple(env.unwrapped.termination_manager.active_terms)
        termination_counts = {name: 0 for name in termination_names}
        failure_condition_counts: dict[str, int] = {}
        episodes_completed = 0
        reward_sum = torch.zeros(args.num_envs, device=rl_device)
        robust_bucket_stats: dict[str, list[dict[str, int]]] = {}
        while (
            simulation_app.is_running()
            and (args.steps <= 0 or step < args.steps)
            and (args.episodes <= 0 or episodes_completed < args.episodes)
        ):
            started = time.perf_counter()
            robust_values = (
                _robust_parameter_values(env)
                if any(
                    label in args.task
                    for label in ROBUST_FAMILY_TASKS
                )
                else {}
            )
            with torch.inference_mode():
                # Sample the contact left by the previous step before this one
                # overwrites it; the terminal step is folded in at reset.
                terminal_metrics.accumulate(env.unwrapped)
                network_obs = player.obs_to_torch(obs)
                actions = player.get_action(network_obs, is_deterministic=True)
                if not bool(torch.isfinite(actions).all()):
                    raise RuntimeError(f"Policy produced a non-finite action at step {step}")
                obs, rewards, dones, _ = env.step(actions)
                observation_tensor = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
                if not bool(torch.isfinite(observation_tensor).all()):
                    raise RuntimeError(f"Environment produced a non-finite observation at step {step}")
                reward_sum += rewards
                episodes_completed += int(dones.sum())
                for name in termination_names:
                    termination_counts[name] += int(env.unwrapped.termination_manager.get_term(name).sum())
                # Two families write the same kind of breakdown under different
                # names, and only the insertion one was ever read. An extract
                # certification that reports 1,925 ``extraction_failed`` and
                # nothing about *which* of the four conditions fired says only
                # that a policy failed, which is the part already known.
                for attribute, term in (
                    ("_insertion_latest_failure_conditions", "insertion_failed"),
                    ("_grapple_latest_failure_conditions", "extraction_failed"),
                    ("_grapple_latest_failure_conditions", "capture_failed"),
                ):
                    latest_failures = getattr(env.unwrapped, attribute, None)
                    if latest_failures is None or term not in termination_names:
                        continue
                    failed = env.unwrapped.termination_manager.get_term(term)
                    for name, condition in latest_failures.items():
                        failure_condition_counts[name] = failure_condition_counts.get(name, 0) + int(
                            (condition & failed).sum()
                        )
                if robust_values:
                    _update_robust_buckets(
                        robust_bucket_stats,
                        robust_values,
                        dones,
                        env.unwrapped.termination_manager.get_term(success_term_name),
                    )
                if player.is_rnn and player.states is not None:
                    for state in player.states:
                        state[:, dones, :] = 0.0
            step += 1
            if args.video and step >= args.video_length:
                break
            delay = dt - (time.perf_counter() - started)
            if args.real_time and delay > 0:
                time.sleep(delay)
        if args.steps > 0 and args.episodes <= 0 and step != args.steps:
            raise RuntimeError(f"Simulation stopped after {step} of {args.steps} requested steps")
        success_name = success_term_name
        success_rate = termination_counts[success_name] / max(episodes_completed, 1)
        success_rate_source = "termination_term_counts"
        if len(terminal_metrics.recorder) > 0:
            # Prefer the reset-safe count.  It also demotes an episode that hit a
            # non-finite or mount failure in the same control step as a success.
            successes = float(terminal_metrics.recorder.column("success").sum())
            success_rate = successes / len(terminal_metrics.recorder)
            success_rate_source = "terminal_metrics_captured_before_reset"
        meets_threshold = args.minimum_success_rate is None or success_rate >= args.minimum_success_rate
        result = {
            "status": "passed",
            "task": args.task,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper(),
            "device": rl_device,
            "gpu": device_description,
            "num_envs": args.num_envs,
            "steps": step,
            "episodes_completed": episodes_completed,
            "termination_counts": termination_counts,
            "success_rate": success_rate,
            "success_rate_source": success_rate_source,
            "minimum_success_rate": args.minimum_success_rate,
            "meets_threshold": meets_threshold,
            "curriculum_stage": args.curriculum_stage,
            "robustness_level": getattr(env_cfg, "robustness_level", None),
            "grasp_mode": (
                "physical_contact"
                if getattr(env_cfg, "contact_grasp", False)
                else "physx_fixed_joint"
                if getattr(env_cfg, "rigid_grasp", False)
                else "secured_fixture"
            ),
            "inspection_view": args.inspection_view,
            # Force policies are compared against each other, and a tighter
            # abort limit truncates the force distribution being compared, so
            # record the limit each run was actually judged under.
            "contact_force_limit_n": (
                float(
                    env.unwrapped.termination_manager.get_term_cfg("excessive_contact_force").params["force_limit_n"]
                )
                if "excessive_contact_force" in termination_names
                else None
            ),
            "stress": stress,
            "mean_cumulative_reward_per_environment": float(reward_sum.mean()),
        }
        # Terminal axial/lateral/orientation/velocity/tool-to-handle errors come
        # from the pre-reset snapshot.  Reading them from the live scene here
        # would measure the freshly reset episode instead.
        result["terminal_metrics"] = _terminal_metrics_report(
            terminal_metrics, episodes_completed, _effective_mass_range(stress)
        )
        if args.episode_metrics is not None:
            _write_episode_metrics(
                args.episode_metrics,
                terminal_metrics.recorder,
                {
                    "task": args.task,
                    "seed": args.seed,
                    "curriculum_stage": args.curriculum_stage,
                    "robustness_level": getattr(env_cfg, "robustness_level", None),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "num_envs": args.num_envs,
                    "contact_force_limit_n": result["contact_force_limit_n"],
                    "stress": stress,
                },
            )
        timeout_failures = getattr(env.unwrapped, "_insertion_timeout_failure_counts", {})
        result["timeout_failed_conditions"] = {name: int(count.item()) for name, count in timeout_failures.items()}
        result["failure_condition_counts"] = failure_condition_counts
        if any(label in args.task for label in ROBUST_FAMILY_TASKS):
            result["randomization_buckets"] = _summarize_robust_buckets(robust_bucket_stats)
        return result
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    report: dict[str, object]
    try:
        report = main()
    except BaseException as exc:
        traceback.print_exc()
        report = {
            "status": "failed",
            "task": args.task,
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "num_envs": args.num_envs,
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
        simulation_app.close()
    if report.get("status") == "passed" and not report.get("meets_threshold", True):
        raise SystemExit(2)
