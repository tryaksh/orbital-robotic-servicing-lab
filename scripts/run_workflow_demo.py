"""Run capture, extraction, transit, and re-insertion in one episode.

Three separately trained checkpoints drive one continuous episode on one module.
The driver switches between them on **measured conditions**, never on a timer:
the capture hands over when the drive torque says the pads are loaded on the pin,
the pull hands over when the module's rear face is clear of the rack mouth, and
the transit hands over when the module reaches the pose the insert policy was
trained from.

What is learned and what is not, stated plainly, because a demonstration that
blurs this is worthless:

* capture, extraction and insertion are trained policies, run deterministically
  from their checkpoints;
* the transit between "clear of the rack" and "lined up to go back in" is
  **scripted**, because there is no contact in it and nothing for a policy to
  learn. It retraces the path the extraction actually took, in reverse. A blind
  axial command does not work and the reason is worth recording: the extracted
  pose leaves the wrist about 200 mm in front of the robot's own base, folded,
  and driving straight back out from there takes the damped-least-squares IK
  through a near-singularity. Measured, it swings the shoulder 74 degrees, drives
  the elbow into its limit, and levers the module out of the pads. Retracing a
  path the arm has already flown is feasible by construction;
* the module is held by real pad-against-pin contact throughout. There is no
  fixed joint and no software fixture in this scene.

The policies are loaded straight from their checkpoints rather than through
RL-Games, because three players in one process would each need their own vector
environment. The network is a three-layer MLP and its observation normaliser is
in the same file, so running it directly is both simpler and easier to audit.

**One run of this is a demonstration. It is not evidence.** With
``--episode_metrics`` the same driver runs many environments in parallel,
headless, and writes one row per completed workflow in exactly the format
``scripts/aggregate_evaluation.py`` pools, so a chained run can be gated and
reported with a Wilson interval like any other claim in this repository. Success
there is re-checked after a settling window rather than taken at the instant a
predicate fires; see ``_workflow_outcome``.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."

TASK = "Isaac-ZeroG-Blade-GrapplePin-Workflow-v0"
#: Control steps between recorded transit waypoints. Four is about 30 mm of pull
#: at the extraction scale, close enough that the return follows the same arc.
TRANSIT_WAYPOINT_STRIDE = 4
#: Control steps spent letting the closure drive the pin against its collar
#: before the pull starts. One second, which is what the pull gate needed to
#: settle and what the extract task's own action term waits out.
SEAT_STEPS = 30
#: Grip error the capture must reach before extraction takes over. The seating
#: feed adds about 3 mm on top, landing near the 12.4 mm the extract task starts
#: its own episodes from.
HANDOVER_GRIP_M = 0.010

#: Phases, as integers, because the driver runs them per environment in parallel.
CAPTURE, SEAT, EXTRACT, TRANSIT, INSERT, DONE = range(6)
PHASE_NAMES = ("capture", "seat", "extract", "transit", "insert", "done")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp_checkpoint", type=Path, required=True)
    parser.add_argument("--extract_checkpoint", type=Path, required=True)
    parser.add_argument("--insert_checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4070)
    parser.add_argument(
        "--curriculum_stage",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Reset distance for the capture. 2 is the widest pose error the grasp policy trained on.",
    )
    parser.add_argument(
        "--workflow",
        choices=("remove", "install", "full"),
        default="install",
        help=(
            "remove: capture a fully installed module and pull it clear of the rack, both learned. "
            "install: capture a module at the rack mouth and seat it, both learned. "
            "full: remove, fly back, and re-install. The return leg is blocked by the pin's yaw limitation, "
            "not by the controller: a single-point pin does not constrain rotation once the rails release the "
            "module, and the grip degrades from 15 mm to 35 mm during the return whatever speed it is flown at."
        ),
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Run this many workflows in parallel. Above one the driver is headless and per-environment.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Stop after this many completed workflows. Zero runs --steps once, which is the demonstration path.",
    )
    parser.add_argument(
        "--episode_metrics",
        type=Path,
        default=None,
        help="Write one row per completed workflow here (.npz), in the format aggregate_evaluation.py pools.",
    )
    parser.add_argument(
        "--inspection_view",
        choices=("task", "grasp", "side", "top", "workcell"),
        default="side",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_dir", type=Path, default=Path("artifacts/demo/workflow"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/demo/workflow_report.json"))
    parser.add_argument(
        "--transit_slowdown",
        type=int,
        default=3,
        help=(
            "Replay the return path this many times slower than the pull. A single-point pin does not constrain "
            "yaw once the rails release the module, so a full-speed replay rotates it in the pads."
        ),
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=0,
        help="Extra steps to hold still after the workflow finishes, so a recording does not cut on the last frame.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.num_envs < 1:
    parser.error("--num_envs must be positive")
if args.episodes < 0:
    parser.error("--episodes must be non-negative")
if args.video and args.num_envs > 1:
    parser.error("--video records one workflow; use --num_envs 1")
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.evaluation import (
    TERMINAL_METRIC_FIELDS,
    TERMINATION_REASONS,
    TerminalEpisodeRecorder,
    round_floats,
    summarize_terminal_episodes,
    wilson_interval,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.grapple import (
    EXTRACTION_ANGULAR_VELOCITY_LIMIT,
    EXTRACTION_LINEAR_VELOCITY_LIMIT,
    WORKFLOW_SETTLE_S,
    capture_established,
    extraction_success_mask,
    grapple_grip_error_metrics,
    grapple_insertion_conditions,
    grapple_insertion_success_mask,
    grip_drive_torque,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import (
    attached_blade_velocity,
    insertion_error_metrics,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world
from zero_g_blade_swap.tasks.blade_swap.grapple_pin_env_cfg import (
    ZeroGBladeGrapplePinExtractEnvCfg,
    ZeroGBladeGrapplePinGraspEnvCfg,
    ZeroGBladeGrapplePinInsertEnvCfg,
)
from zero_g_blade_swap.tasks.blade_swap.workflow_demo_env_cfg import (
    EXTRACT_ACTION_SCALE,
    GRASP_ACTION_SCALE,
    INSERT_ACTION_SCALE,
    TRANSIT_TARGET_BLADE_X,
)
from zero_g_blade_swap.grapple_geometry import EXTRACTED_BLADE_CENTRE_X

#: Held still after the workflow's own predicate fires, before the outcome is
#: judged. A success that evaporates in two thirds of a second was not one.
#: Read from the skill module, because the extraction velocity limits are
#: *derived* from this window and the two must not be able to disagree.
SETTLE_STEPS = round(WORKFLOW_SETTLE_S * 30.0)

def _certified_episode_length_s(cfg_class: type) -> float:
    """Read a skill's episode length off its own task configuration.

    Through the dataclass field rather than the class attribute, because
    ``configclass`` rewrites these into fields and the attribute is not there to
    read. Going through the task class at all is the point: a constant copied
    here would be free to drift away from what the skill is certified on, which
    is the exact failure this budget exists to prevent.
    """

    for field in dataclasses.fields(cfg_class):
        if field.name == "episode_length_s":
            if field.default is not dataclasses.MISSING:
                return float(field.default)
            return float(field.default_factory())
    raise AttributeError(f"{cfg_class.__name__} declares no episode_length_s")


#: Seconds each phase gets, read from the task each policy was certified on so
#: the two can never drift apart. This is the reconciliation between per-skill
#: certification and the chain: before it, a skill was certified on a 12 s
#: episode and then quoted as "it completes in the chain", where the chain
#: happened to grant 45 s. Whichever number read better was the one being
#: quoted. Now the chain gives each skill exactly the clock its own
#: certification gives it, and a phase that overruns fails the workflow.
PHASE_BUDGET_S = (
    _certified_episode_length_s(ZeroGBladeGrapplePinGraspEnvCfg),  # capture
    SEAT_STEPS / 30.0,  # seat, scripted
    _certified_episode_length_s(ZeroGBladeGrapplePinExtractEnvCfg),  # extract
    _certified_episode_length_s(ZeroGBladeGrapplePinExtractEnvCfg),  # transit, replays the pull
    _certified_episode_length_s(ZeroGBladeGrapplePinInsertEnvCfg),  # insert
    float("inf"),  # done, nothing left to time
)

#: Extra per-episode columns this driver records on top of the shared row.
#: ``aggregate_evaluation.py`` aligns runs by column name, so a chained run can
#: be pooled beside a single-skill one without either losing a field.
WORKFLOW_METRIC_FIELDS = (
    *TERMINAL_METRIC_FIELDS,
    "blade_centre_x_m",
    "reached_phase",
    "timed_out_in_phase",
    "grip_error_m",
    "grip_attitude_rad",
    "predicate_fired",
    "all_conditions_after_settling",
)


class CheckpointPolicy:
    """One RL-Games actor, loaded without RL-Games.

    Reproduces exactly what a deterministic player does: clip the observation to
    the configured range, apply the running mean/variance normaliser the policy
    was trained with, run the trunk, and take the mean action.
    """

    def __init__(self, path: Path, device: str, clip_observations: float = 10.0) -> None:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        weights = checkpoint["model"]
        self.device = device
        self.clip_observations = clip_observations
        self.mean = weights["running_mean_std.running_mean"].to(device).float()
        self.variance = weights["running_mean_std.running_var"].to(device).float()
        self.observation_dim = int(self.mean.shape[0])

        layers: list[nn.Module] = []
        index = 0
        while f"a2c_network.trunk.{index}.weight" in weights:
            weight = weights[f"a2c_network.trunk.{index}.weight"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(weights[f"a2c_network.trunk.{index}.bias"])
            layers.extend((layer, nn.ELU()))
            index += 2
        self.trunk = nn.Sequential(*layers).to(device).eval()

        mu_weight = weights["a2c_network.mu.weight"]
        self.mu = nn.Linear(mu_weight.shape[1], mu_weight.shape[0]).to(device)
        self.mu.weight.data.copy_(mu_weight)
        self.mu.bias.data.copy_(weights["a2c_network.mu.bias"])
        self.mu.eval()
        self.action_dim = int(mu_weight.shape[0])
        self.epoch = int(checkpoint.get("epoch", -1))
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest().upper()

    @torch.inference_mode()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise RuntimeError(
                f"{self.path.name} expects {self.observation_dim} observation values, "
                f"received {observation.shape[-1]}. The observation group does not match the policy."
            )
        clipped = observation.clamp(-self.clip_observations, self.clip_observations)
        normalized = ((clipped - self.mean) / torch.sqrt(self.variance + 1.0e-5)).clamp(-5.0, 5.0)
        return self.mu(self.trunk(normalized)).clamp(-1.0, 1.0)


def _blade_centre_x(task) -> torch.Tensor:
    return task.scene["spare_blade"].data.root_pos_w[:, 0] - task.scene.env_origins[:, 0]


def _extraction_held(task) -> torch.Tensor:
    """The extract skill's own success predicate, re-checked.

    Clear of the mouth, still gripped, and no longer moving. A pull that ends
    with the module tumbling free of the pads has not removed anything.
    """

    velocity = attached_blade_velocity(task)
    return (
        (_blade_centre_x(task) <= EXTRACTED_BLADE_CENTRE_X)
        & capture_established(task)
        # Read from the skill, never restated. These limits are derived from
        # this driver's own settling window -- a module still moving at v drifts
        # v * 0.70 s before it is judged -- so a copy here that drifted from the
        # skill's would let the two disagree about the very quantity the settle
        # exists to test. Two constants restated instead of read have already
        # cost this workflow a full certification each.
        & (torch.linalg.vector_norm(velocity[:, :3], dim=-1) <= EXTRACTION_LINEAR_VELOCITY_LIMIT)
        & (torch.linalg.vector_norm(velocity[:, 3:], dim=-1) <= EXTRACTION_ANGULAR_VELOCITY_LIMIT)
    )


#: The five conditions that say the module is seated in the rack. The remaining
#: two conditions ``grapple_insertion_conditions`` returns, ``grasp_position``
#: and ``grasp_orientation``, describe the *gripper's* hold on the pin, which is
#: about to be released anyway; they are recorded separately rather than dropped.
SEATED_CONDITIONS = ("axial_depth", "lateral_alignment", "orientation", "linear_velocity", "angular_velocity")


def _workflow_outcome(task, workflow: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (the workflow's own condition, every condition including the grip).

    Judged live, so the driver can apply it after a settling window rather than
    at the instant a predicate first fires. The install workflow needs the
    distinction: its success predicate fires correctly and then the pin relaxes a
    couple of hundredths of a radian in the pads, which trips the grip-retention
    check while the module stays exactly where it was put.
    """

    if workflow == "remove":
        held = _extraction_held(task)
        return held, held
    conditions = grapple_insertion_conditions(task)
    seated = torch.stack([conditions[name] for name in SEATED_CONDITIONS], dim=-1).all(dim=-1)
    everything = torch.stack(tuple(conditions.values()), dim=-1).all(dim=-1)
    return seated, everything


class WorkflowDriver:
    """The phase machine, per environment.

    One instance drives ``num_envs`` workflows at once. Everything that was a
    Python scalar in the single-environment demonstration is a tensor here, and
    every transition is a mask, because a fleet of workflows do not reach their
    hand-offs on the same step.
    """

    def __init__(self, task, policies, workflow: str, transit_slowdown: int, max_steps: int) -> None:
        self.task = task
        self.policies = policies
        self.workflow = workflow
        self.transit_slowdown = max(1, transit_slowdown)
        device = task.device
        count = task.num_envs
        self.phase = torch.full((count,), CAPTURE, dtype=torch.long, device=device)
        self.phase_started = torch.zeros(count, dtype=torch.long, device=device)
        # Only the learned phases are held to a deadline. The two scripted ones
        # run for a length fixed by construction -- the seat is exactly
        # SEAT_STEPS, the transit is exactly the recorded path replayed -- so a
        # deadline there could only ever fire on an off-by-one, and the episode
        # length still bounds them.
        budgets = [
            float("inf") if index in (SEAT, TRANSIT, DONE) else budget
            for index, budget in enumerate(PHASE_BUDGET_S)
        ]
        self.phase_deadline = torch.tensor(
            [float("inf") if budget == float("inf") else round(budget / float(task.step_dt)) for budget in budgets],
            dtype=torch.float64,
            device=device,
        )
        self.held = torch.zeros(count, dtype=torch.long, device=device)
        self.seat_until = torch.zeros(count, dtype=torch.long, device=device)
        self.done_at = torch.full((count,), -1, dtype=torch.long, device=device)
        #: Control steps *into its own episode* at which the workflow finished.
        #: ``done_at`` counts driver steps, which keep running across episodes, so
        #: it cannot be used for cycle time: the second batch of environments
        #: would report a cycle time of an episode and a half.
        self.done_steps = torch.zeros(count, dtype=torch.long, device=device)
        #: Furthest phase this workflow reached, kept separately because ``phase``
        #: reads DONE whether the workflow finished or ran out of clock.
        self.furthest = torch.full((count,), CAPTURE, dtype=torch.long, device=device)
        #: Which phase overran its skill's own episode budget, or -1.
        self.timed_out_in = torch.full((count,), -1, dtype=torch.long, device=device)
        self.transit_started = torch.zeros(count, dtype=torch.long, device=device)
        self.predicate_fired = torch.zeros(count, dtype=torch.bool, device=device)
        self.judged = torch.zeros(count, dtype=torch.bool, device=device)
        self.outcome = torch.zeros(count, dtype=torch.bool, device=device)
        self.all_conditions = torch.zeros(count, dtype=torch.bool, device=device)
        # Tool positions visited during the pull, sampled so the transit can fly
        # them backwards. Every one of them was reachable a moment ago.
        self.max_waypoints = max(1, max_steps // TRANSIT_WAYPOINT_STRIDE + 2)
        self.waypoints = torch.zeros((self.max_waypoints, count, 3), device=device)
        self.waypoint_write = torch.zeros(count, dtype=torch.long, device=device)
        self.waypoint_read = torch.zeros(count, dtype=torch.long, device=device)
        self.frozen = torch.zeros((count, len(WORKFLOW_METRIC_FIELDS)), dtype=torch.float64, device=device)
        self.frozen_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.required_hold = max(1, int(round(0.30 / float(task.step_dt))))
        self.scales = torch.tensor(
            [
                GRASP_ACTION_SCALE,
                GRASP_ACTION_SCALE,
                EXTRACT_ACTION_SCALE,
                EXTRACT_ACTION_SCALE,
                INSERT_ACTION_SCALE,
                INSERT_ACTION_SCALE,
            ],
            device=device,
        )
        self.arm = task.action_manager.get_term("arm")
        self.gripper = task.action_manager.get_term("gripper")
        self.actions = torch.zeros((count, task.action_manager.total_action_dim), device=device)

    def reset_envs(self, env_ids: torch.Tensor, step: int = 0) -> None:
        """Return the named environments to the start of the workflow."""

        self.phase[env_ids] = CAPTURE
        self.phase_started[env_ids] = step
        self.furthest[env_ids] = CAPTURE
        self.timed_out_in[env_ids] = -1
        self.held[env_ids] = 0
        self.seat_until[env_ids] = 0
        self.done_at[env_ids] = -1
        self.done_steps[env_ids] = 0
        self.transit_started[env_ids] = 0
        self.predicate_fired[env_ids] = False
        self.judged[env_ids] = False
        self.outcome[env_ids] = False
        self.all_conditions[env_ids] = False
        self.waypoint_write[env_ids] = 0
        self.waypoint_read[env_ids] = 0
        self.frozen_valid[env_ids] = False

    def _apply_scales(self) -> None:
        self.arm._scale[:] = self.scales[self.phase]

    def _finish(self, mask: torch.Tensor, step: int) -> None:
        """Stop the named workflows, recording when they stopped, two ways.

        ``done_at`` is a driver step and drives the settling timer; ``done_steps``
        is the count inside the environment's own episode and is what cycle time
        is reported from. They are not the same number once a second batch of
        episodes starts, and conflating them reports a cycle time longer than the
        episode.
        """

        self.phase[mask] = DONE
        self.done_at[mask] = step
        self.done_steps[mask] = self.task.episode_length_buf[mask].to(torch.long) + 1

    def step(self, step: int) -> None:
        """Compute one action for every environment and advance the phase machine."""

        task = self.task
        observations = task.observation_manager.compute()
        phase = self.phase
        entry_phase = self.phase.clone()
        self.actions.zero_()

        # --- every phase gets the clock its own skill was certified on ---------
        # Checked before anything else this step, so a hand-off that lands one
        # control step past the budget fails here rather than being waved
        # through. Without this the chain and the per-skill certification
        # disagree about what the task is, and the temptation is to quote
        # whichever number reads better: measured, the insert skill scores 6.96%
        # on its own 12 s episode and "completes in the chain", where the chain
        # used to grant 45 s.
        elapsed = (step - self.phase_started).to(torch.float64)
        overrun = (elapsed >= self.phase_deadline[self.phase]) & (self.phase != DONE)
        if bool(overrun.any()):
            self.timed_out_in[overrun] = self.phase[overrun]
            self._finish(overrun, step)

        capturing = phase == CAPTURE
        if bool(capturing.any()):
            command = self.policies["capture"].act(observations["grasp"])
            self.actions[capturing] = command[capturing]
        for name, group, mask in (
            ("extract", "extract", phase == EXTRACT),
            ("insert", "insert", phase == INSERT),
        ):
            if bool(mask.any()):
                command = self.policies[name].act(observations[group])
                self.actions[mask, :6] = command[mask]
        # Everything past the capture keeps commanding closure, so the two-stage
        # action term holds the pin instead of relaxing to the capture command.
        self.actions[~capturing, 6] = 1.0

        grip_error, grip_attitude = grapple_grip_error_metrics(task)
        established = capture_established(task)
        tool = end_effector_pose_world(task)[0]
        blade_x = _blade_centre_x(task)

        # --- capture -> seat ------------------------------------------------
        # Hand over on the *next* skill's precondition, not this one's success
        # criterion. The grasp task counts a capture from 20 mm of grip error,
        # and the extract policy has never started from worse than about 12 mm,
        # so handing over at the first qualifying instant puts it 10 mm out of
        # distribution and it reverses into the rack. The grasp policy keeps
        # closing to a 9-to-12 mm median if simply allowed to finish.
        qualifying = capturing & established & (grip_error <= HANDOVER_GRIP_M)
        self.held = torch.where(qualifying, self.held + 1, torch.where(capturing, torch.zeros_like(self.held), self.held))
        promote = capturing & (self.held >= self.required_hold)
        if bool(promote.any()):
            # Latch the holding closure, per environment. TwoStageRobotiqAction
            # drops back to the gentler capture command whenever the grip error
            # exceeds its tolerance, which is right while capturing and
            # catastrophic afterwards: measured, the error drifts past 20 mm
            # during the pull, the fingers open by about 21 mm, and the module is
            # released in mid-transit. A real servicer does not relax its grip
            # once the part is taken.
            self.gripper.hold_latch[promote] = True
            self.phase[promote] = SEAT
            self.seat_until[promote] = step + SEAT_STEPS

        # --- seat -> extract or insert --------------------------------------
        # Hold the arm still and let the closure drive the pin against the
        # collar. The extract skill gets this for free: its task resets with the
        # fingers apart and runs the two-stage capture inside a 1.0 s window
        # while the action term holds the arm, so it only ever sees a *seated*
        # grip. The grasp policy declares capture as soon as the pads are loaded,
        # which is measurably earlier -- finger angle 0.085 against the 0.223 the
        # pin sits at once it is home. Handing that shallow grip to extraction
        # lets the wedge cam the module round, and the attitude error grows from
        # 0.03 to 0.40 rad until the policy is out of distribution.
        seated = (self.phase == SEAT) & (step >= self.seat_until)
        if bool(seated.any()):
            self.phase[seated] = INSERT if self.workflow == "install" else EXTRACT

        # --- extract -> clear ------------------------------------------------
        extracting = self.phase == EXTRACT
        if step % TRANSIT_WAYPOINT_STRIDE == 0 and bool(extracting.any()):
            ids = torch.nonzero(extracting, as_tuple=False).squeeze(-1)
            slots = self.waypoint_write[ids].clamp(max=self.max_waypoints - 1)
            self.waypoints[slots, ids] = tool[ids]
            self.waypoint_write[ids] = (self.waypoint_write[ids] + 1).clamp(max=self.max_waypoints - 1)
        # The extract skill's own success mask, not merely "past the line": it
        # also asks that the module is still gripped and no longer moving, which
        # is what the skill was certified on.
        cleared = extracting & extraction_success_mask(task)
        if bool(cleared.any()):
            if self.workflow == "remove":
                self._finish(cleared, step)
                self.predicate_fired[cleared] = True
            else:
                self.phase[cleared] = TRANSIT
                self.waypoint_read[cleared] = (self.waypoint_write[cleared] - 1).clamp_min(0)
                self.transit_started[cleared] = step

        # --- transit ---------------------------------------------------------
        # Scripted, and the only segment no policy drives: fly the tool back
        # along the waypoints the extraction just visited. Closing the loop on
        # position each step, so a waypoint that is slightly off does not
        # accumulate. Walk the recorded path backwards on the clock, not on
        # proximity: advancing only when close stalls, because the last waypoint
        # was sampled up to a stride before the hand-off, so the tool is already
        # past it and the follower sits there driving the module further out.
        transiting = self.phase == TRANSIT
        if bool(transiting.any()):
            due = transiting & (
                ((step - self.transit_started) % (TRANSIT_WAYPOINT_STRIDE * self.transit_slowdown)) == 0
            )
            self.waypoint_read[due] = (self.waypoint_read[due] - 1).clamp_min(0)
            ids = torch.nonzero(transiting, as_tuple=False).squeeze(-1)
            target = self.waypoints[self.waypoint_read[ids], ids]
            scale = self.scales[TRANSIT][:3]
            self.actions[ids, :3] = ((target - tool[ids]) / scale).clamp(-1.0, 1.0)
            self.actions[ids, 3:6] = 0.0
            arrived = transiting & (self.waypoint_read <= 0) & (blade_x >= TRANSIT_TARGET_BLADE_X - 0.005)
            self.phase[arrived] = INSERT

        # --- insert -> seated --------------------------------------------------
        inserting = self.phase == INSERT
        if bool(inserting.any()):
            # The insert skill's own success mask, including its 0.20 s hold, so
            # the chain is judged by the same predicate the skill is.
            fired = inserting & grapple_insertion_success_mask(task)
            self._finish(fired, step)
            self.predicate_fired[fired] = True

        # --- done: hold still, then judge --------------------------------------
        finished = self.phase == DONE
        if bool(finished.any()):
            self.actions[finished, :6] = 0.0
            self.actions[finished, 6] = 1.0
            ripe = finished & ~self.judged & (step >= self.done_at + SETTLE_STEPS)
            if bool(ripe.any()):
                outcome, everything = _workflow_outcome(task, self.workflow)
                # Both halves are required. The predicate is the skill's own
                # criterion and the re-check is this driver's; a workflow that
                # ran out of clock and happens to be sitting in tolerance
                # afterwards has not completed, and counting it would be a
                # success threshold quietly relaxed.
                outcome = outcome & self.predicate_fired
                everything = everything & self.predicate_fired
                self.outcome[ripe] = outcome[ripe]
                self.all_conditions[ripe] = everything[ripe]
                self.judged[ripe] = True
                self._freeze(ripe, step, grip_error, grip_attitude, blade_x)

        torch.maximum(self.furthest, self.phase, out=self.furthest)
        self.phase_started[self.phase != entry_phase] = step
        self._apply_scales()

    def _freeze(
        self,
        mask: torch.Tensor,
        step: int,
        grip_error: torch.Tensor,
        grip_attitude: torch.Tensor,
        blade_x: torch.Tensor,
    ) -> None:
        """Store the row for environments whose workflow has just been judged.

        Frozen at the moment of judgement rather than read at reset, because a
        completed workflow then idles for the rest of the episode and the state
        at the timeout is not the state that was achieved.
        """

        rows = self._rows(step, grip_error, grip_attitude, blade_x)
        self.frozen[mask] = rows[mask]
        self.frozen_valid[mask] = True

    def _rows(
        self,
        step: int,
        grip_error: torch.Tensor,
        grip_attitude: torch.Tensor,
        blade_x: torch.Tensor,
    ) -> torch.Tensor:
        task = self.task
        axial, lateral, orientation = insertion_error_metrics(task)
        velocity = attached_blade_velocity(task)
        # Cycle time is time to the hand-off that finished the workflow, not to
        # the episode's timeout: a completed workflow then idles, and reporting
        # the idle would make every success look like it took the whole episode.
        steps = torch.where(
            self.done_steps > 0, self.done_steps, task.episode_length_buf.to(self.done_steps.dtype)
        ).to(torch.float64)
        success = self.outcome.to(torch.float64)
        reason = torch.where(
            self.outcome,
            torch.full_like(success, TERMINATION_REASONS.index("extraction_success"))
            if self.workflow == "remove"
            else torch.full_like(success, TERMINATION_REASONS.index("insertion_success")),
            torch.full_like(success, TERMINATION_REASONS.index("time_out")),
        )
        columns = {
            "success": success,
            "termination_reason": reason,
            "curriculum_stage": torch.full_like(success, float(args.curriculum_stage)),
            "control_steps": steps,
            "cycle_time_s": steps * float(task.step_dt),
            "axial_error_m": axial.to(torch.float64),
            "lateral_error_m": lateral.to(torch.float64),
            "orientation_error_rad": orientation.to(torch.float64),
            "blade_linear_velocity_mps": torch.linalg.vector_norm(velocity[:, :3], dim=-1).to(torch.float64),
            "blade_angular_velocity_radps": torch.linalg.vector_norm(velocity[:, 3:], dim=-1).to(torch.float64),
            # Nothing welds the module to the tool in this scene, so these two are
            # a real measurement of whether the grip is still there rather than
            # the tautology they are on the rigid-grasp task.
            "tool_to_handle_error_m": grip_error.to(torch.float64),
            "tool_to_handle_orientation_rad": grip_attitude.to(torch.float64),
            "blade_centre_x_m": blade_x.to(torch.float64),
            "reached_phase": self.furthest.to(torch.float64),
            "timed_out_in_phase": self.timed_out_in.to(torch.float64),
            "grip_error_m": grip_error.to(torch.float64),
            "grip_attitude_rad": grip_attitude.to(torch.float64),
            "predicate_fired": self.predicate_fired.to(torch.float64),
            "all_conditions_after_settling": self.all_conditions.to(torch.float64),
        }
        return torch.stack([columns[name] for name in WORKFLOW_METRIC_FIELDS], dim=-1)

    def harvest(self, env_ids: torch.Tensor, step: int) -> torch.Tensor:
        """Rows for environments about to be reset, terminal state still intact.

        An environment that finished its workflow with less than the settling
        window left on the clock has never been judged, so judge it here rather
        than record it as a failure it did not commit. It still has to satisfy
        the same condition; it simply gets asked at the last moment available.
        """

        outcome, everything = _workflow_outcome(self.task, self.workflow)
        pending = ~self.judged
        self.outcome = torch.where(pending, outcome & self.predicate_fired, self.outcome)
        self.all_conditions = torch.where(pending, everything & self.predicate_fired, self.all_conditions)
        grip_error, grip_attitude = grapple_grip_error_metrics(self.task)
        blade_x = _blade_centre_x(self.task)
        live = self._rows(step, grip_error, grip_attitude, blade_x)
        rows = torch.where(self.frozen_valid.unsqueeze(-1), self.frozen, live)
        return rows[env_ids]


def _chain_report(recorder, workflow: str) -> dict[str, object]:
    """Pool the recorded workflows into a gated summary.

    Deliberately the same statistics the single-skill evaluator reports, because
    a chained run has to be judged the same way a skill is or the comparison
    between them means nothing.
    """

    rows = recorder.rows
    fields = recorder.fields
    summary = summarize_terminal_episodes(rows, fields)
    episodes = int(rows.shape[0])
    column = {name: rows[:, fields.index(name)] for name in fields}
    fired = int((column["predicate_fired"] > 0.5).sum())
    everything = int((column["all_conditions_after_settling"] > 0.5).sum())
    phases = {
        PHASE_NAMES[index]: int((column["reached_phase"] == index).sum())
        for index in range(len(PHASE_NAMES))
        if int((column["reached_phase"] == index).sum())
    }
    low, high = wilson_interval(int(summary["successes"]), episodes)
    return {
        "workflow": workflow,
        "episodes": episodes,
        "successes": int(summary["successes"]),
        "success_rate": summary["success_rate"],
        "success_rate_wilson_95": {"low": low, "high": high},
        # The three numbers that have to be reported together. A predicate that
        # fires and then evaporates is not a completed workflow, and the gap
        # between the second and third is the pin relaxing in the pads *after*
        # the module is already seated, which is not the module coming loose.
        "predicate_fired": fired,
        "held_after_settling": int(summary["successes"]),
        "every_condition_after_settling": everything,
        "settle_seconds": round(SETTLE_STEPS / 30.0, 3),
        "furthest_phase_reached": phases,
        # A phase that overran the episode length its own skill was certified on.
        # This is the number that reconciles the chain with per-skill evidence.
        "phase_budget_overruns": {
            PHASE_NAMES[index]: int((column["timed_out_in_phase"] == index).sum())
            for index in range(len(PHASE_NAMES))
            if int((column["timed_out_in_phase"] == index).sum())
        },
        "phase_budget_s": {PHASE_NAMES[index]: PHASE_BUDGET_S[index] for index in range(len(PHASE_NAMES) - 1)},
        "termination_reasons": summary["termination_reasons"],
        "instability_terminations": summary["instability_terminations"],
        "non_finite_metric_episodes": summary["non_finite_metric_episodes"],
        "terminal_metrics": summary["terminal_metrics"],
    }


def main() -> dict[str, object]:
    env = None
    try:
        device = args.device or "cuda:0"
        env_cfg = parse_env_cfg(TASK, device=device, num_envs=args.num_envs)
        env_cfg.configure_robustness(0)
        env_cfg.seed = args.seed
        # Derive the episode from the phases this workflow actually runs, each at
        # the budget its own skill was certified on, instead of one round number
        # generous enough to hide an overrun. It is also what makes a many-seed
        # sweep affordable: an install that finishes at 14 s no longer idles to
        # 45 s before its environment resets.
        phases = [CAPTURE, SEAT]
        if args.workflow != "install":
            phases.append(EXTRACT)
        if args.workflow == "full":
            phases.append(TRANSIT)
        if args.workflow != "remove":
            phases.append(INSERT)
        budget = sum(
            PHASE_BUDGET_S[index] * (args.transit_slowdown if index == TRANSIT else 1) for index in phases
        )
        env_cfg.episode_length_s = round(budget + SETTLE_STEPS / 30.0 + 1.0, 2)
        print(
            f"[INFO] {args.workflow}: phase budgets "
            f"{[f'{PHASE_NAMES[i]}={PHASE_BUDGET_S[i] * (args.transit_slowdown if i == TRANSIT else 1):.1f}s' for i in phases]} "
            f"-> episode {env_cfg.episode_length_s} s",
            flush=True,
        )
        inspection_views = {
            "grasp": ((0.18, -1.05, 1.02), (0.52, 0.0, 0.72)),
            "side": ((0.52, -1.30, 0.86), (0.50, 0.0, 0.72)),
            "top": ((0.50, -0.05, 1.60), (0.50, 0.0, 0.70)),
            "workcell": ((-0.50, -1.80, 1.25), (0.45, 0.0, 0.72)),
        }
        if args.inspection_view in inspection_views:
            env_cfg.viewer.eye, env_cfg.viewer.lookat = inspection_views[args.inspection_view]

        policies = {
            "capture": CheckpointPolicy(args.grasp_checkpoint, device),
            "extract": CheckpointPolicy(args.extract_checkpoint, device),
            "insert": CheckpointPolicy(args.insert_checkpoint, device),
        }
        for name, policy in policies.items():
            print(
                f"[INFO] {name:8s} obs={policy.observation_dim:3d} act={policy.action_dim} "
                f"epoch={policy.epoch} <- {policy.path.name}",
                flush=True,
            )

        env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        task = env.unwrapped
        # The reset event picks the arm and blade pose from this buffer, so it has
        # to exist and hold the wanted stage *before* the first reset. Nothing has
        # created it yet: with no curriculum term, only the reset itself would,
        # and by then it would have already chosen stage 0.
        task._insertion_curriculum_stage = torch.full(
            (task.num_envs,), args.curriculum_stage, dtype=torch.long, device=task.device
        )
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(args.video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args.steps,
                disable_logger=True,
            )
        env.reset()
        task._insertion_curriculum_stage.fill_(args.curriculum_stage)

        episode_steps = int(round(float(task.max_episode_length)))
        driver = WorkflowDriver(task, policies, args.workflow, args.transit_slowdown, episode_steps)
        collecting = args.episodes > 0
        recorder = TerminalEpisodeRecorder(WORKFLOW_METRIC_FIELDS) if collecting else None
        clock = {"step": 0}

        def on_reset(_env, env_ids) -> None:
            # Called from _reset_idx while the scene still holds the finished
            # episode's terminal state, which is the only moment it can be read.
            if recorder is not None:
                recorder.record(driver.harvest(env_ids, clock["step"]).cpu().numpy())
            driver.reset_envs(env_ids, clock["step"])

        task.enable_terminal_metrics(on_reset)

        timeline: list[dict[str, object]] = []

        def note(event: str, step: int) -> None:
            blade_x = float(_blade_centre_x(task)[0])
            grip, attitude = grapple_grip_error_metrics(task)
            entry = {
                "event": event,
                "step": step,
                "time_s": round(step * float(task.step_dt), 3),
                "blade_centre_x_m": blade_x,
                "grip_error_m": float(grip[0]),
                "grip_attitude_rad": float(attitude[0]),
                "drive_torque_nm": float(grip_drive_torque(task)[0]),
            }
            timeline.append(entry)
            print(
                f"[PHASE] {event:26s} step {step:4d}  t={entry['time_s']:6.2f}s  "
                f"blade_x={blade_x:.4f}  grip={entry['grip_error_m'] * 1000:6.2f}mm  "
                f"torque={entry['drive_torque_nm']:5.2f}Nm",
                flush=True,
            )

        single = task.num_envs == 1 and not collecting
        if single:
            note("start:capture", 0)

        step = 0
        # A collecting run has no step budget of its own: it stops on the episode
        # count, and every environment is reset on the task's own timeout.
        budget = args.steps if not collecting else episode_steps * (args.episodes // task.num_envs + 2)
        progress_every = max(1, episode_steps // 3)
        while step < budget:
            if collecting and len(recorder) >= args.episodes:
                break
            clock["step"] = step
            previous = int(driver.phase[0]) if single else 0
            driver.step(step)
            if single and int(driver.phase[0]) != previous:
                note(f"{PHASE_NAMES[previous]} -> {PHASE_NAMES[int(driver.phase[0])]}", step)
            _, _, terminated, truncated, _ = env.step(driver.actions)
            step += 1
            if collecting and step % progress_every == 0:
                counts = {PHASE_NAMES[index]: int((driver.phase == index).sum()) for index in range(len(PHASE_NAMES))}
                print(
                    f"[CHAIN] step {step:5d}  episodes={len(recorder):4d}/{args.episodes}  {counts}",
                    flush=True,
                )
            if single:
                judged = int(driver.phase[0]) == DONE and bool(driver.judged[0])
                if judged and step >= int(driver.done_at[0]) + SETTLE_STEPS + args.settle_steps:
                    break
                if bool(terminated[0] or truncated[0]):
                    note(f"episode ended during {PHASE_NAMES[int(driver.phase[0])]}", step)
                    break

        combined = (
            hashlib.sha256("".join(policies[name].sha256 for name in ("capture", "extract", "insert")).encode())
            .hexdigest()
            .upper()
        )
        result: dict[str, object] = {
            "task": TASK,
            "workflow": args.workflow,
            "seed": args.seed,
            "num_envs": task.num_envs,
            "curriculum_stage": args.curriculum_stage,
            "checkpoints": {name: str(policy.path) for name, policy in policies.items()},
            "checkpoint_sha256": {name: policy.sha256 for name, policy in policies.items()},
            "policy_set_sha256": combined,
            "learned_phases": {
                "remove": ["capture", "extract"],
                "install": ["capture", "insert"],
                "full": ["capture", "extract", "insert"],
            }[args.workflow],
            "scripted_phases": ["seat"] + (["transit"] if args.workflow == "full" else []),
            "success_definition": (
                "the workflow's own condition re-checked after a "
                f"{SETTLE_STEPS / 30.0:.2f} s settling window, not the instant a predicate fired"
            ),
        }
        if collecting:
            result["chain"] = _chain_report(recorder, args.workflow)
            print(json.dumps(round_floats(result["chain"]), indent=2)[:3000], flush=True)
            if args.episode_metrics is not None:
                args.episode_metrics.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.episode_metrics,
                    rows=np.asarray(recorder.rows, dtype=np.float32),
                    fields=np.asarray(recorder.fields),
                    metadata=np.asarray(
                        json.dumps(
                            {
                                "task": TASK,
                                "seed": args.seed,
                                "curriculum_stage": args.curriculum_stage,
                                "robustness_level": getattr(env_cfg, "robustness_level", None),
                                "checkpoint": f"chained {args.workflow}: capture+extract+insert",
                                # One digest over all three policies, so pooling
                                # runs driven by different checkpoints fails loudly
                                # in aggregate_evaluation.py exactly as it does for
                                # a single skill.
                                "checkpoint_sha256": combined,
                                "checkpoints": {name: policy.sha256 for name, policy in policies.items()},
                                "num_envs": task.num_envs,
                                "contact_force_limit_n": None,
                                "workflow": args.workflow,
                                "stress": {"pose_noise_scale": 1.0, "out_of_distribution": False},
                            }
                        )
                    ),
                )
                print(f"[INFO] Wrote {args.episode_metrics}", flush=True)
        else:
            axial, lateral, orientation = insertion_error_metrics(task)
            grip, attitude = grapple_grip_error_metrics(task)
            conditions = grapple_insertion_conditions(task)
            result["reached_phase"] = PHASE_NAMES[int(driver.phase[0])]
            result["timeline"] = timeline
            result["final"] = {
                "axial_error_m": float(axial[0]),
                "lateral_error_m": float(lateral[0]),
                "orientation_error_rad": float(orientation[0]),
                "grip_error_m": float(grip[0]),
                "grip_attitude_rad": float(attitude[0]),
                "blade_centre_x_m": float(_blade_centre_x(task)[0]),
            }
            result["insertion_conditions"] = {name: bool(value[0]) for name, value in conditions.items()}
            result["predicate_fired"] = bool(driver.predicate_fired[0])
            result["completed"] = bool(driver.outcome[0])
            result["conditions_still_held_after_settling"] = bool(driver.all_conditions[0])
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
        report = {"task": TASK, "error": f"{type(exc).__name__}: {exc}", "completed": False}
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(round_floats(report), indent=2)[:2000], flush=True)
        simulation_app.close()
