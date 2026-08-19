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
import os
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
#: Control steps the install workflow spends driving the tool back to the pose it
#: started the episode at, before handing over to the insert policy.
#:
#: The insert skill resets with the arm at the certified staging pose and the
#: module at its nominal staging pose. A capture ends somewhere else -- measured
#: over 576 chained installations, the arm sits 0.157 rad from nominal on its
#: worst axis, almost all of it wrist_1, and the module 14.5 mm further out along
#: x at the median with a p95 of 40 mm. The tool is on the pin either way, so the
#: hand-off is well gripped and out of distribution at the same time, which is
#: exactly the pairing that three reset reconstructions failed to reproduce.
#:
#: Rather than teach insert a distribution it has never needed, the chain returns
#: to the taught pose. The module is gripped, so moving the tool moves both back
#: together. This is a scripted segment like the seat and the transit, and it is
#: what an industrial arm does between operations: return to a taught waypoint.
#: **Zero, and measured.** Turning this on trades the wrong way: +2.4 points on
#: the state chain (84.38% -> 86.81%) against -7.5 on the camera arm (80.38% ->
#: 72.92%), and the camera arm is the claim that matters. Kept because the 2x2
#: below is the useful physics, and because a longer free-flight transit may want
#: it. Set ALIGN_STEPS to re-enable.
#:
#:                       hold closure   retain closure
#:      idle pause          21.35%          68.75%
#:      align command       88.54%          85.94%
#:      (no pause at all: 84.90%)
#:
#: Read the top row first: **idling while gripping is catastrophic**, because the
#: wedge thrusts the module the whole time, and the retain closure recovers most
#: of it. Any phase that waits must either command or retain.
ALIGN_STEPS = int(os.environ.get("ALIGN_STEPS", "0"))
#: Control: spend the same steps holding still instead of realigning, so the
#: gain can be attributed. An extra pause and an extra command are different
#: interventions and this repository has been caught conflating them before.
ALIGN_HOLD_ONLY = bool(int(os.environ.get("ALIGN_HOLD_ONLY", "0")))
#: Relax to the retain closure while realigning. **Off, and measured**: the
#: obvious reading of the removal result says the wedge should not be thrusting
#: during a segment whose purpose is to stop the module moving, but on the same
#: seed it scores 85.94% against 88.54% with the holding closure kept. The
#: difference from removal is what happens next -- an installation drives the
#: module back into rails immediately afterwards and wants the firm grip, where
#: a removal is finished and wants to be left alone.
ALIGN_RETAIN = bool(int(os.environ.get("ALIGN_RETAIN", "0")))
#: Relax to the retain closure once an installation has seated the module, for
#: the settling window the outcome is re-checked over.
#:
#: **On, and measured.** This file's operating rule is that any phase which waits
#: must either command or retain, and the DONE phase below waits 0.70 s with the
#: holding closure still commanded. Removal already retains the instant the
#: module is free -- the single change that took chained removal from 0 of 570
#: surviving its re-check to 569 of 576 -- and installation had simply never been
#: asked the same question.
#:
#: The argument against it was real and turned out to be wrong: a seated module
#: is back in its rails, and the rails were expected to absorb the wedge thrust
#: exactly as they do during the seating pause. Measured on seed 4070, same
#: policies, same episodes, one switch:
#:
#:      holding closure through the settle    85.94%   (predicate 88.54%)
#:      retain closure through the settle     90.10%   (predicate 90.63%)
#:
#: Four points, from a rule this project had already written down and applied to
#: one workflow but not the other. The prediction that it could buy at most the
#: one point lost between the predicate and the re-check was also wrong, which is
#: worth keeping: the module keeps being pushed for the whole window, so what it
#: buys is bounded by how far a pushed module drifts, not by the current gap.
SEATED_RETAIN = bool(int(os.environ.get("SEATED_RETAIN", "1")))

#: Hold the module through the relocation transit instead of retaining it.
#:
#: A probe, default off, for the one reading of the transit failure that has not
#: been tested. The operating rule is *capture gently, hold hard to move, retain
#: once free*, and the relocation transit is the first phase in this project that
#: **moves** a module through free space -- the removal chain retains at the end
#: of the job, with the module free and stationary, which is a different state.
#:
#: What points at it is where the slip starts. The retreat leg is 78.25 mm long,
#: and the module trails the tool by 65 mm by the end of it: nearly the whole leg,
#: in the first two seconds, not a degradation accumulated over the 734 mm flight.
#: The retain engages at exactly that moment, and it engages by *reducing* the
#: closure on a module that extraction had been holding firmly.
#:
#: So the retain may be buying the wrong thing here. It exists because a wedge
#: under load is a thruster; the cost is that a wedge under less load is a slide.
#: Which dominates is a measurement, and this switch is how to take it.
RELOCATE_TRANSIT_HOLD = bool(int(os.environ.get("RELOCATE_TRANSIT_HOLD", "0")))

#: Fraction of the rotation channels the relocation transit may use to hold the
#: module's attitude, leaving the rest of the differential IK's authority for
#: actually crossing the rack. Overridable so the trade can be swept.
TRANSIT_ATTITUDE_AUTHORITY = float(os.environ.get("TRANSIT_ATTITUDE_AUTHORITY", "0.25"))


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
        choices=("remove", "install", "relocate"),
        default="install",
        help=(
            "remove: capture a fully installed module and pull it clear of the rack, both learned. "
            "install: capture a module at the rack mouth and seat it, both learned. "
            "relocate: capture a module installed in the first bay, pull it clear, fly the three planned legs "
            "to the second bay, and seat it there. Needs a two-bay profile and the insert policy trained on "
            "both bays; it is the ORU changeout this roadmap is for."
        ),
    )
    parser.add_argument(
        "--task",
        default=TASK,
        help=(
            "Workflow profile to run. The GrappleVision profile adds a camera and takes the module's pose "
            "from it instead of from the simulator; everything physical is identical."
        ),
    )
    parser.add_argument(
        "--pose_head_checkpoint",
        type=Path,
        default=None,
        help="Trained module-pose head. Required by the vision profile unless --oracle is given.",
    )
    parser.add_argument(
        "--stable_lighting",
        action="store_true",
        help=(
            "Turn the per-episode sun and rack-albedo randomizers off, and the camera's radiation noise "
            "down, for recording. Randomization is what the pose head is *trained and measured* under; a "
            "recording that strobes through it every episode shows the randomizer rather than the robot. "
            "Never use this for a number."
        ),
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "Run the null control: no image is read and the robot assumes the module is exactly where "
            "the rack nominally presents it. If this scores as well as the camera arm, the perception "
            "result is measuring nothing."
        ),
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help=(
            "Run the vision profile's control arm: the module pose comes from the simulator, through the "
            "identical code path, so the difference between the arms is the estimator and nothing else."
        ),
    )
    parser.add_argument(
        "--module_mass_kg",
        type=float,
        default=None,
        help=(
            "Override the module's mass. The interface specification has to state a payload range, and "
            "the earlier mass sweep was run on the fixed-joint task where mass is nearly vacuous. Held "
            "by contact it is not: inertia levers the grip."
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
        "--handoff_trace",
        type=Path,
        default=None,
        help=(
            "Record the state every phase hands over in, and the state through the settling window, "
            "to this .npz. A skill has to be trained across the states its predecessor actually "
            "produces; that rule has now been broken three times here, and until this existed "
            "nothing measured what those states are. Off by default and free when off."
        ),
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
if args.video or "Vision" in args.task:
    # The vision profile renders a camera and drives Replicator randomizers, and
    # both need the rendering extensions up before the app launches. Without
    # this the rack-albedo randomizer fails to build its Replicator graph and
    # the environment cannot be constructed at all.
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

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
    WORKFLOW_HANDOVER_GRIP_M,
    WORKFLOW_SETTLE_S,
    capture_established,
    extraction_success_mask,
    grapple_grip_attitude_error_world,
    grapple_grip_error_metrics,
    grapple_insertion_conditions,
    grapple_insertion_success_mask,
    grip_drive_torque,
    grip_finger_angle,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import (
    attached_blade_pose_world,
    attached_blade_velocity,
    insertion_error_metrics,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world
from zero_g_blade_swap.tasks.blade_swap.mdp.perception import perceived_module_position_error
from zero_g_blade_swap.tasks.blade_swap.workflow_demo_env_cfg import (
    CAPTURE_BUDGET_S,
    EXTRACT_ACTION_SCALE,
    EXTRACT_BUDGET_S,
    GRASP_ACTION_SCALE,
    HANDOVER_HOLD_S,
    INSERT_ACTION_SCALE,
    INSERT_BUDGET_S,
    SEAT_STEPS,
    TRANSIT_TARGET_BLADE_X,
)
from zero_g_blade_swap.checkpoint_policy import CheckpointPolicy
from zero_g_blade_swap.grapple_geometry import (
    EXTRACTED_BLADE_CENTRE_X,
    TRANSIT_CLEAR_BLADE_CENTRE_X,
)
from zero_g_blade_swap.tasks.blade_swap.assets import SECOND_SLOT_CENTER_Y

#: Held still after the workflow's own predicate fires, before the outcome is
#: judged. A success that evaporates in two thirds of a second was not one.
#: Read from the skill module, because the extraction velocity limits are
#: *derived* from this window and the two must not be able to disagree.
SETTLE_STEPS = round(WORKFLOW_SETTLE_S * 30.0)
#: Grip error the capture must reach before the next skill takes over. Read from
#: the skill module, which uses the same constant as the capture task's own
#: success tolerance, so the two cannot disagree about what "captured" means.
HANDOVER_GRIP_M = WORKFLOW_HANDOVER_GRIP_M

#: Seconds each phase gets, read from the task each policy was certified on so
#: the two can never drift apart. This is the reconciliation between per-skill
#: certification and the chain: before it, a skill was certified on a 12 s
#: episode and then quoted as "it completes in the chain", where the chain
#: happened to grant 45 s. Whichever number read better was the one being
#: quoted. Now the chain gives each skill exactly the clock its own
#: certification gives it, and a phase that overruns fails the workflow.
#:
#: The three budgets, the seat length and the hand-off hold now live in
#: ``workflow_demo_env_cfg`` because the chained-insert *training* task reads the
#: same numbers, and two copies of a phase budget is exactly the drift this
#: whole mechanism exists to prevent.
PHASE_BUDGET_S = (
    CAPTURE_BUDGET_S,  # capture
    SEAT_STEPS / 30.0,  # seat, scripted
    EXTRACT_BUDGET_S,  # extract
    EXTRACT_BUDGET_S,  # transit, replays the pull
    INSERT_BUDGET_S,  # insert
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
    # What the estimator was actually wrong by, averaged and at its worst over
    # the episode. Without this a vision arm can only be diagnosed by inference:
    # the two-bay camera arm collapsed to 25.00% on one seed of three and the
    # standing explanation blamed the pose head's tail, which a direct
    # measurement of the head then refuted. Zero on the oracle arm by
    # construction, and on any state-only run.
    "perceived_error_mean_m",
    "perceived_error_max_m",
)

#: One row every time an environment changes phase, so the distribution a skill
#: is actually handed can be compared against the distribution its own reset
#: draws from. The arm joints are here because that is what the reset writes: a
#: hand-off can sit inside the grip tolerance and still be a joint configuration
#: the receiving policy has never seen, which is exactly what made the first
#: chained extract reverse into the rack.
HANDOFF_TRACE_FIELDS = (
    "step",
    "env",
    "from_phase",
    "to_phase",
    "grip_error_m",
    "grip_attitude_rad",
    "finger_angle_rad",
    "drive_torque_nm",
    "blade_x_m",
    "blade_y_m",
    "blade_z_m",
    "blade_qw",
    "blade_qx",
    "blade_qy",
    "blade_qz",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
    "tool_x_m",
    "tool_y_m",
    "tool_z_m",
    "arm_joint_0",
    "arm_joint_1",
    "arm_joint_2",
    "arm_joint_3",
    "arm_joint_4",
    "arm_joint_5",
)
#: One row per environment per step of the settling window. A pooled terminal
#: number cannot distinguish a module that was never settled from one that was
#: settled and then pushed, and those want opposite fixes.
SETTLE_TRACE_FIELDS = (
    "step",
    "env",
    "steps_since_done",
    "grip_error_m",
    "grip_attitude_rad",
    "finger_angle_rad",
    "drive_torque_nm",
    "blade_x_m",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
)


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

    def __init__(
        self, task, policies, workflow: str, transit_slowdown: int, max_steps: int, tracing: bool = False
    ) -> None:
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
        #: Which axis each planned leg travels along, for the relocation's
        #: follower. Zero for the replayed transit, which is sampled along a
        #: flown path rather than laid out, and does not use it.
        self.leg_axis = torch.zeros((self.max_waypoints, count), dtype=torch.long, device=device)
        self.waypoint_write = torch.zeros(count, dtype=torch.long, device=device)
        self.waypoint_read = torch.zeros(count, dtype=torch.long, device=device)
        # Running estimator error, accumulated every control step of the episode.
        self.perceived_error_sum = torch.zeros(count, dtype=torch.float64, device=device)
        self.perceived_error_steps = torch.zeros(count, dtype=torch.float64, device=device)
        self.perceived_error_max = torch.zeros(count, dtype=torch.float64, device=device)
        self.frozen = torch.zeros((count, len(WORKFLOW_METRIC_FIELDS)), dtype=torch.float64, device=device)
        self.frozen_valid = torch.zeros(count, dtype=torch.bool, device=device)
        # Read, not restated: the same hold the capture skill's own success
        # predicate requires, so the chain and the skill cannot disagree.
        self.required_hold = max(1, int(round(HANDOVER_HOLD_S / float(task.step_dt))))
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
        # The tool pose each episode starts from, which is the pose the insert
        # skill's own reset places the arm at. Recorded on the first step rather
        # than in reset_envs, because the reset callback runs while the previous
        # episode's terminal state is still being harvested.
        self.reset_tool_pos = torch.zeros((count, 3), device=device)
        self.reset_tool_rot = torch.zeros((count, 4), device=device)
        self.reset_tool_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.tracing = tracing
        self.handoff_rows: list[np.ndarray] = []
        self.settle_rows: list[np.ndarray] = []
        self.env_index = torch.arange(count, dtype=torch.float64, device=device)
        # The action term's own joint ids, so the trace records the joints the
        # reset writes rather than the first six of whatever order the scene has.
        arm_joint_ids = getattr(self.arm, "_joint_ids", None)
        self.arm_joint_ids = list(range(6)) if arm_joint_ids is None else list(arm_joint_ids)

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
        self.perceived_error_sum[env_ids] = 0.0
        self.perceived_error_steps[env_ids] = 0.0
        self.perceived_error_max[env_ids] = 0.0
        self.predicate_fired[env_ids] = False
        self.judged[env_ids] = False
        self.outcome[env_ids] = False
        self.all_conditions[env_ids] = False
        self.waypoint_write[env_ids] = 0
        self.waypoint_read[env_ids] = 0
        self.frozen_valid[env_ids] = False
        self.reset_tool_valid[env_ids] = False

    def _apply_scales(self) -> None:
        self.arm._scale[:] = self.scales[self.phase]

    def _trace_state(self) -> dict[str, torch.Tensor]:
        """Every quantity both traces share, read once per step."""

        task = self.task
        grip_error, grip_attitude = grapple_grip_error_metrics(task)
        blade_position, blade_orientation = attached_blade_pose_world(task)
        velocity = attached_blade_velocity(task)
        return {
            "grip_error_m": grip_error.to(torch.float64),
            "grip_attitude_rad": grip_attitude.to(torch.float64),
            "finger_angle_rad": grip_finger_angle(task).to(torch.float64),
            "drive_torque_nm": grip_drive_torque(task).to(torch.float64),
            "blade_local": (blade_position - task.scene.env_origins).to(torch.float64),
            # The module's orientation travels with its position or the pair is
            # not a pose. Sampling an arm pose against a module whose attitude
            # was defaulted is what made the arm-only bank unfaithful.
            "blade_quat": blade_orientation.to(torch.float64),
            "blade_linear_velocity_mps": torch.linalg.vector_norm(velocity[:, :3], dim=-1).to(torch.float64),
            "blade_angular_velocity_radps": torch.linalg.vector_norm(velocity[:, 3:], dim=-1).to(torch.float64),
        }

    def _column(self, value: float) -> torch.Tensor:
        return torch.full((self.task.num_envs, 1), value, dtype=torch.float64, device=self.task.device)

    def _record_handoff(self, mask: torch.Tensor, step: int, entry_phase: torch.Tensor) -> None:
        task = self.task
        state = self._trace_state()
        tool = (end_effector_pose_world(task)[0] - task.scene.env_origins).to(torch.float64)
        joints = task.scene["robot"].data.joint_pos[:, self.arm_joint_ids].to(torch.float64)
        rows = torch.cat(
            (
                self._column(float(step)),
                self.env_index.unsqueeze(-1),
                entry_phase.to(torch.float64).unsqueeze(-1),
                self.phase.to(torch.float64).unsqueeze(-1),
                state["grip_error_m"].unsqueeze(-1),
                state["grip_attitude_rad"].unsqueeze(-1),
                state["finger_angle_rad"].unsqueeze(-1),
                state["drive_torque_nm"].unsqueeze(-1),
                state["blade_local"],
                state["blade_quat"],
                state["blade_linear_velocity_mps"].unsqueeze(-1),
                state["blade_angular_velocity_radps"].unsqueeze(-1),
                tool,
                joints,
            ),
            dim=-1,
        )
        self.handoff_rows.append(rows[mask].cpu().numpy())

    def _record_settle(self, mask: torch.Tensor, step: int) -> None:
        state = self._trace_state()
        rows = torch.cat(
            (
                self._column(float(step)),
                self.env_index.unsqueeze(-1),
                (step - self.done_at).to(torch.float64).unsqueeze(-1),
                state["grip_error_m"].unsqueeze(-1),
                state["grip_attitude_rad"].unsqueeze(-1),
                state["finger_angle_rad"].unsqueeze(-1),
                state["drive_torque_nm"].unsqueeze(-1),
                state["blade_local"][:, :1],
                state["blade_linear_velocity_mps"].unsqueeze(-1),
                state["blade_angular_velocity_radps"].unsqueeze(-1),
            ),
            dim=-1,
        )
        self.settle_rows.append(rows[mask].cpu().numpy())

    def trace_npz(self) -> dict[str, np.ndarray]:
        def stack(rows: list[np.ndarray], fields: tuple[str, ...]) -> np.ndarray:
            return np.concatenate(rows) if rows else np.zeros((0, len(fields)), dtype=np.float64)

        return {
            "handoff": stack(self.handoff_rows, HANDOFF_TRACE_FIELDS),
            "handoff_fields": np.asarray(HANDOFF_TRACE_FIELDS),
            "settle": stack(self.settle_rows, SETTLE_TRACE_FIELDS),
            "settle_fields": np.asarray(SETTLE_TRACE_FIELDS),
        }

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
        # Immediately after the observation, because that is when the perceived
        # grip-error term has just run and cached how wrong it was. Zero on the
        # oracle and on state-only tasks, which is the right reading: those have
        # no estimator to be wrong.
        perceived = perceived_module_position_error(task).to(torch.float64)
        self.perceived_error_sum += perceived
        self.perceived_error_steps += 1.0
        self.perceived_error_max = torch.maximum(self.perceived_error_max, perceived)
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
        tool, tool_rot = end_effector_pose_world(task)
        blade_x = _blade_centre_x(task)
        fresh = ~self.reset_tool_valid
        if bool(fresh.any()):
            self.reset_tool_pos[fresh] = tool[fresh]
            self.reset_tool_rot[fresh] = tool_rot[fresh]
            self.reset_tool_valid[fresh] = True

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
        # --- seat -> align (install) or extract -------------------------------
        # Removal hands straight over: the extract skill is certified from this
        # pose and the chain measures 98.78% doing so. Installation realigns
        # first, for the reason ALIGN_STEPS documents.
        # Every workflow that begins by taking a module *out* hands the seat
        # straight to extraction; only a pure installation starts with the module
        # already at the mouth and goes to the insert phase.
        if self.workflow in ("remove", "relocate"):
            seated = (self.phase == SEAT) & (step >= self.seat_until)
            if bool(seated.any()):
                self.phase[seated] = EXTRACT
        else:
            aligning = (self.phase == SEAT) & (step >= self.seat_until) & self.reset_tool_valid
            if bool(aligning.any()):
                ids = torch.nonzero(aligning, as_tuple=False).squeeze(-1)
                # The pin is seated by now, so the holding closure is no longer
                # buying anything and is actively pushing: the same wedge thrust
                # that made every chained removal fail its settling re-check.
                # Measured here as an idle 2 s pause costing 84.90% -> 21.35%.
                if ALIGN_RETAIN:
                    self.gripper.retain_latch[aligning] = True
                scale = self.scales[INSERT]
                # Orientation only. Returning the *position* as well measured
                # 78.65% against an 84.90% baseline on the same seed, because a
                # capture leaves the module 14.5 mm nearer the slot than nominal
                # and driving back to the taught point throws that progress away
                # -- which insert, with 6.5 s of median slack in a 20 s budget,
                # cannot afford to redo. The rotation is the part that is out of
                # distribution: 0.157 rad at the median, almost all wrist_1.
                if not ALIGN_HOLD_ONLY:
                    # Module-relative, not episode-relative. Driving toward the
                    # orientation the episode *started* at assumes the module is
                    # where it nominally sits, which is true for the state chain
                    # and false for the vision profile, where the module is
                    # displaced by an amount only the camera reveals. Measured:
                    # that version took the oracle arm 80.38% -> 86.63% and the
                    # camera arm 80.38% -> 72.92%, which is the signature of a
                    # correction that fights the displacement it cannot see.
                    #
                    # grapple_grip_attitude_error_world is the rotation between
                    # the tool and the head-on capture attitude *of the module as
                    # it actually is*, so it is right wherever the module is.
                    rotation_error = grapple_grip_attitude_error_world(task)[ids]
                    self.actions[ids, 3:6] = (rotation_error / scale[3:6]).clamp(-1.0, 1.0)
            arrived = (self.phase == SEAT) & (step >= self.seat_until + ALIGN_STEPS)
            if bool(arrived.any()):
                # Insertion drives the module into rails, so the grip has to
                # carry contact again.
                self.gripper.retain_latch[arrived] = False
                self.phase[arrived] = INSERT

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
                # Stop squeezing the moment the module is out. Both the capture
                # and hold commands drive far past the 0.223 rad the pads rest
                # at on the wedge, so the drive saturates at 10 N-m and the
                # wedge turns that into thrust along the pull axis. The rails
                # absorb it while the module is railed; once it is free there is
                # nothing in zero gravity to oppose it. Traced through the
                # settling window, an extraction firing at 0.008 m/s reaches
                # 0.103 m/s in 0.70 s at a constant 10 N-m with the grip never
                # slipping -- which is why every chained removal here has fired
                # its predicate and failed the re-check.
                self.gripper.retain_latch[cleared] = True
            else:
                # Same thrust, same reason, and this is the documented cause of
                # the full round trip's failure: the grip degrades from 15 mm to
                # 35 mm during the return "whatever speed it is flown at", which
                # is the signature of a constant push rather than of inertia.
                # The module is unconstrained for the whole transit, so retain
                # through it and firm up again before it meets the rails.
                self.gripper.retain_latch[cleared] = True
                self.phase[cleared] = TRANSIT
                self.transit_started[cleared] = step
                if self.workflow == "relocate":
                    # See RELOCATE_TRANSIT_HOLD: the relocation is the first
                    # phase here that moves a module through free space rather
                    # than releasing one at the end of a job, and the rule for
                    # moving is to hold.
                    if RELOCATE_TRANSIT_HOLD:
                        self.gripper.retain_latch[cleared] = False
                    self._plan_lateral_transit(cleared, tool, blade_x)
                else:
                    self.waypoint_read[cleared] = (self.waypoint_write[cleared] - 1).clamp_min(0)

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
            ids = torch.nonzero(transiting, as_tuple=False).squeeze(-1)
            target = self.waypoints[self.waypoint_read[ids], ids]
            if self.workflow == "relocate":
                # A *planned* path advances on arrival, not on the clock, and the
                # distinction is not cosmetic. The replay below walks waypoints
                # sampled four steps apart along a path the arm has just flown,
                # so a fixed cadence tracks it; the relocation's three legs are
                # 78 mm, 220 mm and 436 mm long, and the same cadence would move
                # the target three times before the tool had crossed the first
                # one -- dragging the module diagonally through the flare the
                # retreat exists to clear.
                #
                # Proximity is the wrong rule for the replay, for the reason its
                # own comment gives, and the right one here: these targets are
                # exact rather than sampled, so the tool really does reach them.
                # Along the leg's own axis, not the full 3-D distance.
                #
                # Each planned leg moves along exactly one axis -- back in x,
                # across in y, in again in x -- and the 3-D test silently assumed
                # nothing else would move the tool. Holding the module's attitude
                # does move it: with the attitude command in, the tool sat 1 mm
                # from its waypoint in x and 53 mm away in 3-D, so the follower
                # never ticked over and 50 of 64 environments were still on the
                # first leg when the episode ended. The leg is finished when the
                # distance it was laid out to cover is covered.
                reached = torch.zeros_like(self.waypoint_read, dtype=torch.bool)
                axis = self.leg_axis[self.waypoint_read[ids], ids]
                along = (target - tool[ids]).gather(1, axis.unsqueeze(-1)).squeeze(-1)
                reached[ids] = along.abs() <= 0.005
                due = transiting & reached
                # The last leg is not a transit. It drives the module 436 mm
                # along the pull axis into the second bay's channel, past that
                # bay's lead-in flares and between its rails -- which is an
                # insertion by every physical measure, and this project's
                # operating rule says a module meeting rails has to be *held*,
                # not retained.
                #
                # Measured, with the retain left on through it: the tool flew the
                # whole leg and stopped 0.4 mm from its waypoint while the module
                # travelled about 95 mm of the 436 and sat at x = 0.176 against
                # the 0.578 the arrival test needs. Every one of 64 environments
                # then timed out inside the transit, because the arrival test is
                # also what releases the retain -- the module could not be driven
                # in while retained, and the retain was not released until it was
                # driven in. Releasing it one leg earlier breaks that deadlock at
                # the physically correct moment.
                self.gripper.retain_latch[due & (self.waypoint_read <= 1)] = False
            else:
                # Never on the step the transit begins: that would consume the
                # first waypoint before the tool had been commanded toward it.
                due = (
                    transiting
                    & (step > self.transit_started)
                    & (((step - self.transit_started) % (TRANSIT_WAYPOINT_STRIDE * self.transit_slowdown)) == 0)
                )
            self.waypoint_read[due] = (self.waypoint_read[due] - 1).clamp_min(0)
            target = self.waypoints[self.waypoint_read[ids], ids]
            scale = self.scales[TRANSIT]
            self.actions[ids, :3] = ((target - tool[ids]) / scale[:3]).clamp(-1.0, 1.0)
            self.actions[ids, 3:6] = 0.0
            if self.workflow == "relocate":
                # Hold the module's attitude for the whole flight, because
                # nothing else does and the pin cannot.
                #
                # Zeroing these channels is right for the replayed transit, which
                # retraces a path the arm has just flown while the module is
                # still nearly in its rails. It is wrong for a 734 mm crossing of
                # free space, and the measurement is unambiguous: the tool
                # finishes at local x = 0.2474 against a planned 0.2475 -- exactly
                # where it should be -- while the tool-to-module offset has gone
                # from -0.335 m at transit entry to +0.305 m. That is a sign
                # flip, not a slip. The module has swung end-for-end about the
                # pin, which is why grip error stays at 24 mm throughout: the
                # tool is still on the pin, and the pin is no longer pointing the
                # way it was.
                #
                # This project has measured four separate ways that a parallel-jaw
                # grip on a passive feature cannot resist a moment about the
                # closing axis. A phase that moves a module through free space and
                # commands nothing about its attitude is therefore not holding
                # still, exactly as an arm that stops commanding while gripping is
                # not holding still. Same rule, the rotational half of it.
                # Bounded, not saturated. The grip attitude error a pull leaves
                # is 0.1 to 0.3 rad and the rotation scale is 0.020 rad per step,
                # so a plain proportional command sits on its clamp for the whole
                # flight -- and a differential IK solving one 6-D command then
                # spends its authority turning the wrist instead of crossing the
                # rack. Measured that way: the module was held beautifully, grip
                # error 24 mm to 12 mm, and the tool was still sitting at the
                # retreat waypoint 1,450 steps later with the episode running out
                # underneath it.
                #
                # A quarter of the authority still corrects far faster than the
                # module can tumble, because it is opposing a drift rather than
                # chasing a setpoint.
                rotation_error = grapple_grip_attitude_error_world(task)[ids]
                self.actions[ids, 3:6] = (rotation_error / scale[3:6]).clamp(
                    -TRANSIT_ATTITUDE_AUTHORITY, TRANSIT_ATTITUDE_AUTHORITY
                )
            # Rate-limiting the last leg to a third of the command, the way the
            # replayed transit is slowed, was tried here and measured *worse*:
            # the module ended at x = -0.158 against -0.003 at full command, and
            # the count that had crossed fell from 46 to 19 because the tool
            # itself lagged its waypoint. It is not in, and the reason it did not
            # help is that the module was not being driven too fast, it was being
            # driven at the wrong bay -- see `_plan_lateral_transit`.
            arrived = transiting & (self.waypoint_read <= 0) & (blade_x >= TRANSIT_TARGET_BLADE_X - 0.005)
            if self.workflow == "relocate":
                # And it has to have crossed, not merely come back out to the
                # right depth in the bay it started in.
                #
                # Asked of the *module*, in the rack's own frame, rather than of
                # the tool relative to where this episode happened to start. The
                # tool-relative form passed while the module sat 93 mm outside the
                # channel, because it measured a displacement rather than an
                # arrival -- the same error that put the cross leg in the wrong
                # place. Half the bay pitch is the midpoint between the two bays,
                # so this asks which bay the module is now in front of.
                blade_y = (
                    self.task.scene["spare_blade"].data.root_pos_w[:, 1] - self.task.scene.env_origins[:, 1]
                )
                arrived = arrived & (blade_y <= 0.5 * SECOND_SLOT_CENTER_Y)
            # Insertion drives the module back into its rails, so the grip has to
            # carry contact again. Retaining through that is the failure the
            # capture/hold split exists to prevent.
            self.gripper.retain_latch[arrived] = False
            self.phase[arrived] = INSERT

        # --- insert -> seated --------------------------------------------------
        inserting = self.phase == INSERT
        if bool(inserting.any()):
            # The insert skill's own success mask, including its 0.20 s hold, so
            # the chain is judged by the same predicate the skill is.
            fired = inserting & grapple_insertion_success_mask(task)
            self._finish(fired, step)
            self.predicate_fired[fired] = True
            # A seated module is a finished job, and the DONE phase below waits
            # 0.70 s while still commanding the holding closure -- which is the
            # one thing rule "any phase that waits must either command or
            # retain" forbids. Removal already retains the moment the module is
            # free; installation does not, and the asymmetry was never measured
            # rather than reasoned. Off by default so the certified number is
            # untouched until it is.
            if SEATED_RETAIN:
                self.gripper.retain_latch[fired] = True

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
        if self.tracing:
            # After every transition this step has resolved, so the row records
            # the state the *next* phase begins from rather than a phase midway
            # through handing over.
            changed = self.phase != entry_phase
            if bool(changed.any()):
                self._record_handoff(changed, step, entry_phase)
            # Through the settling window inclusive of both ends, so a module
            # that was settled when the predicate fired and is not settled when
            # it is judged shows up as a curve rather than as two numbers.
            settling = (self.phase == DONE) & (self.done_at >= 0) & ((step - self.done_at) <= SETTLE_STEPS)
            if bool(settling.any()):
                self._record_settle(settling, step)
        self.phase_started[self.phase != entry_phase] = step
        self._apply_scales()

    def _plan_lateral_transit(self, mask: torch.Tensor, tool: torch.Tensor, blade_x: torch.Tensor) -> None:
        """Lay out the path from the first bay to the second, three waypoints.

        The removal transit replays the pull backwards, which is feasible by
        construction because the arm has just flown it. A relocation has no such
        path: nothing has ever been to the second bay. So it is planned, and the
        only two numbers in it are derived rather than chosen.

        *Retreat first.* Extraction stops the instant the module's rear face
        clears the mouth, which is the right definition of "removed" and the
        wrong place to turn: the lead-in flares stand proud of the mouth, so a
        module that travels sideways from there drags its nose across the
        neighbouring bay's flare. ``TRANSIT_RETREAT_M`` is how much further back
        the module has to come for its front face to clear the flare plane, and
        it comes out of the flare geometry at about 78 mm.

        *Then across, then back in.* The final waypoint puts the module where the
        insert skill expects to start, and the tool-to-module offset is *measured*
        at this instant rather than assumed --- the module sits wherever the
        capture and the pull have left it in the pads, and this whole project's
        recurring defect is a constant restated instead of read.

        **Every leg targets the bay, not a displacement, and the lateral one had
        to be corrected to do so.** It was written as ``back_y + SECOND_SLOT_CENTER_Y``
        --- cross 220 mm from wherever the tool is --- while the two axial legs
        were already written as absolute positions with the measured offset added.
        A relative cross carries whatever lateral error capture and extraction
        left into the second bay, and measured over 64 relocations that error is
        about 93 mm at the median. The bay's channel half-width is 72.5 mm, so the
        module arrived outside the channel it was being pushed into, jammed its
        nose on the lead-in flare, and every episode timed out in the transit.
        The symptom was a tool sitting 0.4 mm from its final waypoint with the
        module 580 mm behind it.

        Every leg is followed closed-loop on position, like the removal transit,
        so a waypoint that is slightly off does not accumulate; and the module is
        retained while it flies through free space, then held again for the last
        leg, which drives it between the second bay's rails and is an insertion in
        everything but name.
        """

        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        # Measured now: where the tool sits relative to the module it is holding,
        # on both axes the transit moves along. `blade_x` and `blade_y` are
        # environment-local and `tool` is world, so each offset carries this
        # environment's origin -- which is exactly right, because the waypoints
        # these build are world too and the origin cancels on arrival.
        blade_y = (
            self.task.scene["spare_blade"].data.root_pos_w[:, 1] - self.task.scene.env_origins[:, 1]
        )
        tool_to_blade_x = tool[ids, 0] - blade_x[ids]
        tool_to_blade_y = tool[ids, 1] - blade_y[ids]
        retreat_x = TRANSIT_CLEAR_BLADE_CENTRE_X + tool_to_blade_x
        staging_x = TRANSIT_TARGET_BLADE_X + tool_to_blade_x
        staging_y = SECOND_SLOT_CENTER_Y + tool_to_blade_y

        back = tool[ids].clone()
        back[:, 0] = retreat_x
        across = back.clone()
        across[:, 1] = staging_y
        approach = across.clone()
        approach[:, 0] = staging_x

        # Written in reverse, because the follower walks the buffer downwards.
        self.waypoints[2, ids] = back
        self.waypoints[1, ids] = across
        self.waypoints[0, ids] = approach
        self.waypoint_read[ids] = 2
        # The axis each leg is laid out along, so the follower can ask whether
        # the leg is finished rather than whether the tool is at a point. Holding
        # the module's attitude moves the tool off that point on the other two
        # axes, which is correct behaviour and used to stall the follower.
        self.leg_axis[2, ids] = 0  # back: along x
        self.leg_axis[1, ids] = 1  # across: along y
        self.leg_axis[0, ids] = 0  # in again: along x

    def transit_progress(self) -> str:
        """One line describing where the transit follower actually is.

        Empty unless something is transiting, so it costs nothing on the
        workflows that do not fly anywhere. Reports the leg each environment is
        on and how far it still is from that leg's waypoint, because a follower
        that has stalled and one that is merely slow produce identical phase
        counts -- and the first relocation run spent eleven minutes not saying
        which it was.
        """

        transiting = self.phase == TRANSIT
        if not bool(transiting.any()):
            return ""
        ids = torch.nonzero(transiting, as_tuple=False).squeeze(-1)
        tool = end_effector_pose_world(self.task)[0][ids]
        target = self.waypoints[self.waypoint_read[ids], ids]
        distance = torch.linalg.vector_norm(target - tool, dim=-1)
        legs = torch.bincount(self.waypoint_read[ids], minlength=3).tolist()
        # Each conjunct of `arrived` separately, because the first relocation run
        # showed the tool sitting 0.4 mm from its last waypoint and not arriving,
        # and a single boolean cannot say which clause is the one refusing.
        blade_x = _blade_centre_x(self.task)[ids]
        lateral = (
            self.task.scene["spare_blade"].data.root_pos_w[:, 1] - self.task.scene.env_origins[:, 1]
        )[ids]
        # Is the module still on the tool at all? A tool that flies its whole
        # path while the module does not follow has either lost the grip or is
        # dragging the module against something, and grip error tells the two
        # apart: a lost module's error grows without bound, a snagged one's does
        # not. Reported here because the hand-off trace only samples phase
        # boundaries, and this failure lives in the middle of a phase.
        grip_error, _ = grapple_grip_error_metrics(self.task)
        grip_error = grip_error[ids]
        return (
            f"  transit: legs_remaining={legs[:3]} "
            f"to_waypoint_m p50={float(distance.median()):.4f} max={float(distance.max()):.4f} "
            f"| last_leg={int((self.waypoint_read[ids] <= 0).sum())} "
            f"blade_x_ok={int((blade_x >= TRANSIT_TARGET_BLADE_X - 0.005).sum())} "
            f"(p50={float(blade_x.median()):.4f} need>={TRANSIT_TARGET_BLADE_X - 0.005:.4f}) "
            f"crossed={int((lateral <= 0.5 * SECOND_SLOT_CENTER_Y).sum())} "
            f"(p50={float(lateral.median()):.4f} need<={0.5 * SECOND_SLOT_CENTER_Y:.4f}) "
            f"| grip_error_m p50={float(grip_error.median()):.4f} max={float(grip_error.max()):.4f} "
            # The tool's own position, in the same environment-local frame the
            # module is reported in. Everything above is either a distance or a
            # module position, and three of those cannot be reconciled without
            # knowing where the tool actually is.
            f"| tool_local_x p50={float((tool[:, 0] - self.task.scene.env_origins[ids, 0]).median()):.4f} "
            f"target_local_x p50={float((target[:, 0] - self.task.scene.env_origins[ids, 0]).median()):.4f}"
        )

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
            "perceived_error_mean_m": self.perceived_error_sum / self.perceived_error_steps.clamp_min(1.0),
            "perceived_error_max_m": self.perceived_error_max,
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
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        env_cfg.configure_robustness(0)
        env_cfg.seed = args.seed
        if args.pose_head_checkpoint is not None:
            if not args.pose_head_checkpoint.is_file():
                raise FileNotFoundError(args.pose_head_checkpoint)
            env_cfg.pose_head_checkpoint = str(args.pose_head_checkpoint.resolve())
        if args.module_mass_kg is not None:
            env_cfg.scene.spare_blade.spawn.mass_props.mass = args.module_mass_kg
            print(f"[INFO] Module mass set to {args.module_mass_kg} kg")
        if args.stable_lighting:
            # Recording only, and the report says so, so a clip can never be
            # mistaken for a measurement made under easier conditions.
            for term in ("rack_albedo", "orbital_sun"):
                if getattr(env_cfg.events, term, None) is not None:
                    setattr(env_cfg.events, term, None)
            for group in ("rgb",):
                obs = getattr(env_cfg.observations, group, None)
                if obs is not None and getattr(obs, "rgb", None) is not None:
                    obs.rgb.params["noise_std_range"] = (0.0, 0.002)
            print("[INFO] Visual randomization off for recording; this run is not evidence")
        if args.blind:
            env_cfg.pose_head_blind = True
        if args.oracle:
            # The control arm: the module pose comes from the simulator but
            # through the identical observation term, so the difference between
            # the arms is the estimator and cannot be a different code path.
            env_cfg.pose_head_oracle_blend = 1.0
        # Derive the episode from the phases this workflow actually runs, each at
        # the budget its own skill was certified on, instead of one round number
        # generous enough to hide an overrun. It is also what makes a many-seed
        # sweep affordable: an install that finishes at 14 s no longer idles to
        # 45 s before its environment resets.
        phases = [CAPTURE, SEAT]
        if args.workflow != "install":
            phases.append(EXTRACT)
        # Every workflow that flies the module anywhere needs the transit in its
        # episode, and the relocation was missing from this list -- so its
        # episode was derived as capture + seat + extract + insert and the three
        # planned legs had no room at all. That is precisely the defect this
        # derivation exists to prevent, and it would have shown up as a
        # relocation chain that lands below the product of its parts with the
        # cause invisible.
        #
        # The relocation's transit is *not* multiplied by `--transit_slowdown`,
        # and that is not an omission. The slowdown exists because a full-speed
        # replay of the pull rotates the module in the pads, and it is applied to
        # a follower that advances on a fixed cadence. The relocation's follower
        # advances on arrival, so a slowdown factor would change nothing about
        # how it flies and would only inflate the episode.
        #
        # `PHASE_BUDGET_S[TRANSIT]` is the extract skill's certified 25 s, which
        # is ample rather than tight: the three legs are 78.2 mm and 436.1 mm
        # along the pull axis at 0.24 m/s and 220 mm across it at 0.12 m/s, so
        # the planned path is 734 mm and about 4.0 s of pure travel. The margin
        # is for closed-loop tracking, not for the distance.
        if args.workflow == "relocate":
            phases.append(TRANSIT)
        if args.workflow != "remove":
            phases.append(INSERT)
        budget = sum(
            PHASE_BUDGET_S[index]
            * (args.transit_slowdown if index == TRANSIT and args.workflow != "relocate" else 1)
            for index in phases
        )
        if args.workflow != "remove":
            # The scripted realign runs inside the seat phase, so it is not in
            # PHASE_BUDGET_S and has to be added to the episode explicitly.
            budget += ALIGN_STEPS / 30.0
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

        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
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
        driver = WorkflowDriver(
            task,
            policies,
            args.workflow,
            args.transit_slowdown,
            episode_steps,
            tracing=args.handoff_trace is not None,
        )
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
                # A stalled scripted phase looks exactly like a slow one in the
                # counts alone, and the relocation's first run spent eleven
                # minutes not saying which it was. The follower's own state is
                # what distinguishes them: which leg it is on and how far it
                # still is from that leg's waypoint.
                print(
                    f"[CHAIN] step {step:5d}  episodes={len(recorder):4d}/{args.episodes}  {counts}"
                    f"{driver.transit_progress()}",
                    flush=True,
                )
            if single:
                judged = int(driver.phase[0]) == DONE and bool(driver.judged[0])
                if judged and step >= int(driver.done_at[0]) + SETTLE_STEPS + args.settle_steps:
                    break
                if bool(terminated[0] or truncated[0]):
                    note(f"episode ended during {PHASE_NAMES[int(driver.phase[0])]}", step)
                    break

        # Written before anything that formats a report, because it is the
        # expensive half of this run and it must not be hostage to the cheap
        # half. The relocation's first trace cost eleven minutes of simulation
        # and produced no file at all, because a dict literal below was missing
        # a workflow key -- and a diagnostic that is thrown away when something
        # else goes wrong is a diagnostic that is missing when it is needed.
        if args.handoff_trace is not None:
            args.handoff_trace.parent.mkdir(parents=True, exist_ok=True)
            trace = driver.trace_npz()
            np.savez_compressed(args.handoff_trace, **trace)
            print(
                f"[INFO] Wrote {args.handoff_trace}: "
                f"{trace['handoff'].shape[0]} hand-offs, {trace['settle'].shape[0]} settling rows",
                flush=True,
            )

        combined = (
            hashlib.sha256("".join(policies[name].sha256 for name in ("capture", "extract", "insert")).encode())
            .hexdigest()
            .upper()
        )
        result: dict[str, object] = {
            "task": args.task,
            "visual_randomization": "off (recording)" if args.stable_lighting else "on",
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
                "relocate": ["capture", "extract", "insert"],
            }[args.workflow],
            "scripted_phases": ["seat"]
            + (["transit"] if args.workflow == "relocate" else []),
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
                                "task": args.task,
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
        if not collecting:
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
        report = {"task": args.task, "error": f"{type(exc).__name__}: {exc}", "completed": False}
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(round_floats(report), indent=2)[:2000], flush=True)
        simulation_app.close()
