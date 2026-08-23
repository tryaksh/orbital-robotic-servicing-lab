"""Run capture, extraction, transit, and re-insertion in one episode.

Separately trained capture and extraction checkpoints begin one continuous
episode on one module. The driver switches on **measured conditions**, never on a timer:
the capture hands over when the drive torque says the pads are loaded on the pin,
the pull hands over when the module's rear face is clear of the rack mouth, and
the transit hands over when the module reaches the pose the insert policy was
trained from.

What is learned and what is not, stated plainly, because a demonstration that
blurs this is worthless:

* capture and extraction are trained policies, run deterministically from their
  checkpoints;
* once the module is clear, a physical six-axis service shuttle takes the load,
  the robot opens and retreats, and guarded closed-loop motion retreats, crosses
  bays, aligns, and inserts.  The shuttle writes only force-drive targets; it
  never writes a robot or payload pose;
* the receiving bay is deliberately designed for robotic service: straight
  relieved rails and a vertical lead-in replace a passive flare/floor geometry
  that stopped a micrometre-aligned module at the mouth.

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
import math
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."

TASK = "Isaac-ZeroG-Blade-GrapplePin-Workflow-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_RUNTIME_SOURCES = (
    Path("scripts/run_workflow_demo.py"),
    Path("src/zero_g_blade_swap/fiducial.py"),
    Path("src/zero_g_blade_swap/tasks/blade_swap/assets.py"),
    Path("src/zero_g_blade_swap/tasks/blade_swap/mdp/perception.py"),
    Path("src/zero_g_blade_swap/tasks/blade_swap/scene_cfg.py"),
    Path("src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py"),
)
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
TRANSIT_ALIGN_ATTITUDE_AUTHORITY = float(os.environ.get("TRANSIT_ALIGN_ATTITUDE_AUTHORITY", "0.25"))
TRANSIT_HOLD_ATTITUDE_AUTHORITY = float(os.environ.get("TRANSIT_HOLD_ATTITUDE_AUTHORITY", "1.0"))
RELOCATE_FINAL_LEG_POSITION_AUTHORITY = float(os.environ.get("RELOCATE_FINAL_LEG_POSITION_AUTHORITY", "0.33"))
BASE_RAIL_TARGET_STEP_M = 0.0020
BASE_STAGE_OUTER_LOOP_GAIN = 0.08
BASE_STAGE_ROTATION_STEP_DEG = 0.03
# The D6 drive is a position servo, so its target must move much slower than a
# velocity command.  A 0.20 gain advanced the set-point by 7.5 degrees/second,
# outran the physical joint, and wound all three axes into their stops.  This
# bounded outer-loop gain lets the spring-damped stage settle as it corrects.
BASE_STAGE_ROTATION_GAIN = 0.02
BASE_STAGE_ROTATION_LIMIT_DEG = 18.0
BASE_STAGE_MAX_TRANSLATION_LEAD_M = 0.100
BASE_STAGE_GUARDED_AXIAL_STEP_M = 0.0005
BASE_STAGE_MIN_TARGET_M = (-0.200, -0.600, -0.400)
BASE_STAGE_MAX_TARGET_M = (0.800, 0.200, 0.400)
BASE_RAIL_MIN_TARGET_M = -0.340
BASE_RAIL_MAX_TARGET_M = 0.100
BASE_STAGE_ARM_STIFFNESS_MULTIPLIER = 64.0
STAGE_ALIGNMENT_CAPTURE_RAD = 0.065
# The guarded shuttle must not treat sub-frame RGB-D estimator noise as a
# physical loss of alignment and retract hundreds of millimetres. These bounds
# remain inside the rail lead-in envelope and above the certified RGB-D p95
# errors (1.68 mm and 0.0121 rad); the final seating predicate is unchanged.
FIDUCIAL_GUARDED_LATERAL_TOLERANCE_M = 0.002
FIDUCIAL_GUARDED_ORIENTATION_TOLERANCE_RAD = 0.015
#: Clearance the retreat leaves between the module's real front corner and the
#: plane the lead-in flares stand on.
#:
#: ``TRANSIT_CLEAR_BLADE_CENTRE_X`` is derived for a module that is square to the
#: rack, and a module that has just been pulled out of the rails is not: measured
#: at the extract hand-off it sits 0.12 to 0.13 rad off, which swings its front
#: corner 7.7 mm further forward than half its length. The nominal retreat
#: therefore stops with the corner 8 mm *inside* the flare, and the crossing leg
#: then drags it along the flare and stalls -- 0.75 mm/step of commanded lateral
#: motion against a plate.
#:
#: The retreat below is recomputed from the module's *measured* corners at the
#: instant the rails let go. This is the margin on top of that.
TRANSIT_FLARE_CLEARANCE_M = 0.010
#: The guarded robot-driven insertion's axial advance per control step, and how
#: far its target may lead the module it is pushing.
#:
#: 1 mm per step at 30 Hz is 30 mm/s, so the 167 mm from the staging pose to the
#: seated pose takes about 5.6 s of a 30 s budget. The lead cap is what makes it
#: guarded rather than merely slow: a target that keeps advancing while the arm
#: cannot follow turns a stall into a lunge, and the module is between rails.
#: Share of the rotation command the rigid transit and the guarded insertion
#: may use. One, unless a sweep says otherwise: see ``_step_rigid_transit``.
RIGID_TRANSIT_ATTITUDE_AUTHORITY = float(os.environ.get("RIGID_TRANSIT_ATTITUDE_AUTHORITY", "1.0"))
#: Legs the rigid transit flies, and **why there are four rather than three.**
#:
#: The obvious plan is retreat, cross, insert, and it is what the pad-held
#: follower does. It does not work for a carried module, and the reason is the
#: workcell rather than the payload. ``docs/service_interface_spec.md`` section
#: 6a measures a region around the arm's own axis in which position and the
#: head-on attitude cannot both be held: inside it the solver trades one for the
#: other at about 7.5 metres per radian. Bay 0 sits on that axis. So a leg that
#: asks the arm to cross *and* square at the source depth gets a compromise, and
#: measured here it settles at 0.164 rad of attitude error -- which, on a rigid
#: payload, is 56 mm of module position error it cannot correct either.
#:
#: The same measurement says where the squaring is free: every cell 220 mm or
#: more off the base's plane succeeds, and the second bay is at 220 mm exactly.
#: So the crossing holds whatever attitude the rails released the module in, and
#: the squaring happens after it has arrived, where the arm can afford it.
#: Whether the lock gives up its rigidity to mate. Set from --mating_mode.
MATING_MODE = "compliant"
#: How far the module-space trim may move the tool beyond its feed-forward.
#:
#: **Five millimetres, and the small number is the point.** At the compliance's
#: full 25 mm stroke the trim is not a correction, it is a shove: measured, it
#: drove the entering module into the channel wall hard enough to pop it out
#: sideways and 200 mm back toward the bay it came from, twice. Five millimetres
#: against a 40 kN/m spring is 200 N, inside the mating cap, which is enough to
#: pull a module the last fraction of a millimetre and not enough to wedge one.
MATING_TRIM_LIMIT_M = 0.005
RIGID_TRANSIT_LEGS = 5
#: Control steps a leg may spend not meeting its gate before the follower gives
#: up on it, advances, and records that it did.
#:
#: Not a convenience. Two of the five legs ask the arm to *square* the carried
#: module to the rack, and whether it can is a property of where the arm is
#: standing rather than of how long it is given: a resolved-rate controller
#: converges to the best attitude its own branch admits and then stops, which is
#: what this project measured as a reach boundary long before anything was being
#: carried. A leg with no timeout turns that into a deadlock and a timeout row;
#: a leg with one turns it into a number, per leg, in the report.
RIGID_TRANSIT_LEG_TIMEOUT_STEPS = int(os.environ.get("RIGID_TRANSIT_LEG_TIMEOUT_STEPS", "400"))
GUARDED_INSERT_AXIAL_STEP_M = 0.0010
GUARDED_INSERT_MAX_LEAD_M = 0.010
#: Margin on the derived depth at which an engaged latch jaw would enter the
#: slot mouth. Five millimetres, against a 17 mm interlock at zero seek.
GUARDED_INSERT_RELEASE_MARGIN_M = 0.005
#: Terminal tolerance for the time-parameterized reverse joint trajectory.
#: Intermediate samples are actuator set-points, not stop-and-settle poses;
#: requiring convergence at every sample deadlocked on finite drive error.
JOINT_REPLAY_CONVERGENCE_RAD = 0.040


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
        "--perception_backend",
        choices=("pose_head", "fiducial_pnp"),
        default="pose_head",
        help=(
            "Camera estimator used by the vision profile. fiducial_pnp uses only RGB, calibrated "
            "intrinsics/extrinsics, and the module's four visual datum patches; it needs no checkpoint."
        ),
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
        "--latch_on_release",
        action="store_true",
        help=(
            "Engage the modelled capture latch the instant the module clears the rails, instead of leaving "
            "it off. The latch was refuted engaged on capture, where it jams the module in the rails; "
            "release-time compliant-latch probes remain experimental and have not completed relocation. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--latch_rated_torque_nm",
        type=float,
        default=5.0,
        help="Rating of the latch, in newton-metres. Only used with --latch_on_release.",
    )
    parser.add_argument(
        "--latch_rated_force_n",
        type=float,
        default=250.0,
        help=(
            "Translational rating of the form-locking latch. The earlier torque-only experiment is "
            "reproduced with 0 N; a positive value models the axial/lateral load path a rigidizing "
            "capture mechanism must provide after rail release. Only used with --latch_on_release."
        ),
    )
    parser.add_argument(
        "--latch_joint_mode",
        choices=("compliant", "fixed"),
        default="compliant",
        help=(
            "Use the experimental one-sided compliant wrench or a two-body, break-rated PhysX "
            "fixed joint after a learned physical capture. Only used with --latch_on_release."
        ),
    )
    parser.add_argument(
        "--latch_position_stiffness_n_per_m",
        type=float,
        default=2_500.0,
        help="Translational stiffness of the release-time latch in N/m.",
    )
    parser.add_argument(
        "--latch_position_damping_ratio",
        type=float,
        default=0.9,
        help="Translational damping ratio of the release-time latch.",
    )
    parser.add_argument(
        "--latch_rotation_stiffness_nm_per_rad",
        type=float,
        default=10.0,
        help="Rotational stiffness of the release-time latch in N-m/rad.",
    )
    parser.add_argument(
        "--latch_rotation_damping_ratio",
        type=float,
        default=0.9,
        help="Rotational damping ratio of the release-time latch.",
    )
    parser.add_argument(
        "--mating_force_cap_n",
        type=float,
        default=0.0,
        help=(
            "Force the mating compliance may apply. Zero derives it as stiffness x stroke. "
            "Measured: more is not better -- 1000 N against this rack wedges the module at a third "
            "of its travel where 400 N walks it in, because a lead-in guides a part it can move and "
            "jams one it cannot."
        ),
    )
    parser.add_argument(
        "--mating_mode",
        choices=("compliant", "rigid"),
        default="compliant",
        help=(
            "Whether the form lock gives up its rigidity where the module meets the rack. "
            "rigid keeps the weld through the seating, which needs a channel that admits the "
            "attitude the arm delivers; compliant gives the lead-in something it can push."
        ),
    )
    parser.add_argument(
        "--destination_channel_relief_m",
        type=float,
        default=0.0,
        help=(
            "Per-side clearance added to the destination bay's channel. Zero is the rack as built. "
            "A module delivered rigidly needs L*theta/2 of it; one delivered compliantly should not, "
            "and that difference is the experiment."
        ),
    )
    parser.add_argument(
        "--base_rail_on_relocation",
        action="store_true",
        help=(
            "Command the compliant D6 mount's lateral drive during the collision-clear relocation crossing, "
            "using it as a physical seventh-axis rail."
        ),
    )
    parser.add_argument(
        "--base_rail_arm_mode",
        choices=("ik_attitude", "joint_hold"),
        default="ik_attitude",
        help=(
            "ik_attitude lets the arm align payload attitude while the rail owns lateral motion; "
            "joint_hold is the measured diagnostic control that carries a fixed arm posture."
        ),
    )
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
import omni.usd
import torch

from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv, quat_mul
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Gf, UsdPhysics

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
    MATING_ROTATION_LIMIT_RAD,
    MATING_TRAVEL_LIMIT_M,
    EXTRACTION_LINEAR_VELOCITY_LIMIT,
    WORKFLOW_HANDOVER_GRIP_M,
    WORKFLOW_SETTLE_S,
    capture_established,
    extraction_success_mask,
    arm_grapple_latch,
    grapple_grip_attitude_error_world,
    grapple_grip_error_metrics,
    grapple_insertion_conditions,
    grapple_insertion_success_mask,
    grapple_latch_diagnostics,
    grapple_latch_rigid,
    grapple_latched,
    release_grapple_latch,
    soften_grapple_latch,
    grip_drive_torque,
    grip_finger_angle,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import (
    INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS,
    INSERTION_AXIAL_DEPTH_TOLERANCE_M,
    INSERTION_LATERAL_TOLERANCE_M,
    INSERTION_LINEAR_VELOCITY_LIMIT_MPS,
    INSERTION_ORIENTATION_TOLERANCE_RAD,
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
    TRANSIT_TARGET_BLADE_POSE,
)
from zero_g_blade_swap.checkpoint_policy import CheckpointPolicy
from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    EXTRACTED_BLADE_CENTRE_X,
    FLARE_LEADING_X,
    SLOT_MOUTH_X,
    TRANSIT_CLEAR_BLADE_CENTRE_X,
)
from zero_g_blade_swap.service_latch import (
    ENGAGED_DEPTH_FROM_FLANGE_M as _LATCH_ENGAGED_DEPTH_M,
)
from zero_g_blade_swap.service_latch import (
    MODULE_FACE_FROM_FLANGE_M as LATCH_MODULE_FACE_DEPTH_M,
)

#: Depth from the flange the far face of an engaged latch jaw reaches at zero
#: carriage seek. Named here so the release interlock below reads as one
#: expression rather than an index into a tuple.
LATCH_ENGAGED_FAR_DEPTH_M = _LATCH_ENGAGED_DEPTH_M[1]


from zero_g_blade_swap.grapple_geometry import SLOT_ENTRY_RAMP_CATCH_M
from zero_g_blade_swap.tasks.blade_swap.assets import SECOND_SLOT_CENTER_Y, SECOND_SLOT_INSERTED_POS

#: The envelope the guarded insertion's advance is checked against, and it is
#: **the mouth's, not the seated tolerance**.
#:
#: The first version of this used the payload shuttle's own guard -- 1.5 mm and
#: 3 mrad -- and that is the right number for a six-axis metrology stage and the
#: wrong one for an arm: the transit delivers the module to the staging pose at
#: about 13 mrad, so the guard was never satisfied and the insertion never
#: advanced at all. A guard that no achievable state can pass is not fail-closed,
#: it is closed.
#:
#: The physically meaningful condition is whether the bay can still accept the
#: module: inside the lead-in's catch the contact geometry walks it into the
#: channel, and outside it the module jams on a plate whatever the controller
#: does. So the guard is the lead-in, and the tight tolerances stay where they
#: belong -- on the seated success predicate, which is unchanged.
GUARDED_INSERT_LATERAL_TOLERANCE_M = SLOT_ENTRY_RAMP_CATCH_M
GUARDED_INSERT_ORIENTATION_TOLERANCE_RAD = SLOT_ENTRY_RAMP_CATCH_M / (0.5 * BLADE_LENGTH_M)

#: Held still after the workflow's own predicate fires, before the outcome is
#: judged. A success that evaporates in two thirds of a second was not one.
#: Read from the skill module, because the extraction velocity limits are
#: *derived* from this window and the two must not be able to disagree.
SETTLE_STEPS = round(WORKFLOW_SETTLE_S * 30.0)
#: Grip error the capture must reach before the next skill takes over. Read from
#: the skill module, which uses the same constant as the capture task's own
#: success tolerance, so the two cannot disagree about what "captured" means.
HANDOVER_GRIP_M = WORKFLOW_HANDOVER_GRIP_M
#: The two-bay live workflow is allowed to leave preflight only when the camera
#: says the requested source bay is occupied and destination bay is clear.  The
#: head's sigmoid values are decision scores rather than calibrated confidence,
#: so this threshold is reported plainly and no uncertainty claim is made.
OCCUPANCY_PLAN_THRESHOLD = 0.5

#: Fail-closed relocation-to-insert contract.  Position and attitude are the
#: insert task's full-distance reset displaced to bay 1; tolerances and motion
#: limits are the insert success envelope itself.  The receiving policy was
#: never trained on an arbitrary point after the module crossed the rack
#: midpoint, so that weaker condition cannot authorize a hand-off.
RELOCATION_INSERT_STAGING_POS = (
    TRANSIT_TARGET_BLADE_POSE[0],
    SECOND_SLOT_CENTER_Y,
    TRANSIT_TARGET_BLADE_POSE[2],
)
RELOCATION_INSERT_STAGING_ROT = TRANSIT_TARGET_BLADE_POSE[3:7]
INSERT_HANDOFF_POSITION_TOLERANCE_M = INSERTION_LATERAL_TOLERANCE_M

#: What "the robot carried it" has to mean numerically, **derived** from the
#: contract above rather than chosen.
#:
#: ``_plan_lateral_transit`` measures the tool-to-module transform once, at the
#: instant extraction clears the rails, and lays out every remaining waypoint
#: from it: the final tool pose is the desired module pose in bay 1 minus that
#: measured offset. So any change in that transform during the flight lands on
#: the staging pose one millimetre for one millimetre, and the staging pose has
#: to arrive inside the receiving policy's own success envelope. The retention
#: limit is therefore not a new tolerance -- it is the hand-off tolerance, read
#: backwards.
TRANSIT_RETENTION_POSITION_LIMIT_M = INSERT_HANDOFF_POSITION_TOLERANCE_M
TRANSIT_RETENTION_ORIENTATION_LIMIT_RAD = INSERTION_ORIENTATION_TOLERANCE_RAD

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
    "latch_engaged",
    "latch_relative_position_error_m",
    "latch_relative_orientation_error_rad",
    "latch_applied_force_n",
    "latch_applied_torque_nm",
    "latch_force_saturated",
    "latch_torque_saturated",
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
#: One row per environment per sampled control step of the transit, which is the
#: only phase in this project where the robot carries an unrailed module through
#: free space. The handoff trace records the two instants that bracket it; that
#: cannot distinguish a grip that held from one that failed and was recovered by
#: the destination geometry, and those are opposite results.
#:
#: The load-bearing columns are the ``tool_to_module_*_drift`` ones. They are the
#: transit-entry tool-to-module transform re-expressed in the *current* tool
#: frame and differenced against the transform recorded at entry, so a module
#: carried rigidly reads zero however far the arm flies, and a module that slips
#: in the pads reads the slip, regardless of where the arm happens to be.
TRANSIT_TRACE_FIELDS = (
    "step",
    "env",
    "steps_since_transit_start",
    "waypoint_read",
    "grip_error_m",
    "grip_attitude_rad",
    "finger_angle_rad",
    "drive_torque_nm",
    "latch_engaged",
    "tool_to_module_drift_m",
    "tool_to_module_drift_rad",
    "tool_to_module_x_drift_m",
    "tool_to_module_y_drift_m",
    "tool_to_module_z_drift_m",
    "tool_travel_m",
    "module_travel_m",
    "blade_x_m",
    "blade_y_m",
    "blade_z_m",
    "tool_x_m",
    "tool_y_m",
    "tool_z_m",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
)
#: Control steps between recorded transit-retention samples. Every second step
#: at 30 Hz is 15 Hz, fast enough to catch the instant a grip lets go -- the
#: measured loss on the passive interface happens inside the first two seconds --
#: and cheap enough to leave on for a many-environment batch.
TRANSIT_TRACE_STRIDE = 2
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
        self,
        task,
        policies,
        workflow: str,
        transit_slowdown: int,
        max_steps: int,
        tracing: bool = False,
        release_latch_required: bool = False,
        base_rail_enabled: bool = False,
        base_rail_arm_mode: str = "ik_attitude",
    ) -> None:
        self.task = task
        self.policies = policies
        self.workflow = workflow
        self.transit_slowdown = max(1, transit_slowdown)
        self.release_latch_required = release_latch_required
        #: Whether this run carries the module on a rigid robot-side form lock.
        #: The two transit controllers below are selected on it and nothing
        #: else, so every previously certified path is bit-identical.
        self.rigid_transit = release_latch_required and not base_rail_enabled and workflow == "relocate"
        self.base_rail_enabled = base_rail_enabled
        self.base_rail_joint_hold = base_rail_arm_mode == "joint_hold"
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
            float("inf") if index in (SEAT, TRANSIT, DONE) else budget for index, budget in enumerate(PHASE_BUDGET_S)
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
        #: The three module poses the rigid transit drives through, and the
        #: attitude to hold at each. Module poses rather than tool poses,
        #: because the collision clearances that decide them are about the
        #: module and the rack, not about the wrist.
        self.module_leg_pos = torch.zeros((RIGID_TRANSIT_LEGS, count, 3), device=device)
        self.module_leg_rot = torch.zeros((RIGID_TRANSIT_LEGS, count, 4), device=device)
        self.module_leg_rot[..., 0] = 1.0
        self.transit_leg_entered = torch.zeros(count, dtype=torch.long, device=device)
        self.transit_leg_forced = torch.zeros((RIGID_TRANSIT_LEGS, count), dtype=torch.bool, device=device)
        self.transit_leg_residual_rad = torch.zeros((RIGID_TRANSIT_LEGS, count), device=device)
        self.transit_leg_residual_m = torch.zeros((RIGID_TRANSIT_LEGS, count), device=device)
        self.guarded_insert_target_x = torch.zeros(count, device=device)
        self.guarded_insert_release_x = torch.zeros(count, device=device)
        self.guarded_insert_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.guarded_insert_holds = torch.zeros(count, dtype=torch.long, device=device)
        #: What the retreat was actually laid out for, and why.
        self.transit_clear_centre_x = torch.zeros(count, device=device)
        self.transit_front_overhang_m = torch.zeros(count, device=device)
        # --- robot-carried retention evidence ---------------------------------
        # The tool-to-module transform recorded at the instant the transit
        # begins, and the running worst deviation from it. This is the whole
        # measurement that distinguishes a robot that carries a module from one
        # that lets a carrier take it: if the robot is carrying, this transform
        # is preserved, and if it is not, this is the number that says so and
        # when.
        self.transit_reference_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.transit_reference_pos_tool = torch.zeros((count, 3), device=device)
        self.transit_reference_rot_tool = torch.zeros((count, 4), device=device)
        self.transit_reference_rot_tool[:, 0] = 1.0
        self.transit_entry_tool_pos = torch.zeros((count, 3), device=device)
        self.transit_entry_blade_pos = torch.zeros((count, 3), device=device)
        self.transit_max_drift_m = torch.zeros(count, device=device)
        self.transit_max_drift_rad = torch.zeros(count, device=device)
        self.transit_final_drift_m = torch.zeros(count, device=device)
        self.transit_final_drift_rad = torch.zeros(count, device=device)
        self.transit_max_grip_error_m = torch.zeros(count, device=device)
        self.transit_max_grip_attitude_rad = torch.zeros(count, device=device)
        self.transit_min_drive_torque_nm = torch.full((count,), float("inf"), device=device)
        self.transit_tool_travel_m = torch.zeros(count, device=device)
        self.transit_module_travel_m = torch.zeros(count, device=device)
        self.transit_samples = torch.zeros(count, dtype=torch.long, device=device)
        #: Driver step at which the drift first exceeded the retention limit, or
        #: -1. A pooled maximum cannot say whether a grip failed at the first
        #: retreat or at the last millimetre of the approach, and those are
        #: different faults with different fixes.
        self.transit_loss_step = torch.full((count,), -1, dtype=torch.long, device=device)
        self.transit_rows: list[np.ndarray] = []
        self.predicate_fired = torch.zeros(count, dtype=torch.bool, device=device)
        self.judged = torch.zeros(count, dtype=torch.bool, device=device)
        self.outcome = torch.zeros(count, dtype=torch.bool, device=device)
        self.all_conditions = torch.zeros(count, dtype=torch.bool, device=device)
        # Tool positions visited during the pull, sampled so the transit can fly
        # them backwards. Every one of them was reachable a moment ago.
        self.max_waypoints = max(1, max_steps // TRANSIT_WAYPOINT_STRIDE + 2)
        self.waypoints = torch.zeros((self.max_waypoints, count, 3), device=device)
        self.extraction_joint_waypoints = torch.zeros((self.max_waypoints, count, 6), device=device)
        self.extraction_blade_pose_waypoints = torch.zeros((self.max_waypoints, count, 7), device=device)
        self.relocation_staging_pos = torch.tensor(RELOCATION_INSERT_STAGING_POS, device=device)
        self.relocation_staging_rot = torch.tensor(RELOCATION_INSERT_STAGING_ROT, device=device)
        self.relocation_hold_tool_rot = torch.zeros((count, 4), device=device)
        self.relocation_hold_tool_rot[:, 0] = 1.0
        self.relocation_blade_relative_to_tool = torch.zeros((count, 3), device=device)
        self.relocation_blade_relative_rot_to_tool = torch.zeros((count, 4), device=device)
        self.relocation_blade_relative_rot_to_tool[:, 0] = 1.0
        self.relocation_desired_tool_rot = torch.zeros((count, 4), device=device)
        self.relocation_desired_tool_rot[:, 0] = 1.0
        self.relocation_alignment_tool_pos = torch.zeros((count, 3), device=device)
        self.relocation_final_tool_hold = torch.zeros((count, 3), device=device)
        self.relocation_final_tool_aligned = torch.zeros((count, 3), device=device)
        self.relocation_aligning = torch.zeros(count, dtype=torch.bool, device=device)
        self.relocation_aligned = torch.zeros(count, dtype=torch.bool, device=device)
        self.relocation_stage_retreat_done = torch.zeros(count, dtype=torch.bool, device=device)
        self.relocation_stage_lateral_done = torch.zeros(count, dtype=torch.bool, device=device)
        self.relocation_stage_attitude_done = torch.zeros(count, dtype=torch.bool, device=device)
        self.relocation_stage_translated = torch.zeros(count, dtype=torch.bool, device=device)
        self.relocation_joint_replaying = torch.zeros(count, dtype=torch.bool, device=device)
        self.relocation_joint_replay_index = torch.zeros(count, dtype=torch.long, device=device)
        self.relocation_joint_replay_stop = torch.zeros(count, dtype=torch.long, device=device)
        self.relocation_joint_replay_steps = torch.zeros(count, dtype=torch.long, device=device)
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
        # Initial visual rack-state decision.  A relocation request is fixed as
        # bay 0 -> bay 1 by this task profile; the learned occupancy branch must
        # confirm that precondition before manipulation continues.
        self.plan_checked = torch.zeros(count, dtype=torch.bool, device=device)
        self.plan_passed = torch.zeros(count, dtype=torch.bool, device=device)
        self.initial_occupancy_scores = torch.full((count, 2), float("nan"), device=device)
        # Run-level latch evidence deliberately survives environment resets. A
        # single diagnostic often reaches the task timeout, whose auto-reset
        # clears the event term before the JSON report is formatted; without
        # this accumulator the report would say "never engaged" precisely when
        # the terminal run is the one we need to inspect.
        self.latch_ever_engaged = torch.zeros(count, dtype=torch.bool, device=device)
        #: Whether the driver commanded the form lock to let go, and when. A
        #: latch that is never released is a weld, and the report has to be able
        #: to tell the two apart.
        self.latch_released = torch.zeros(count, dtype=torch.bool, device=device)
        self.latch_released_at = torch.full((count,), -1, dtype=torch.long, device=device)
        #: And when it stopped being a weld and became a spring, which is a
        #: different event from letting go and has to be in the record as one.
        self.latch_softened = torch.zeros(count, dtype=torch.bool, device=device)
        self.latch_softened_at = torch.full((count,), -1, dtype=torch.long, device=device)
        #: And whether the *hand* let go, which is a different event and the one
        #: the acceptance rule is written about: the fingers open only after the
        #: module has passed depth, lateral, attitude, velocity, and the 0.70 s
        #: settling re-check. Before this the chain simply held on for ever,
        #: which is a defensible way to end a batch run and not a way to end a
        #: demonstration of a servicing operation.
        self.gripper_released = torch.zeros(count, dtype=torch.bool, device=device)
        self.gripper_released_at = torch.full((count,), -1, dtype=torch.long, device=device)
        self.latch_first_engagement_episode_step = torch.full((count,), -1, dtype=torch.long, device=device)
        self.latch_max_position_error_m = torch.zeros(count, device=device)
        self.latch_max_orientation_error_rad = torch.zeros(count, device=device)
        self.latch_max_applied_force_n = torch.zeros(count, device=device)
        self.latch_max_applied_torque_nm = torch.zeros(count, device=device)
        self.latch_force_saturation_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.latch_seek_travel_m = torch.zeros(count, device=device)
        self.latch_seek_refusals = torch.zeros(count, dtype=torch.long, device=device)
        self.latch_torque_saturation_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.rail_commanded_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.stage_drive_target_m = torch.zeros((count, 3), device=device)
        self.stage_goal_target_m = torch.zeros((count, 3), device=device)
        # Angular D6 drive targets are expressed in degrees by USD.  They are
        # kept separately from the SI translation targets so a report cannot
        # accidentally mix units.
        self.stage_rotation_drive_target_deg = torch.zeros((count, 3), device=device)
        self.payload_stage_engaged = torch.zeros(count, dtype=torch.bool, device=device)
        self.payload_stage_capture_pos = torch.zeros((count, 3), device=device)
        self.payload_stage_capture_rot = torch.zeros((count, 4), device=device)
        self.payload_stage_capture_rot[:, 0] = 1.0
        self.payload_stage_insert_hold = torch.zeros(count, dtype=torch.long, device=device)
        self.payload_stage_control_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.payload_stage_last_error_world = torch.zeros((count, 3), device=device)
        self.payload_stage_last_error_stage = torch.zeros((count, 3), device=device)
        # Compatibility aliases for evidence/tests written while the stage had
        # only one driven rail axis.
        self.rail_drive_target_m = self.stage_drive_target_m[:, 1]
        self.rail_goal_target_m = self.stage_goal_target_m[:, 1]
        self.rail_max_mount_deflection_m = torch.zeros(count, device=device)
        self.rail_max_mount_translation_axis_m = torch.zeros(count, device=device)
        self.rail_max_mount_rotation_axis_rad = torch.zeros(count, device=device)
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
        self.payload_stage_joints: list[UsdPhysics.Joint] = []
        self.stage_drive_target_attributes: list[tuple[object, object, object]] = []
        self.stage_rotation_drive_target_attributes: list[tuple[object, object, object]] = []
        if self.base_rail_enabled:
            stage = omni.usd.get_context().get_stage()
            for index in range(count):
                joint_prim = stage.GetPrimAtPath(f"/World/envs/env_{index}/PayloadStage/Joint")
                if not joint_prim.IsValid():
                    raise RuntimeError(f"Payload stage joint is missing for env {index}")
                self.payload_stage_joints.append(UsdPhysics.Joint(joint_prim))
                attributes = []
                for axis in (UsdPhysics.Tokens.transX, UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ):
                    drive = UsdPhysics.DriveAPI.Get(joint_prim, axis)
                    target_attribute = drive.GetTargetPositionAttr()
                    if not drive or not target_attribute.IsValid():
                        raise RuntimeError(
                            f"Base stage {axis} drive is missing for env {index}; refusing direct state writes"
                        )
                    attributes.append(target_attribute)
                self.stage_drive_target_attributes.append(tuple(attributes))
                rotation_attributes = []
                for axis in (UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
                    drive = UsdPhysics.DriveAPI.Get(joint_prim, axis)
                    target_attribute = drive.GetTargetPositionAttr()
                    if not drive or not target_attribute.IsValid():
                        raise RuntimeError(
                            f"Base stage {axis} drive is missing for env {index}; refusing direct state writes"
                        )
                    rotation_attributes.append(target_attribute)
                self.stage_rotation_drive_target_attributes.append(tuple(rotation_attributes))

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
        # Only the *reference* is per-episode. The accumulators below it are
        # deliberately left alone, for the same reason the latch evidence is:
        # every environment is reset before the report is formatted, so a
        # per-episode accumulator would report an empty transit on precisely
        # the runs whose transit is the question.
        self.transit_reference_valid[env_ids] = False
        self.transit_leg_entered[env_ids] = 0
        self.guarded_insert_target_x[env_ids] = 0.0
        self.guarded_insert_release_x[env_ids] = 0.0
        self.guarded_insert_steps[env_ids] = 0
        self.guarded_insert_holds[env_ids] = 0
        self.latch_released[env_ids] = False
        self.latch_released_at[env_ids] = -1
        self.latch_softened[env_ids] = False
        self.latch_softened_at[env_ids] = -1
        self.gripper_released[env_ids] = False
        self.gripper_released_at[env_ids] = -1
        self.transit_reference_pos_tool[env_ids] = 0.0
        self.transit_reference_rot_tool[env_ids] = 0.0
        self.transit_reference_rot_tool[env_ids, 0] = 1.0
        self.transit_entry_tool_pos[env_ids] = 0.0
        self.transit_entry_blade_pos[env_ids] = 0.0
        self.perceived_error_sum[env_ids] = 0.0
        self.perceived_error_steps[env_ids] = 0.0
        self.perceived_error_max[env_ids] = 0.0
        self.plan_checked[env_ids] = False
        self.plan_passed[env_ids] = False
        self.initial_occupancy_scores[env_ids] = float("nan")
        self.relocation_aligning[env_ids] = False
        self.relocation_aligned[env_ids] = False
        self.relocation_stage_retreat_done[env_ids] = False
        self.relocation_stage_lateral_done[env_ids] = False
        self.relocation_stage_attitude_done[env_ids] = False
        self.relocation_stage_translated[env_ids] = False
        self.relocation_joint_replaying[env_ids] = False
        self.relocation_joint_replay_index[env_ids] = 0
        self.relocation_joint_replay_stop[env_ids] = 0
        self.relocation_joint_replay_steps[env_ids] = 0
        self.predicate_fired[env_ids] = False
        self.judged[env_ids] = False
        self.outcome[env_ids] = False
        self.all_conditions[env_ids] = False
        self.waypoint_write[env_ids] = 0
        self.waypoint_read[env_ids] = 0
        self.frozen_valid[env_ids] = False
        self.reset_tool_valid[env_ids] = False
        self.rail_commanded_steps[env_ids] = 0
        self.stage_drive_target_m[env_ids] = 0.0
        self.stage_goal_target_m[env_ids] = 0.0
        self.stage_rotation_drive_target_deg[env_ids] = 0.0
        self.payload_stage_engaged[env_ids] = False
        self.payload_stage_insert_hold[env_ids] = 0
        self.payload_stage_control_steps[env_ids] = 0
        self.payload_stage_last_error_world[env_ids] = 0.0
        self.payload_stage_last_error_stage[env_ids] = 0.0
        self.rail_max_mount_deflection_m[env_ids] = 0.0
        self.rail_max_mount_translation_axis_m[env_ids] = 0.0
        self.rail_max_mount_rotation_axis_rad[env_ids] = 0.0
        if self.base_rail_enabled:
            self._set_stage_arm_servo(env_ids, strengthened=False)
            for index in env_ids.detach().cpu().tolist():
                self.payload_stage_joints[index].GetJointEnabledAttr().Set(False)
                for attribute in self.stage_drive_target_attributes[index]:
                    attribute.Set(0.0)
                for attribute in self.stage_rotation_drive_target_attributes[index]:
                    attribute.Set(0.0)

    def _apply_scales(self) -> None:
        self.arm._scale[:] = self.scales[self.phase]

    def _set_stage_arm_servo(self, env_ids: torch.Tensor, *, strengthened: bool) -> None:
        """Switch the arm gains while it transfers the ORU to the shuttle."""

        if env_ids.numel() == 0:
            return
        robot = self.task.scene["robot"]
        defaults_k = robot.data.default_joint_stiffness[env_ids][:, self.arm_joint_ids]
        defaults_d = robot.data.default_joint_damping[env_ids][:, self.arm_joint_ids]
        multiplier = BASE_STAGE_ARM_STIFFNESS_MULTIPLIER if strengthened else 1.0
        robot.write_joint_stiffness_to_sim(
            defaults_k * multiplier,
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )
        robot.write_joint_damping_to_sim(
            defaults_d * math.sqrt(multiplier),
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )

    def _engage_payload_stage(self, env_ids: torch.Tensor) -> None:
        """Transfer the extracted ORU to the physical D6 service shuttle."""

        if env_ids.numel() == 0:
            return
        task = self.task
        anchor = task.scene["mount_anchor"]
        blade = task.scene["spare_blade"]
        inverse_anchor = quat_inv(anchor.data.root_quat_w)
        local_position = quat_apply(inverse_anchor, blade.data.root_pos_w - anchor.data.root_pos_w)
        blade_to_world_aligned_joint = quat_inv(blade.data.root_quat_w)
        stage = omni.usd.get_context().get_stage()
        for index in env_ids.detach().cpu().tolist():
            position = local_position[index].detach().cpu().tolist()
            orientation = blade_to_world_aligned_joint[index].detach().cpu().tolist()
            joint = self.payload_stage_joints[index]
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.GetLocalRot1Attr().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
            joint.GetJointEnabledAttr().Set(True)
            release_joint = UsdPhysics.FixedJoint(
                stage.GetPrimAtPath(f"/World/envs/env_{index}/ReleaseLatchJoint/Joint")
            )
            if release_joint and release_joint.GetPrim().IsValid():
                release_joint.GetJointEnabledAttr().Set(False)
        self.payload_stage_capture_pos[env_ids] = blade.data.root_pos_w[env_ids] - task.scene.env_origins[env_ids]
        self.payload_stage_capture_rot[env_ids] = blade.data.root_quat_w[env_ids]
        self.payload_stage_engaged[env_ids] = True
        estimator = getattr(task, "_module_state_estimator", None)
        if estimator is not None:
            estimator.mark_payload_stage_engaged(env_ids)
        if hasattr(task, "_grapple_latched"):
            task._grapple_latched[env_ids] = False
        if hasattr(task, "_grapple_latch_armed"):
            task._grapple_latch_armed[env_ids] = False

    def _payload_feedback(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local pose and velocity from the deployed estimator when present.

        The state task has no estimator and keeps exact simulator feedback.  A
        vision task uses the same cached RGB estimate as its learned policies,
        including for shuttle alignment and the guarded insertion predicate.
        Simulator pose remains available only to final diagnostic scoring.
        """

        estimator = getattr(self.task, "_module_state_estimator", None)
        if estimator is not None:
            pose, velocity = estimator.estimate()
            pose_wxyz = estimator.pose_wxyz()
            return pose[:, :3], pose_wxyz[:, 3:7], velocity
        blade_position, blade_orientation = attached_blade_pose_world(self.task)
        return (
            blade_position - self.task.scene.env_origins,
            blade_orientation,
            attached_blade_velocity(self.task),
        )

    def _trace_state(self) -> dict[str, torch.Tensor]:
        """Every quantity both traces share, read once per step."""

        task = self.task
        grip_error, grip_attitude = grapple_grip_error_metrics(task)
        blade_position, blade_orientation = attached_blade_pose_world(task)
        velocity = attached_blade_velocity(task)
        latch = grapple_latch_diagnostics(task)
        return {
            "grip_error_m": grip_error.to(torch.float64),
            "grip_attitude_rad": grip_attitude.to(torch.float64),
            "finger_angle_rad": grip_finger_angle(task).to(torch.float64),
            "drive_torque_nm": grip_drive_torque(task).to(torch.float64),
            "latch_engaged": latch["engaged"].to(torch.float64),
            "latch_relative_position_error_m": latch["position_error_m"].to(torch.float64),
            "latch_relative_orientation_error_rad": latch["orientation_error_rad"].to(torch.float64),
            "latch_applied_force_n": latch["applied_force_n"].to(torch.float64),
            "latch_applied_torque_nm": latch["applied_torque_nm"].to(torch.float64),
            "latch_force_saturated": latch["force_saturated"].to(torch.float64),
            "latch_torque_saturated": latch["torque_saturated"].to(torch.float64),
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
                state["latch_engaged"].unsqueeze(-1),
                state["latch_relative_position_error_m"].unsqueeze(-1),
                state["latch_relative_orientation_error_rad"].unsqueeze(-1),
                state["latch_applied_force_n"].unsqueeze(-1),
                state["latch_applied_torque_nm"].unsqueeze(-1),
                state["latch_force_saturated"].unsqueeze(-1),
                state["latch_torque_saturated"].unsqueeze(-1),
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
            "transit": stack(self.transit_rows, TRANSIT_TRACE_FIELDS),
            "transit_fields": np.asarray(TRANSIT_TRACE_FIELDS),
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

    def _observe_latch(self) -> None:
        """Accumulate latch evidence before a task timeout can reset the term."""

        latch = grapple_latch_diagnostics(self.task)
        engaged = latch["engaged"]
        newly_observed = engaged & ~self.latch_ever_engaged
        self.latch_first_engagement_episode_step[newly_observed] = latch["first_engagement_episode_step"][
            newly_observed
        ]
        self.latch_ever_engaged |= engaged
        self.latch_seek_travel_m[newly_observed] = latch["seek_travel_m"][newly_observed]
        self.latch_seek_refusals = torch.maximum(self.latch_seek_refusals, latch["seek_refusals"])
        self.latch_max_position_error_m = torch.maximum(self.latch_max_position_error_m, latch["max_position_error_m"])
        self.latch_max_orientation_error_rad = torch.maximum(
            self.latch_max_orientation_error_rad, latch["max_orientation_error_rad"]
        )
        self.latch_max_applied_force_n = torch.maximum(self.latch_max_applied_force_n, latch["max_applied_force_n"])
        self.latch_max_applied_torque_nm = torch.maximum(
            self.latch_max_applied_torque_nm, latch["max_applied_torque_nm"]
        )
        self.latch_force_saturation_steps += latch["force_saturated"].to(torch.long)
        self.latch_torque_saturation_steps += latch["torque_saturated"].to(torch.long)
        robot = self.task.scene["robot"]
        anchor = self.task.scene["mount_anchor"]
        mount_translation_error = robot.data.root_pos_w - anchor.data.root_pos_w
        mount_rotation = axis_angle_from_quat(quat_mul(anchor.data.root_quat_w, quat_inv(robot.data.root_quat_w)))
        if self.base_rail_enabled:
            blade = self.task.scene["spare_blade"]
            blade_local = blade.data.root_pos_w - self.task.scene.env_origins
            actual_travel = blade_local - self.payload_stage_capture_pos
            stage_tracking_error = actual_travel - self.stage_drive_target_m
            mount_translation_error = torch.where(
                self.payload_stage_engaged.unsqueeze(-1),
                stage_tracking_error,
                torch.zeros_like(stage_tracking_error),
            )
            mount_rotation = torch.where(
                self.payload_stage_engaged.unsqueeze(-1),
                axis_angle_from_quat(quat_mul(blade.data.root_quat_w, quat_inv(self.payload_stage_capture_rot))),
                torch.zeros_like(mount_rotation),
            )
        mount_deflection = torch.linalg.vector_norm(mount_translation_error, dim=-1)
        self.rail_max_mount_deflection_m = torch.maximum(
            self.rail_max_mount_deflection_m,
            mount_deflection,
        )
        self.rail_max_mount_translation_axis_m = torch.maximum(
            self.rail_max_mount_translation_axis_m,
            mount_translation_error.abs().amax(dim=-1),
        )
        self.rail_max_mount_rotation_axis_rad = torch.maximum(
            self.rail_max_mount_rotation_axis_rad,
            mount_rotation.abs().amax(dim=-1),
        )

    def _begin_transit_reference(self, mask, tool, tool_rot) -> None:
        """Record the tool-to-module transform the transit is planned from.

        This is the same transform ``_plan_lateral_transit`` measures and builds
        every remaining waypoint out of. Recording it here, in the tool's own
        frame, is what makes the drift below a statement about the *grip* rather
        than about where the arm happens to be: the arm is supposed to move.
        """

        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        blade = self.task.scene["spare_blade"]
        inverse_tool = quat_inv(tool_rot[ids])
        self.transit_reference_pos_tool[ids] = quat_apply(inverse_tool, blade.data.root_pos_w[ids] - tool[ids])
        self.transit_reference_rot_tool[ids] = quat_mul(inverse_tool, blade.data.root_quat_w[ids])
        self.transit_entry_tool_pos[ids] = tool[ids]
        self.transit_entry_blade_pos[ids] = blade.data.root_pos_w[ids]
        self.transit_reference_valid[ids] = True

    def _observe_transit_retention(self, transiting, step: int, tool, tool_rot) -> None:
        """Accumulate, and optionally record, the carried-module evidence.

        Runs every control step of every transit whether or not a trace file was
        asked for, because the summary it feeds is an acceptance gate, and an
        acceptance gate that depends on a diagnostic flag being passed is not
        one.
        """

        active = transiting & self.transit_reference_valid
        if not bool(active.any()):
            return
        task = self.task
        blade = task.scene["spare_blade"]
        inverse_tool = quat_inv(tool_rot)
        relative_pos = quat_apply(inverse_tool, blade.data.root_pos_w - tool)
        relative_rot = quat_mul(inverse_tool, blade.data.root_quat_w)
        drift_vector = relative_pos - self.transit_reference_pos_tool
        drift_m = torch.linalg.vector_norm(drift_vector, dim=-1)
        drift_rad = torch.linalg.vector_norm(
            axis_angle_from_quat(quat_mul(relative_rot, quat_inv(self.transit_reference_rot_tool))),
            dim=-1,
        )
        tool_travel = torch.linalg.vector_norm(tool - self.transit_entry_tool_pos, dim=-1)
        module_travel = torch.linalg.vector_norm(blade.data.root_pos_w - self.transit_entry_blade_pos, dim=-1)
        grip_error, grip_attitude = grapple_grip_error_metrics(task)
        drive_torque = grip_drive_torque(task)
        zero = torch.zeros_like(drift_m)
        self.transit_max_drift_m = torch.maximum(self.transit_max_drift_m, torch.where(active, drift_m, zero))
        self.transit_max_drift_rad = torch.maximum(self.transit_max_drift_rad, torch.where(active, drift_rad, zero))
        self.transit_final_drift_m = torch.where(active, drift_m, self.transit_final_drift_m)
        self.transit_final_drift_rad = torch.where(active, drift_rad, self.transit_final_drift_rad)
        self.transit_max_grip_error_m = torch.maximum(
            self.transit_max_grip_error_m, torch.where(active, grip_error, zero)
        )
        self.transit_max_grip_attitude_rad = torch.maximum(
            self.transit_max_grip_attitude_rad, torch.where(active, grip_attitude, zero)
        )
        self.transit_min_drive_torque_nm = torch.where(
            active,
            torch.minimum(self.transit_min_drive_torque_nm, drive_torque),
            self.transit_min_drive_torque_nm,
        )
        self.transit_tool_travel_m = torch.where(active, tool_travel, self.transit_tool_travel_m)
        self.transit_module_travel_m = torch.where(active, module_travel, self.transit_module_travel_m)
        self.transit_samples = torch.where(active, self.transit_samples + 1, self.transit_samples)
        lost = (
            active
            & (self.transit_loss_step < 0)
            & (
                (drift_m > TRANSIT_RETENTION_POSITION_LIMIT_M)
                | (drift_rad > TRANSIT_RETENTION_ORIENTATION_LIMIT_RAD)
            )
        )
        self.transit_loss_step[lost] = step
        if not self.tracing or (step % TRANSIT_TRACE_STRIDE) != 0:
            return
        latch = grapple_latch_diagnostics(task)
        velocity = attached_blade_velocity(task)
        blade_local = blade.data.root_pos_w - task.scene.env_origins
        tool_local = tool - task.scene.env_origins
        rows = torch.cat(
            (
                self._column(float(step)),
                self.env_index.unsqueeze(-1),
                (step - self.transit_started).to(torch.float64).unsqueeze(-1),
                self.waypoint_read.to(torch.float64).unsqueeze(-1),
                grip_error.to(torch.float64).unsqueeze(-1),
                grip_attitude.to(torch.float64).unsqueeze(-1),
                grip_finger_angle(task).to(torch.float64).unsqueeze(-1),
                drive_torque.to(torch.float64).unsqueeze(-1),
                latch["engaged"].to(torch.float64).unsqueeze(-1),
                drift_m.to(torch.float64).unsqueeze(-1),
                drift_rad.to(torch.float64).unsqueeze(-1),
                drift_vector.to(torch.float64),
                tool_travel.to(torch.float64).unsqueeze(-1),
                module_travel.to(torch.float64).unsqueeze(-1),
                blade_local.to(torch.float64),
                tool_local.to(torch.float64),
                torch.linalg.vector_norm(velocity[:, :3], dim=-1).to(torch.float64).unsqueeze(-1),
                torch.linalg.vector_norm(velocity[:, 3:], dim=-1).to(torch.float64).unsqueeze(-1),
            ),
            dim=-1,
        )
        self.transit_rows.append(rows[active].cpu().numpy())

    def step(self, step: int) -> None:
        """Compute one action for every environment and advance the phase machine."""

        task = self.task
        observations = task.observation_manager.compute()
        self._observe_latch()
        plan_blocked = torch.zeros_like(self.plan_checked)
        if self.workflow == "relocate":
            estimator = getattr(task, "_module_state_estimator", None)
            ready = ~self.plan_checked & (task.episode_length_buf >= 2)
            if estimator is not None and estimator.backend == "fiducial_pnp":
                # Reset observations precede the first rendered camera frame.
                # Keep the robot frozen until a genuine detection exists;
                # zero-confidence initialization priors may never approve the
                # source/destination plan.
                ready &= estimator.confidence > 0.0
            if estimator is not None and bool(ready.any()):
                occupancy = estimator.occupancy_probabilities()
                if occupancy is None or occupancy.shape[1] != 2:
                    raise RuntimeError(
                        "The visual two-bay relocation requires a two-output occupancy head; "
                        "refusing to run a hard-coded bay plan without the perception preflight."
                    )
                self.initial_occupancy_scores[ready] = occupancy[ready]
                accepted = (occupancy[:, 0] >= OCCUPANCY_PLAN_THRESHOLD) & (occupancy[:, 1] < OCCUPANCY_PLAN_THRESHOLD)
                self.plan_checked[ready] = True
                self.plan_passed[ready] = accepted[ready]
                # Camera warm-up is a preflight, not part of the capture
                # policy's execution budget.  Start that clock only after the
                # requested rack state has been accepted.
                self.phase_started[ready & accepted] = step
                rejected = ready & ~accepted
                if bool(rejected.any()):
                    # A perception-to-planning failure is a terminal workflow
                    # result, not permission to fall back to the scenario's
                    # simulator-known reset bay.
                    self._finish(rejected, step)
                first = int(torch.nonzero(ready, as_tuple=False)[0, 0].item())
                print(
                    "[PLAN] visual occupancy preflight "
                    f"passed={int(accepted[ready].sum())}/{int(ready.sum())} "
                    "source=bay0 destination=bay1 "
                    f"scores=[{float(occupancy[first, 0]):.4f},{float(occupancy[first, 1]):.4f}] "
                    f"threshold={OCCUPANCY_PLAN_THRESHOLD:.2f}",
                    flush=True,
                )
            if estimator is not None:
                # No arm or gripper command may run while the camera is warming
                # up, or after the visual rack-state request is rejected.  The
                # earlier implementation called the gate a preflight but still
                # ran the capture policy for two steps and closed the gripper on
                # the rejection step.
                plan_blocked = ~self.plan_checked | ~self.plan_passed
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

        capturing = (phase == CAPTURE) & ~plan_blocked
        if bool(capturing.any()):
            command = self.policies["capture"].act(observations["grasp"])
            self.actions[capturing] = command[capturing]
        for name, group, mask in (
            ("extract", "extract", (phase == EXTRACT) & ~plan_blocked),
            (
                "insert",
                "insert",
                (phase == INSERT) & ~plan_blocked & ~(self.payload_stage_engaged & self.base_rail_enabled),
            ),
        ):
            if bool(mask.any()):
                command = self.policies[name].act(observations[group])
                self.actions[mask, :6] = command[mask]
        # Everything past the capture keeps commanding closure, so the two-stage
        # action term holds the pin instead of relaxing to the capture command.
        self.actions[~capturing & ~plan_blocked, 6] = 1.0
        # After the service shuttle accepts the ORU, the robot is no longer the
        # load path and opens its fingers.  The module remains secured by the
        # physical D6 joint throughout transport and guarded insertion.
        self.actions[self.payload_stage_engaged, 6] = -1.0

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
        self.held = torch.where(
            qualifying, self.held + 1, torch.where(capturing, torch.zeros_like(self.held), self.held)
        )
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
            self.extraction_joint_waypoints[slots, ids] = task.scene["robot"].data.joint_pos[ids][:, self.arm_joint_ids]
            blade_asset = task.scene["spare_blade"]
            self.extraction_blade_pose_waypoints[slots, ids, :3] = (
                blade_asset.data.root_pos_w[ids] - task.scene.env_origins[ids]
            )
            self.extraction_blade_pose_waypoints[slots, ids, 3:] = blade_asset.data.root_quat_w[ids]
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
                # And, if the task has a latch configured to wait for it, this is
                # the instant it engages: the rails have just let go, so a
                # restoring torque has nothing left to jam the module against.
                # A no-op on every task whose latch is off, which is all of them
                # unless one is asked for by configuration.
                arm_grapple_latch(task, cleared)
                self.phase[cleared] = TRANSIT
                self.transit_started[cleared] = step
                # Freeze the transform the whole flight is planned from, before
                # anything commands the arm, so the retention record starts at
                # the same instant the plan does.
                self._begin_transit_reference(cleared, tool, tool_rot)
                if self.workflow == "relocate":
                    # See RELOCATE_TRANSIT_HOLD: the relocation is the first
                    # phase here that moves a module through free space rather
                    # than releasing one at the end of a job, and the rule for
                    # moving is to hold.
                    # A release-time latch qualifies from the loaded capture.
                    # Do not relax the fingers in the same control step that
                    # arms it: the interval event runs after action processing,
                    # so the gentle retain can remove the torque predicate
                    # before the latch ever records its transform.  Keep the
                    # full holding closure until engagement is observed on the
                    # next driver step; latch-off behavior is unchanged.
                    if RELOCATE_TRANSIT_HOLD or self.release_latch_required:
                        self.gripper.retain_latch[cleared] = False
                    self._plan_lateral_transit(cleared, tool, tool_rot, blade_x)
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
        # Measured before the follower is allowed to command anything, so the
        # record is of the state the transit is in rather than of the state it
        # was just driven into.
        self._observe_transit_retention(transiting, step, tool, tool_rot)
        if self.rigid_transit and bool(transiting.any()):
            # **A different controller, because it is a different problem.**
            #
            # Everything in the `elif` below was written for a module held in
            # the pads, which moves in the grip while the tool moves, so the
            # follower servos the *tool* and corrects the module afterwards.
            # With the form lock engaged the module is a rigid extension of the
            # wrist, and then servoing the tool and hoping is not merely
            # unnecessary, it diverges: the alignment sub-phase computes a tool
            # target from the module's position, rotating the tool moves the
            # module, and the target is never recomputed. Measured on the first
            # rigid run, that loop walked the tool attitude from 0.66 rad to
            # 1.61 rad and dragged the module 380 mm backwards out of the cell.
            #
            # A rigid payload has a closed-form answer instead. The module's
            # pose is a fixed transform from the tool's, so a desired module
            # pose *is* a desired tool pose, and the whole transit is three
            # module waypoints and one servo. See ``_step_rigid_transit``.
            arrived = self._step_rigid_transit(transiting, step, tool, tool_rot)
            if bool(arrived.any()):
                self._begin_guarded_insert(arrived, step)
            self.gripper.retain_latch[arrived] = False
            self.phase[arrived] = INSERT
        elif bool(transiting.any()):
            ids = torch.nonzero(transiting, as_tuple=False).squeeze(-1)
            # The collision-clear retreat is part of the physical route too.
            # Originally only learned-extraction joints were recorded, so the
            # reverse replay jumped directly from the fully retreated pose to
            # the extraction hand-off.  That unsampled 80 mm Cartesian jump was
            # a 0.20 rad wrist command under load and the arm could not follow
            # it.  Sample this scripted leg at the same cadence as extraction;
            # after the bay crossing the replay is continuous end to end.
            record_retreat = (
                transiting
                & self.base_rail_enabled
                & (self.waypoint_read == 2)
                & ((step - self.transit_started) % TRANSIT_WAYPOINT_STRIDE == 0)
            )
            if bool(record_retreat.any()):
                retreat_ids = torch.nonzero(record_retreat, as_tuple=False).squeeze(-1)
                retreat_slots = self.waypoint_write[retreat_ids].clamp(max=self.max_waypoints - 1)
                robot = task.scene["robot"]
                blade_asset = task.scene["spare_blade"]
                self.extraction_joint_waypoints[retreat_slots, retreat_ids] = robot.data.joint_pos[retreat_ids][
                    :, self.arm_joint_ids
                ]
                self.extraction_blade_pose_waypoints[retreat_slots, retreat_ids, :3] = (
                    blade_asset.data.root_pos_w[retreat_ids] - task.scene.env_origins[retreat_ids]
                )
                self.extraction_blade_pose_waypoints[retreat_slots, retreat_ids, 3:] = blade_asset.data.root_quat_w[
                    retreat_ids
                ]
                self.waypoint_write[retreat_ids] = (self.waypoint_write[retreat_ids] + 1).clamp(
                    max=self.max_waypoints - 1
                )
            if self.workflow == "relocate" and bool(self.relocation_aligning.any()):
                alignment_position_error = torch.linalg.vector_norm(self.relocation_alignment_tool_pos - tool, dim=-1)
                alignment_orientation_error = torch.linalg.vector_norm(
                    axis_angle_from_quat(quat_mul(self.relocation_desired_tool_rot, quat_inv(tool_rot))),
                    dim=-1,
                )
                if self.base_rail_enabled:
                    blade_position, blade_orientation, _ = self._payload_feedback()
                    alignment_position_error = torch.linalg.vector_norm(
                        blade_position - self.relocation_staging_pos.unsqueeze(0), dim=-1
                    )
                    alignment_orientation_error = torch.linalg.vector_norm(
                        axis_angle_from_quat(
                            quat_mul(
                                self.relocation_staging_rot.unsqueeze(0).expand_as(blade_orientation),
                                quat_inv(blade_orientation),
                            )
                        ),
                        dim=-1,
                    )
                alignment_complete = (
                    transiting
                    & self.relocation_aligning
                    & (alignment_position_error <= INSERT_HANDOFF_POSITION_TOLERANCE_M)
                    & (alignment_orientation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD)
                )
                if self.base_rail_enabled:
                    alignment_complete = (
                        transiting
                        & self.relocation_aligning
                        & (alignment_orientation_error <= STAGE_ALIGNMENT_CAPTURE_RAD)
                    )
                pre_stage_alignment_complete = (
                    alignment_complete & self.base_rail_enabled & ~self.relocation_stage_translated
                )
                if bool(pre_stage_alignment_complete.any()):
                    aligned_ids = torch.nonzero(pre_stage_alignment_complete, as_tuple=False).squeeze(-1)
                    robot = task.scene["robot"]
                    anchor = task.scene["mount_anchor"]
                    current_mount = robot.data.root_pos_w[aligned_ids] - anchor.data.root_pos_w[aligned_ids]
                    blade_local = (
                        task.scene["spare_blade"].data.root_pos_w[aligned_ids] - task.scene.env_origins[aligned_ids]
                    )
                    lower = self.stage_drive_target_m.new_tensor(BASE_STAGE_MIN_TARGET_M)
                    upper = self.stage_drive_target_m.new_tensor(BASE_STAGE_MAX_TARGET_M)
                    stage_goal = current_mount.clone()
                    stage_goal[:, 1] += self.relocation_staging_pos[1] - blade_local[:, 1]
                    self.stage_goal_target_m[aligned_ids] = torch.maximum(torch.minimum(stage_goal, upper), lower)
                    self._set_stage_arm_servo(aligned_ids, strengthened=True)
                self.relocation_aligning[alignment_complete] = False
                self.relocation_aligned[alignment_complete] = True
                final_alignment_complete = alignment_complete & (not self.base_rail_enabled)
                self.waypoint_read[final_alignment_complete] = 0
                self.gripper.retain_latch[alignment_complete] = False
            if self.workflow == "relocate" and self.base_rail_enabled and bool(self.relocation_joint_replaying.any()):
                replaying = transiting & self.relocation_joint_replaying
                replay_ids = torch.nonzero(replaying, as_tuple=False).squeeze(-1)
                replay_targets = self.extraction_joint_waypoints[
                    self.relocation_joint_replay_index[replay_ids], replay_ids
                ]
                current_joints = task.scene["robot"].data.joint_pos[replay_ids][:, self.arm_joint_ids]
                replay_joint_error = (current_joints - replay_targets).abs().amax(dim=-1)
                replay_target_reached = torch.zeros_like(replaying)
                replay_target_reached[replay_ids] = replay_joint_error <= JOINT_REPLAY_CONVERGENCE_RAD
                replay_at_stop = torch.zeros_like(replaying)
                replay_at_stop[replay_ids] = (
                    self.relocation_joint_replay_index[replay_ids] <= (self.relocation_joint_replay_stop[replay_ids])
                )
                replay_due = replaying & (
                    (step - self.transit_started) % (TRANSIT_WAYPOINT_STRIDE * self.transit_slowdown) == 0
                )
                # This is a densely sampled joint trajectory, so intermediate
                # points are time-indexed actuator targets rather than separate
                # poses at which the robot should stop.  A convergence gate on
                # every sample deadlocked as soon as finite-effort tracking
                # error exceeded 0.04 rad.  Replay at the recorded cadence and
                # require physical convergence only at the terminal target.
                replay_advance = replay_due & ~replay_at_stop
                self.relocation_joint_replay_index[replay_advance] -= 1
                self.relocation_joint_replay_steps[replaying] += 1
                replay_complete_ids = replay_ids[replay_at_stop[replay_ids] & replay_target_reached[replay_ids]]
                if replay_complete_ids.numel() > 0:
                    self._set_stage_arm_servo(replay_complete_ids, strengthened=False)
                    self.relocation_joint_replaying[replay_complete_ids] = False
                    self.relocation_aligned[replay_complete_ids] = True
                    self.waypoint_read[replay_complete_ids] = 0
                    self.gripper.retain_latch[replay_complete_ids] = False
            if self.workflow == "relocate" and self.release_latch_required:
                latch_engaged = grapple_latched(task)
                awaiting_latch = transiting & ~latch_engaged
                self.gripper.retain_latch[awaiting_latch] = False
                # Gentle retain is safe only after the form lock has captured
                # the loaded transform, and only before the final rail-contact
                # leg.  The last leg's existing handoff below switches back to
                # the full holding closure.
                may_retain = transiting & latch_engaged & (self.waypoint_read > 0) & ~self.relocation_joint_replaying
                if not RELOCATE_TRANSIT_HOLD:
                    self.gripper.retain_latch[may_retain] = True
            target = self.waypoints[self.waypoint_read[ids], ids]
            if self.workflow == "relocate":
                aligning_ids = self.relocation_aligning[ids]
                aligned_final_ids = self.relocation_aligned[ids] & (self.waypoint_read[ids] <= 0)
                target = torch.where(
                    aligning_ids.unsqueeze(-1),
                    self.relocation_alignment_tool_pos[ids],
                    torch.where(
                        aligned_final_ids.unsqueeze(-1),
                        self.relocation_final_tool_aligned[ids],
                        target,
                    ),
                )
            if self.workflow == "relocate":
                # Once the normal arm follower has pulled the payload clear of
                # the rack, freeze the *measured* extraction pose and move the
                # physical lateral carriage.  Trying to rotate to a nominal
                # rack attitude here created an unreachable six-axis IK target
                # (measured residual: 40 mm / 0.072 rad) and prevented the
                # carriage from ever starting.  The two bays are parallel, so
                # the source extraction pose is already the correct attitude;
                # after the lateral move we can reverse the joint path that was
                # actually flown under load.
                begin_stage_alignment = (
                    transiting
                    & (self.waypoint_read == 1)
                    & ~self.relocation_aligning
                    & ~self.relocation_aligned
                    & ~self.relocation_stage_translated
                    & self.base_rail_enabled
                )
                if bool(begin_stage_alignment.any()):
                    stage_ids = torch.nonzero(begin_stage_alignment, as_tuple=False).squeeze(-1)
                    robot = task.scene["robot"]
                    # The arm has just executed the collision-clear retreat,
                    # which is deliberately beyond the learned extraction
                    # terminal state.  Record that physically reached joint
                    # pose as the first reverse-replay sample.  Without it the
                    # first replay command jumped roughly 80 mm back toward the
                    # rack and stalled 0.068 rad from its target under load.
                    retreat_slots = self.waypoint_write[stage_ids].clamp(max=self.max_waypoints - 1)
                    self.extraction_joint_waypoints[retreat_slots, stage_ids] = robot.data.joint_pos[stage_ids][
                        :, self.arm_joint_ids
                    ]
                    blade_asset = task.scene["spare_blade"]
                    self.extraction_blade_pose_waypoints[retreat_slots, stage_ids, :3] = (
                        blade_asset.data.root_pos_w[stage_ids] - task.scene.env_origins[stage_ids]
                    )
                    self.extraction_blade_pose_waypoints[retreat_slots, stage_ids, 3:] = blade_asset.data.root_quat_w[
                        stage_ids
                    ]
                    self.waypoint_write[stage_ids] = (self.waypoint_write[stage_ids] + 1).clamp(
                        max=self.max_waypoints - 1
                    )
                    blade_local = (
                        task.scene["spare_blade"].data.root_pos_w[stage_ids] - task.scene.env_origins[stage_ids]
                    )
                    self._engage_payload_stage(stage_ids)
                    stage_goal = torch.zeros_like(blade_local)
                    stage_goal[:, 1] = self.relocation_staging_pos[1] - blade_local[:, 1]
                    lower = self.stage_drive_target_m.new_tensor(BASE_STAGE_MIN_TARGET_M)
                    upper = self.stage_drive_target_m.new_tensor(BASE_STAGE_MAX_TARGET_M)
                    self.stage_goal_target_m[stage_ids] = torch.maximum(torch.minimum(stage_goal, upper), lower)
                    self._set_stage_arm_servo(stage_ids, strengthened=False)
                    self.relocation_aligned[begin_stage_alignment] = True
                    self.gripper.retain_latch[begin_stage_alignment] = False
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
                if self.base_rail_enabled:
                    stage_leg = self.waypoint_read[ids] == 1
                    # Compliance is a load-dependent inner-drive offset, not a
                    # payload positioning error: the outer loop below closes
                    # directly on the carried module.
                    payload_position, payload_orientation, payload_velocity = self._payload_feedback()
                    estimator = getattr(task, "_module_state_estimator", None)
                    sensor_ready = torch.ones_like(ids, dtype=torch.bool)
                    if estimator is not None and estimator.backend == "fiducial_pnp":
                        sensor_ready = estimator.fiducial_current_detection[ids]
                    stage_blade_position = payload_position[ids]
                    stage_blade_y_error = (stage_blade_position[:, 1] - self.relocation_staging_pos[1]).abs()
                    stage_blade_position_error = torch.linalg.vector_norm(
                        stage_blade_position - self.relocation_staging_pos.unsqueeze(0), dim=-1
                    )
                    stage_blade_orientation = payload_orientation[ids]
                    stage_blade_orientation_error = torch.linalg.vector_norm(
                        axis_angle_from_quat(
                            quat_mul(
                                self.relocation_staging_rot.unsqueeze(0).expand_as(stage_blade_orientation),
                                quat_inv(stage_blade_orientation),
                            )
                        ),
                        dim=-1,
                    )
                    stage_velocity = payload_velocity[ids]
                    retreat_stage_ready = (
                        sensor_ready
                        & self.relocation_aligned[ids]
                        & ~self.relocation_stage_retreat_done[ids]
                        & ((stage_blade_position[:, 0] - TRANSIT_CLEAR_BLADE_CENTRE_X).abs() <= 0.005)
                        & (torch.linalg.vector_norm(stage_velocity[:, :3], dim=-1) <= 0.060)
                    )
                    lateral_stage_ready = (
                        sensor_ready
                        & self.relocation_aligned[ids]
                        & self.relocation_stage_retreat_done[ids]
                        & ~self.relocation_stage_lateral_done[ids]
                        & ~self.relocation_stage_translated[ids]
                        & (stage_blade_y_error <= 0.035)
                        & (stage_blade_position[:, 0] <= TRANSIT_CLEAR_BLADE_CENTRE_X + 0.015)
                    )
                    attitude_stage_ready = (
                        sensor_ready
                        & self.relocation_aligned[ids]
                        & self.relocation_stage_lateral_done[ids]
                        & ~self.relocation_stage_attitude_done[ids]
                        & (
                            (stage_blade_position[:, 1:] - self.relocation_staging_pos[1:].unsqueeze(0))
                            .abs()
                            .amax(dim=-1)
                            <= 0.0005
                        )
                        & (stage_blade_orientation_error <= 0.002)
                        & (torch.linalg.vector_norm(stage_velocity[:, :3], dim=-1) <= 0.060)
                        & (
                            torch.linalg.vector_norm(stage_velocity[:, 3:], dim=-1)
                            <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS
                        )
                    )
                    full_stage_ready = (
                        sensor_ready
                        & self.relocation_aligned[ids]
                        & self.relocation_stage_attitude_done[ids]
                        & ~self.relocation_stage_translated[ids]
                        & (stage_blade_position_error <= INSERT_HANDOFF_POSITION_TOLERANCE_M)
                        & (stage_blade_orientation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD)
                        & (
                            torch.linalg.vector_norm(stage_velocity[:, :3], dim=-1)
                            <= INSERTION_LINEAR_VELOCITY_LIMIT_MPS
                        )
                        & (
                            torch.linalg.vector_norm(stage_velocity[:, 3:], dim=-1)
                            <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS
                        )
                    )
                    stage_ready = torch.where(
                        ~self.relocation_stage_retreat_done[ids],
                        retreat_stage_ready,
                        torch.where(
                            ~self.relocation_stage_lateral_done[ids],
                            lateral_stage_ready,
                            torch.where(
                                self.relocation_stage_attitude_done[ids],
                                full_stage_ready,
                                attitude_stage_ready,
                            ),
                        ),
                    )
                    final_stage_ready = (
                        self.relocation_aligned[ids]
                        & self.relocation_stage_translated[ids]
                        & (stage_blade_position_error <= INSERT_HANDOFF_POSITION_TOLERANCE_M)
                        & (
                            torch.linalg.vector_norm(stage_velocity[:, :3], dim=-1)
                            <= INSERTION_LINEAR_VELOCITY_LIMIT_MPS
                        )
                        & (
                            torch.linalg.vector_norm(stage_velocity[:, 3:], dim=-1)
                            <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS
                        )
                    )
                    stage_ready = torch.where(self.relocation_stage_translated[ids], final_stage_ready, stage_ready)
                    reached[ids] = torch.where(stage_leg, stage_ready, reached[ids])
                due = transiting & reached
                retreat_stage_complete = (
                    due
                    & (self.waypoint_read == 1)
                    & self.base_rail_enabled
                    & self.relocation_aligned
                    & ~self.relocation_stage_retreat_done
                )
                lateral_stage_complete = (
                    due
                    & (self.waypoint_read == 1)
                    & self.base_rail_enabled
                    & self.relocation_aligned
                    & self.relocation_stage_retreat_done
                    & ~self.relocation_stage_lateral_done
                    & ~self.relocation_stage_translated
                )
                stage_translation_complete = (
                    due
                    & (self.waypoint_read == 1)
                    & self.base_rail_enabled
                    & self.relocation_aligned
                    & self.relocation_stage_lateral_done
                    & self.relocation_stage_attitude_done
                    & ~self.relocation_stage_translated
                )
                attitude_stage_complete = (
                    due
                    & (self.waypoint_read == 1)
                    & self.base_rail_enabled
                    & self.relocation_aligned
                    & self.relocation_stage_lateral_done
                    & ~self.relocation_stage_attitude_done
                )
                if bool(retreat_stage_complete.any()):
                    self.relocation_stage_retreat_done[retreat_stage_complete] = True
                    due = due & ~retreat_stage_complete
                if bool(lateral_stage_complete.any()):
                    self.relocation_stage_lateral_done[lateral_stage_complete] = True
                    due = due & ~lateral_stage_complete
                if bool(attitude_stage_complete.any()):
                    self.relocation_stage_attitude_done[attitude_stage_complete] = True
                    due = due & ~attitude_stage_complete
                if bool(stage_translation_complete.any()):
                    self.relocation_stage_translated[stage_translation_complete] = True
                    # The six-DOF stage has already closed position, attitude,
                    # and velocity at this point.  Keep the arm in its measured
                    # collision-clear joint pose and consume the cross-bay
                    # waypoint immediately; handing attitude back to arm IK here
                    # reintroduced the very coupled-positioning failure the
                    # physical stage removes.
                    self.gripper.retain_latch[stage_translation_complete] = False
                start_alignment = (
                    due
                    & (self.waypoint_read == 1)
                    & ~self.relocation_aligning
                    & ~self.relocation_aligned
                    & ~self.relocation_joint_replaying
                )
                if bool(start_alignment.any()):
                    start_ids = torch.nonzero(start_alignment, as_tuple=False).squeeze(-1)
                    if self.base_rail_enabled:
                        self.relocation_aligned[start_alignment] = True
                    else:
                        blade_position = task.scene["spare_blade"].data.root_pos_w[start_ids]
                        self.relocation_alignment_tool_pos[start_ids] = blade_position - quat_apply(
                            self.relocation_desired_tool_rot[start_ids],
                            self.relocation_blade_relative_to_tool[start_ids],
                        )
                        self.relocation_aligning[start_alignment] = True
                    self.gripper.retain_latch[start_alignment] = False
                # Alignment is a substage at the clear cross-bay waypoint, not
                # permission to consume that waypoint.  Only the subsequent
                # converged step advances to the axial leg.
                due = due & ~self.relocation_aligning & ~self.relocation_joint_replaying
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
            if self.workflow == "relocate":
                aligning_ids = self.relocation_aligning[ids]
                aligned_final_ids = self.relocation_aligned[ids] & (self.waypoint_read[ids] <= 0)
                target = torch.where(
                    aligning_ids.unsqueeze(-1),
                    self.relocation_alignment_tool_pos[ids],
                    torch.where(
                        aligned_final_ids.unsqueeze(-1),
                        self.relocation_final_tool_aligned[ids],
                        target,
                    ),
                )
            scale = self.scales[TRANSIT]
            position_error = target - tool[ids]
            if self.workflow == "relocate" and self.base_rail_enabled:
                root_quat = task.scene["robot"].data.root_quat_w[ids]
                position_error = quat_apply(quat_inv(root_quat), position_error)
            position_action = (position_error / scale[:3]).clamp(-1.0, 1.0)
            if self.workflow == "relocate":
                stage_positioning = (
                    (self.waypoint_read[ids] == 1)
                    & (
                        (~self.relocation_aligning[ids] & self.relocation_aligned[ids])
                        | (self.relocation_aligning[ids] & self.relocation_stage_translated[ids])
                    )
                    & ~self.relocation_joint_replaying[ids]
                    & self.base_rail_enabled
                )
                estimator = getattr(task, "_module_state_estimator", None)
                if estimator is not None and estimator.backend == "fiducial_pnp":
                    # A missed shuttle-frame detection freezes drive targets;
                    # force drives may settle, but no new motion is authorized
                    # from a held pose estimate.
                    stage_positioning &= estimator.fiducial_current_detection[ids]
                if bool(stage_positioning.any()):
                    stage_ids = ids[stage_positioning]
                    stage_error = self.stage_goal_target_m[stage_ids] - self.stage_drive_target_m[stage_ids]
                    # Close the lateral loop on the carried module rather than
                    # assuming a target-to-pose identity through a compliant
                    # drive.  The former open-loop target stopped 28--71 mm
                    # short under arm load despite reaching its commanded D6
                    # value.  This remains a force-driven physical stage: only
                    # the drive target is changed, never robot/payload state.
                    payload_position, payload_orientation, _ = self._payload_feedback()
                    blade_position = payload_position[stage_ids]
                    blade_error = self.relocation_staging_pos.unsqueeze(0) - blade_position
                    blade_error_stage = blade_error
                    self.payload_stage_control_steps[stage_ids] += 1
                    self.payload_stage_last_error_world[stage_ids] = blade_error
                    self.payload_stage_last_error_stage[stage_ids] = blade_error_stage
                    owns_lateral = self.relocation_stage_retreat_done[stage_ids]
                    lateral_goal = blade_position.clone()
                    lateral_goal[:, 0] = TRANSIT_CLEAR_BLADE_CENTRE_X
                    lateral_goal[:, 1] = self.relocation_staging_pos[1]
                    lateral_goal[:, 2] = self.payload_stage_capture_pos[stage_ids, 2]
                    lateral_error_stage = lateral_goal - blade_position
                    stage_error[owns_lateral] = BASE_STAGE_OUTER_LOOP_GAIN * lateral_error_stage[owns_lateral]
                    owns_retreat = ~owns_lateral
                    retreat_error_world = blade_position.new_zeros(blade_position.shape)
                    retreat_error_world[:, 0] = TRANSIT_CLEAR_BLADE_CENTRE_X - blade_position[:, 0]
                    retreat_error_stage = retreat_error_world
                    stage_error[owns_retreat, 0] = BASE_STAGE_OUTER_LOOP_GAIN * retreat_error_stage[owns_retreat, 0]
                    owns_rack_alignment = self.relocation_stage_lateral_done[stage_ids]
                    alignment_goal = self.relocation_staging_pos.unsqueeze(0).expand_as(blade_position).clone()
                    alignment_goal[:, 0] = TRANSIT_CLEAR_BLADE_CENTRE_X
                    alignment_error_stage = alignment_goal - blade_position
                    stage_error[owns_rack_alignment] = (
                        BASE_STAGE_OUTER_LOOP_GAIN * alignment_error_stage[owns_rack_alignment]
                    )
                    owns_axial = self.relocation_stage_attitude_done[stage_ids]
                    orientation_error_now = torch.linalg.vector_norm(
                        axis_angle_from_quat(
                            quat_mul(
                                self.relocation_staging_rot.unsqueeze(0).expand_as(payload_orientation[stage_ids]),
                                quat_inv(payload_orientation[stage_ids]),
                            )
                        ),
                        dim=-1,
                    )
                    lateral_guard_tolerance = 0.001
                    orientation_guard_tolerance = 0.003
                    estimator = getattr(task, "_module_state_estimator", None)
                    if estimator is not None and estimator.backend == "fiducial_pnp":
                        lateral_guard_tolerance = FIDUCIAL_GUARDED_LATERAL_TOLERANCE_M
                        orientation_guard_tolerance = FIDUCIAL_GUARDED_ORIENTATION_TOLERANCE_RAD
                    axial_alignment_ok = (
                        (blade_position[:, 1:] - self.relocation_staging_pos[1:].unsqueeze(0)).abs().amax(dim=-1)
                        <= lateral_guard_tolerance
                    ) & (orientation_error_now <= orientation_guard_tolerance)
                    guarded_axial = owns_axial & axial_alignment_ok
                    guarded_retract = owns_axial & ~axial_alignment_ok
                    stage_error[guarded_axial] = BASE_STAGE_OUTER_LOOP_GAIN * blade_error_stage[guarded_axial]
                    stage_error[guarded_retract] = BASE_STAGE_OUTER_LOOP_GAIN * alignment_error_stage[guarded_retract]
                    target_delta = stage_error.clamp(-BASE_RAIL_TARGET_STEP_M, BASE_RAIL_TARGET_STEP_M)
                    target_delta[owns_axial] = target_delta[owns_axial].clamp(
                        -BASE_STAGE_GUARDED_AXIAL_STEP_M,
                        BASE_STAGE_GUARDED_AXIAL_STEP_M,
                    )
                    previous_target = self.stage_drive_target_m[stage_ids].clone()
                    lower = self.stage_drive_target_m.new_tensor(BASE_STAGE_MIN_TARGET_M)
                    upper = self.stage_drive_target_m.new_tensor(BASE_STAGE_MAX_TARGET_M)
                    self.stage_drive_target_m[stage_ids] = torch.maximum(
                        torch.minimum(previous_target + target_delta, upper), lower
                    )
                    actual_travel_stage = blade_position - self.payload_stage_capture_pos[stage_ids]
                    if bool(owns_axial.any()):
                        axial_targets = self.stage_drive_target_m[stage_ids][owns_axial]
                        axial_actual = actual_travel_stage[owns_axial]
                        self.stage_drive_target_m[stage_ids[owns_axial]] = torch.maximum(
                            torch.minimum(
                                axial_targets,
                                axial_actual + BASE_STAGE_MAX_TRANSLATION_LEAD_M,
                            ),
                            axial_actual - BASE_STAGE_MAX_TRANSLATION_LEAD_M,
                        )
                    self.stage_goal_target_m[stage_ids] = self.stage_drive_target_m[stage_ids]
                    for stage_id in stage_ids.detach().cpu().tolist():
                        for axis, attribute in enumerate(self.stage_drive_target_attributes[stage_id]):
                            attribute.Set(float(self.stage_drive_target_m[stage_id, axis]))
                    if bool(owns_rack_alignment.any()):
                        pose_ids = stage_ids[owns_rack_alignment]
                        blade_orientation = payload_orientation[pose_ids]
                        rotation_error_world_rad = axis_angle_from_quat(
                            quat_mul(
                                self.relocation_staging_rot.unsqueeze(0).expand_as(blade_orientation),
                                quat_inv(blade_orientation),
                            )
                        )
                        rotation_error_stage_rad = rotation_error_world_rad
                        # This D6 joint's body 1 is the payload itself.  Its
                        # positive target therefore follows the world-frame
                        # correction because the fixed anchor is unrotated.
                        rotation_delta_deg = torch.rad2deg(BASE_STAGE_ROTATION_GAIN * rotation_error_stage_rad).clamp(
                            -BASE_STAGE_ROTATION_STEP_DEG, BASE_STAGE_ROTATION_STEP_DEG
                        )
                        previous_rotation_target = self.stage_rotation_drive_target_deg[pose_ids].clone()
                        self.stage_rotation_drive_target_deg[pose_ids] = (
                            previous_rotation_target + rotation_delta_deg
                        ).clamp(-BASE_STAGE_ROTATION_LIMIT_DEG, BASE_STAGE_ROTATION_LIMIT_DEG)
                        for pose_id in pose_ids.detach().cpu().tolist():
                            for axis, attribute in enumerate(self.stage_rotation_drive_target_attributes[pose_id]):
                                attribute.Set(float(self.stage_rotation_drive_target_deg[pose_id, axis]))
                    position_action[stage_positioning] = 0.0
                    target_changed = (self.stage_drive_target_m[stage_ids] - previous_target).abs().amax(dim=-1) > 1e-9
                    self.rail_commanded_steps[stage_ids] += target_changed.to(torch.long)
                final_translation = (
                    (self.waypoint_read[ids] <= 0) & self.relocation_aligned[ids] & ~self.relocation_aligning[ids]
                )
                position_authority = torch.where(
                    final_translation,
                    torch.full_like(final_translation, RELOCATE_FINAL_LEG_POSITION_AUTHORITY, dtype=torch.float32),
                    torch.ones_like(final_translation, dtype=torch.float32),
                )
                position_action = position_action * position_authority.unsqueeze(-1)
            self.actions[ids, :3] = position_action
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
                # Translate with the attitude held at extraction clear. Mixing
                # the 330 mm final leg and rack alignment in one differential-
                # IK command made the solver spend its authority rotating: the
                # arm stalled while the bounded latch saturated and lost its
                # transform. Once the module centre reaches the staging point,
                # rotate about that point while the position target compensates
                # for the captured tool-to-module offset.
                command_rack_attitude = self.relocation_aligning[ids] | self.relocation_aligned[ids]
                if self.base_rail_enabled and not self.base_rail_joint_hold:
                    command_rack_attitude = command_rack_attitude | (self.waypoint_read[ids] == 1)
                desired_tool_rot = torch.where(
                    command_rack_attitude.unsqueeze(-1),
                    self.relocation_desired_tool_rot[ids],
                    self.relocation_hold_tool_rot[ids],
                )
                if self.base_rail_enabled:
                    root_quat = task.scene["robot"].data.root_quat_w[ids]
                    inverse_root = quat_inv(root_quat)
                    current_tool_rot = quat_mul(inverse_root, tool_rot[ids])
                    desired_tool_rot = quat_mul(inverse_root, desired_tool_rot)
                else:
                    current_tool_rot = tool_rot[ids]
                rotation_error = axis_angle_from_quat(quat_mul(desired_tool_rot, quat_inv(current_tool_rot)))
                attitude_authority = torch.where(
                    self.relocation_aligning[ids],
                    torch.full_like(
                        self.relocation_aligning[ids],
                        TRANSIT_ALIGN_ATTITUDE_AUTHORITY,
                        dtype=torch.float32,
                    ),
                    torch.full_like(
                        self.relocation_aligning[ids],
                        TRANSIT_HOLD_ATTITUDE_AUTHORITY,
                        dtype=torch.float32,
                    ),
                )
                # Crossing the rack midplane with a fixed head-on attitude
                # drives this six-axis arm through a measured DLS singularity.
                # The payload is now form-locked to the wrist, so let the wrist
                # choose its orientation during the collision-clear lateral
                # translation, then restore rack attitude at the stationary
                # destination-clear waypoint below. Before the fixed latch this
                # was invalid because the passive pin let the module tumble
                # independently of the tool; that failure mode no longer exists.
                crossing_clear = (
                    (self.waypoint_read[ids] == 1)
                    & ~self.relocation_aligning[ids]
                    & ~self.relocation_aligned[ids]
                    & (getattr(task.cfg, "latch_joint_mode", "compliant") == "fixed")
                    & (not self.base_rail_enabled)
                )
                attitude_authority = torch.where(
                    crossing_clear,
                    torch.zeros_like(attitude_authority),
                    attitude_authority,
                )
                raw_rotation_action = rotation_error / scale[3:6]
                self.actions[ids, 3:6] = torch.maximum(
                    torch.minimum(raw_rotation_action, attitude_authority.unsqueeze(-1)),
                    -attitude_authority.unsqueeze(-1),
                )
            secured_for_handoff = torch.where(
                self.payload_stage_engaged,
                self.payload_stage_engaged,
                established,
            )
            arrived = transiting & (self.waypoint_read <= 0) & secured_for_handoff
            if self.workflow == "relocate":
                # The receiving checkpoint starts at one exact full-distance
                # rack pose.  Crossing the bay midpoint and reaching roughly
                # the right x-depth previously admitted hand-offs 164 mm high,
                # 22 mm off-centre, rotating at 0.50 rad/s, and 1.49 rad out of
                # attitude.  That was not a difficult insertion; it was an
                # invalid policy precondition.  Reuse the insertion success
                # envelope as a fail-closed staging contract.
                blade_local, blade_orientation, blade_velocity = self._payload_feedback()
                staging_position = self.relocation_staging_pos.unsqueeze(0)
                staging_orientation = self.relocation_staging_rot.unsqueeze(0).expand_as(blade_orientation)
                staging_position_error = torch.linalg.vector_norm(blade_local - staging_position, dim=-1)
                staging_orientation_error = torch.linalg.vector_norm(
                    axis_angle_from_quat(quat_mul(staging_orientation, quat_inv(blade_orientation))),
                    dim=-1,
                )
                staging_linear_speed = torch.linalg.vector_norm(blade_velocity[:, :3], dim=-1)
                staging_angular_speed = torch.linalg.vector_norm(blade_velocity[:, 3:], dim=-1)
                secured = secured_for_handoff
                if self.release_latch_required and not self.base_rail_enabled:
                    secured = secured & grapple_latched(task)
                arrived = (
                    arrived
                    & secured
                    & (staging_position_error <= INSERT_HANDOFF_POSITION_TOLERANCE_M)
                    & (staging_orientation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD)
                    & (staging_linear_speed <= INSERTION_LINEAR_VELOCITY_LIMIT_MPS)
                    & (staging_angular_speed <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS)
                )
            # Insertion drives the module back into its rails, so the grip has to
            # carry contact again.  ``established`` is also a hard precondition:
            # a latch probe once advanced merely because a flipped module's
            # centre crossed the axial/lateral planes while its grip point was
            # 831 mm from the tool.  Geometry alone is not a valid hand-off.
            # Retaining through insertion is the failure the capture/hold split
            # exists to prevent.
            self.gripper.retain_latch[arrived] = False
            if self.base_rail_enabled and bool(arrived.any()):
                self._set_stage_arm_servo(
                    torch.nonzero(arrived, as_tuple=False).squeeze(-1),
                    strengthened=False,
                )
            self.phase[arrived] = INSERT

        # --- insert -> seated --------------------------------------------------
        inserting = self.phase == INSERT
        if self.rigid_transit and bool(inserting.any()):
            fired = self._step_guarded_insert(inserting, step, tool, tool_rot)
            self._finish(fired, step)
            self.predicate_fired[fired] = True
            if SEATED_RETAIN:
                self.gripper.retain_latch[fired] = True
        elif bool(inserting.any()):
            shuttle_inserting = inserting & self.payload_stage_engaged & self.base_rail_enabled
            estimator = getattr(task, "_module_state_estimator", None)
            if estimator is not None and estimator.backend == "fiducial_pnp":
                shuttle_inserting &= estimator.fiducial_current_detection
            if bool(shuttle_inserting.any()):
                shuttle_ids = torch.nonzero(shuttle_inserting, as_tuple=False).squeeze(-1)
                payload_position, payload_orientation, payload_velocity = self._payload_feedback()
                blade_local = payload_position[shuttle_ids]
                inserted_target = blade_local.new_tensor(SECOND_SLOT_INSERTED_POS).unsqueeze(0)
                blade_error_world = inserted_target - blade_local
                blade_error_stage = blade_error_world
                target_delta = (BASE_STAGE_OUTER_LOOP_GAIN * blade_error_stage).clamp(
                    -BASE_STAGE_GUARDED_AXIAL_STEP_M,
                    BASE_STAGE_GUARDED_AXIAL_STEP_M,
                )
                previous_target = self.stage_drive_target_m[shuttle_ids].clone()
                lower = self.stage_drive_target_m.new_tensor(BASE_STAGE_MIN_TARGET_M)
                upper = self.stage_drive_target_m.new_tensor(BASE_STAGE_MAX_TARGET_M)
                self.stage_drive_target_m[shuttle_ids] = torch.maximum(
                    torch.minimum(previous_target + target_delta, upper),
                    lower,
                )
                # A force drive may trail its command while entering contact.
                # Bound that lead exactly as in the staging leg so a temporary
                # stop cannot accumulate a large hidden target and release it
                # as an impulse once the contact clears.
                actual_travel_stage = blade_local - self.payload_stage_capture_pos[shuttle_ids]
                self.stage_drive_target_m[shuttle_ids] = torch.maximum(
                    torch.minimum(
                        self.stage_drive_target_m[shuttle_ids],
                        actual_travel_stage + BASE_STAGE_MAX_TRANSLATION_LEAD_M,
                    ),
                    actual_travel_stage - BASE_STAGE_MAX_TRANSLATION_LEAD_M,
                )
                self.stage_goal_target_m[shuttle_ids] = self.stage_drive_target_m[shuttle_ids]
                rotation_error_world = axis_angle_from_quat(
                    quat_mul(
                        self.relocation_staging_rot.unsqueeze(0).expand_as(payload_orientation[shuttle_ids]),
                        quat_inv(payload_orientation[shuttle_ids]),
                    )
                )
                rotation_error_stage = rotation_error_world
                rotation_delta_deg = torch.rad2deg(BASE_STAGE_ROTATION_GAIN * rotation_error_stage).clamp(
                    -BASE_STAGE_ROTATION_STEP_DEG, BASE_STAGE_ROTATION_STEP_DEG
                )
                self.stage_rotation_drive_target_deg[shuttle_ids] = (
                    self.stage_rotation_drive_target_deg[shuttle_ids] + rotation_delta_deg
                ).clamp(-BASE_STAGE_ROTATION_LIMIT_DEG, BASE_STAGE_ROTATION_LIMIT_DEG)
                for shuttle_id in shuttle_ids.detach().cpu().tolist():
                    for axis, attribute in enumerate(self.stage_drive_target_attributes[shuttle_id]):
                        attribute.Set(float(self.stage_drive_target_m[shuttle_id, axis]))
                    for axis, attribute in enumerate(self.stage_rotation_drive_target_attributes[shuttle_id]):
                        attribute.Set(float(self.stage_rotation_drive_target_deg[shuttle_id, axis]))
                self.rail_commanded_steps[shuttle_ids] += (
                    (self.stage_drive_target_m[shuttle_ids] - previous_target).abs().amax(dim=-1) > 1e-9
                ).to(torch.long)

                target = payload_position.new_tensor(SECOND_SLOT_INSERTED_POS)
                axial = (payload_position[:, 0] - target[0]).abs()
                lateral = torch.linalg.vector_norm(payload_position[:, 1:3] - target[1:3], dim=-1)
                orientation = torch.linalg.vector_norm(
                    axis_angle_from_quat(
                        quat_mul(
                            self.relocation_staging_rot.unsqueeze(0).expand_as(payload_orientation),
                            quat_inv(payload_orientation),
                        )
                    ),
                    dim=-1,
                )
                seated_now = (
                    shuttle_inserting
                    & (axial <= INSERTION_AXIAL_DEPTH_TOLERANCE_M)
                    & (lateral <= INSERTION_LATERAL_TOLERANCE_M)
                    & (orientation <= INSERTION_ORIENTATION_TOLERANCE_RAD)
                    & (torch.linalg.vector_norm(payload_velocity[:, :3], dim=-1) <= INSERTION_LINEAR_VELOCITY_LIMIT_MPS)
                    & (
                        torch.linalg.vector_norm(payload_velocity[:, 3:], dim=-1)
                        <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS
                    )
                )
                self.payload_stage_insert_hold = torch.where(
                    seated_now,
                    self.payload_stage_insert_hold + 1,
                    torch.zeros_like(self.payload_stage_insert_hold),
                )
                required_insert_hold = max(1, int(round(0.20 / float(task.step_dt))))
                shuttle_fired = shuttle_inserting & (self.payload_stage_insert_hold >= required_insert_hold)
            else:
                shuttle_fired = torch.zeros_like(inserting)
            # The learned arm path keeps its own predicate.  The shuttle path
            # uses the identical seated geometry/motion envelope plus a 0.20 s
            # hold, with the D6 joint replacing the no-longer-relevant grip
            # conditions after physical handoff.
            learned_fired = inserting & ~self.payload_stage_engaged & grapple_insertion_success_mask(task)
            fired = shuttle_fired | learned_fired
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
            # A completed manipulation retains the module through the settling
            # re-check. A planning rejection never began manipulation, so it
            # must keep the gripper open instead of turning DONE into a hidden
            # close command.
            self.actions[finished & ~plan_blocked, 6] = 1.0
            self.actions[finished & plan_blocked, 6] = 0.0
            # The shuttle, not the gripper, owns the delivered module. Keep the
            # arm visibly open after handoff and during the settle proof.
            self.actions[finished & self.payload_stage_engaged, 6] = -1.0
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
                # **This is the release the acceptance rule is about.** Every
                # condition has now been re-checked after the settling window,
                # so the hand may open -- and only now. Recorded, because an
                # operation that ends with the robot still holding the module
                # has not finished, and one that lets go early has not
                # succeeded.
                verified = ripe & self.outcome & self.all_conditions
                self.gripper_released |= verified
                self.gripper_released_at[verified] = step
            # Held after the judgement as well, so the open persists for the
            # rest of the recording rather than for one frame.
            self.actions[finished & self.gripper_released, 6] = -1.0

        if self.base_rail_enabled:
            rail_joint_hold = (
                (self.phase == TRANSIT)
                & (self.waypoint_read == 1)
                & ~self.relocation_aligning
                & self.relocation_aligned
                & ~self.relocation_joint_replaying
            )
            self.arm.set_joint_hold_mask(rail_joint_hold)
            replay_ids = torch.nonzero(self.relocation_joint_replaying, as_tuple=False).squeeze(-1)
            if replay_ids.numel() > 0:
                replay_targets = self.extraction_joint_waypoints[
                    self.relocation_joint_replay_index[replay_ids], replay_ids
                ]
                self.arm.set_joint_target_override(replay_ids, replay_targets)

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

    def _rigid_tool_command(
        self,
        ids: torch.Tensor,
        module_pos: torch.Tensor,
        module_rot: torch.Tensor,
        tool_rot_now: torch.Tensor,
        measured_module_pos: torch.Tensor | None = None,
    ):
        """Turn a desired *module* pose into the tool pose that produces it.

        The form lock fixes ``blade = tool * offset``, so this is one inversion
        rather than a control problem. Everything the rigid transit and the
        guarded insertion command goes through here, which is why neither of
        them needs a sub-phase.

        **The position is inverted through the tool's** ``tool_rot_now``, **not
        through the attitude being asked for**, and that is the difference
        between a controller that converges and one that does not. The module
        hangs about 340 mm in front of the tool, so a tool that is 0.16 rad from
        the attitude it was told to hold puts the module 54 mm from where it was
        told to go -- and if the position target is computed from the *desired*
        attitude, that 54 mm never closes while the attitude is short. Measured
        exactly there twice: the crossing leg parked with its lateral error at
        61 mm and then at 64 mm, both times equal to the standing attitude error
        times the offset. Inverting through the attitude the tool actually has
        makes the position converge on its own timescale and leaves the attitude
        to converge on its own, which is what the two legs after it are for.
        """

        tool_rot = quat_mul(module_rot, quat_inv(self.relocation_blade_relative_rot_to_tool[ids]))
        tool_pos = module_pos - quat_apply(tool_rot_now, self.relocation_blade_relative_to_tool[ids])
        if measured_module_pos is not None and bool(grapple_latch_rigid(self.task)[ids].logical_not().any()):
            # **Close the loop on the module, not on the tool.**
            #
            # The line above is feed-forward: it assumes the module sits at the
            # offset recorded when the lock engaged. While the lock is a weld
            # that is exact. While it is a spring it is not -- the module lags
            # the tool by force over stiffness, and the rack pushes back hard
            # enough to matter. Measured, the seating parked 3.5 mm short of its
            # target and stayed there for six hundred steps: a steady-state
            # error, which is what feed-forward alone always leaves.
            #
            # Adding the module's own error moves the tool that much further,
            # which loads the spring until the module arrives. Bounded by the
            # compliance's stroke, because past its stop the extra command is
            # not compliance any more, it is a shove.
            # Only where the lock has actually softened. While it is a weld
            # the module is exactly where the feed-forward puts it, and adding
            # its tracking error there just doubles the position gain on a
            # rigid payload -- measured, that walked the carried module 208 mm
            # back toward the bay it came from.
            compliant = grapple_latch_rigid(self.task)[ids].logical_not().unsqueeze(-1)
            tool_pos = tool_pos + compliant * (module_pos - measured_module_pos).clamp(
                -MATING_TRIM_LIMIT_M, MATING_TRIM_LIMIT_M
            )
        return tool_pos, tool_rot

    def _drive_tool_to(
        self,
        ids: torch.Tensor,
        tool: torch.Tensor,
        tool_rot: torch.Tensor,
        target_pos: torch.Tensor,
        target_rot: torch.Tensor,
        scale: torch.Tensor,
        attitude_authority: float,
    ) -> None:
        """Command one bounded Cartesian correction toward a tool pose."""

        self.actions[ids, :3] = ((target_pos - tool[ids]) / scale[:3]).clamp(-1.0, 1.0)
        rotation_error = axis_angle_from_quat(quat_mul(target_rot, quat_inv(tool_rot[ids])))
        self.actions[ids, 3:6] = (rotation_error / scale[3:6]).clamp(-attitude_authority, attitude_authority)

    def _step_rigid_transit(self, transiting: torch.Tensor, step: int, tool: torch.Tensor, tool_rot: torch.Tensor) -> torch.Tensor:
        """Fly the carried module through three waypoints, in module space.

        Leg 2 retreats along the rack axis until the module's *measured* front
        corner is behind the lead-in flares. Leg 1 crosses to the second bay at
        that depth and squares the module to the rack while it goes, which is
        free there because squaring only shortens the overhang that leg 2 just
        made room for. Leg 0 drives it in to the pose the insertion begins from.
        Each leg is finished when the *module* has covered the distance the leg
        was laid out to cover -- not when the tool has, because the clearance
        the leg exists for is a module clearance.
        """

        task = self.task
        ids = torch.nonzero(transiting, as_tuple=False).squeeze(-1)
        # **The holding closure stays on through the carry, and that is
        # measured rather than assumed.**
        #
        # The obvious reading is that the pads have nothing left to do once the
        # form lock is engaged, and that their wedge thrust -- hundreds of
        # newtons, from a drive saturating at 10 N-m through a 106.2 mm/rad
        # transmission -- is then only a preload the arm has to fight with a
        # 216 N-m/rad wrist. Relaxing to the retain closure to test that made
        # the carried attitude *worse*, not better: 0.328 rad off square at the
        # final leg against 0.066 rad with the holding closure kept, on the same
        # seed and the same policies. Whatever the standing attitude error is,
        # it is not the wedge preload, and the pads are still doing something
        # the lock does not replace.
        blade = task.scene["spare_blade"]
        module_pos = blade.data.root_pos_w[ids] - task.scene.env_origins[ids]
        module_rot = blade.data.root_quat_w[ids]
        leg = self.waypoint_read[ids]
        target_pos_local = self.module_leg_pos[leg, ids]
        target_rot = self.module_leg_rot[leg, ids]
        position_error = target_pos_local - module_pos
        orientation_error = torch.linalg.vector_norm(
            axis_angle_from_quat(quat_mul(target_rot, quat_inv(module_rot))), dim=-1
        )
        # **Full authority on attitude, on every leg, and that is the opposite
        # of what the pad-held follower does.**
        #
        # The legacy path bounds rotation to a quarter of the command because a
        # module held in the pads tumbles slowly and a differential IK solving
        # one 6-D command spends its authority turning the wrist instead of
        # crossing the rack. Under the form lock the trade reverses: the module
        # cannot tumble, and giving the attitude away is not a cosmetic cost --
        # the module *is* the wrist, so a wrist that winds carries the module
        # round with it. Measured at a quarter authority, the crossing leg wound
        # the module from 0.12 rad to 2.85 rad off square while tracking its
        # lateral target, which is the reach-boundary trade
        # ``evidence/attitude_wall_lateral_profile.json`` describes, taken in the
        # direction a carried payload cannot survive.
        #
        # At full authority the solver has to hold the attitude and surrenders
        # position instead, which stalls a leg rather than losing the module --
        # a failure that is visible in the report and recoverable, against one
        # that is neither.
        authority = RIGID_TRANSIT_ATTITUDE_AUTHORITY
        target_tool_pos, target_tool_rot = self._rigid_tool_command(
            ids,
            target_pos_local + task.scene.env_origins[ids],
            target_rot,
            tool_rot[ids],
            blade.data.root_pos_w[ids],
        )
        self._drive_tool_to(ids, tool, tool_rot, target_tool_pos, target_tool_rot, self.scales[TRANSIT], authority)

        squared = (orientation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD) & (
            torch.linalg.vector_norm(position_error, dim=-1) <= INSERT_HANDOFF_POSITION_TOLERANCE_M
        )
        retreat_done = (leg == 4) & (position_error[:, 0].abs() <= 0.005)
        cross_done = (leg == 2) & (position_error[:, 1:].abs().amax(dim=-1) <= 0.005)
        square_done = ((leg == 3) | (leg == 1)) & squared
        met = retreat_done | cross_done | square_done
        # A leg that has converged as far as this arm's own branch allows is
        # finished whether or not it met its gate, and the residual it stopped
        # at is the measurement. Recorded per leg, per environment.
        stalled = (~met) & (
            (step - self.transit_leg_entered[ids]) >= RIGID_TRANSIT_LEG_TIMEOUT_STEPS
        )
        if bool(stalled.any()):
            stalled_ids = ids[stalled]
            self.transit_leg_forced[leg[stalled], stalled_ids] = True
            self.transit_leg_residual_rad[leg[stalled], stalled_ids] = orientation_error[stalled]
            self.transit_leg_residual_m[leg[stalled], stalled_ids] = torch.linalg.vector_norm(
                position_error[stalled], dim=-1
            )
        advance = torch.zeros_like(transiting)
        advance[ids] = met | stalled
        # **Rigid becomes compliant here, at the end of the squaring leg, and
        # the reason is that this is where the rack takes over.**
        #
        # The last leg is not a transit. It drives the module 450 mm along the
        # rack axis, and its nose reaches the lead-in about 10 mm in -- so the
        # mating starts here, not at the phase boundary two legs later. Softened
        # at the phase boundary instead, the module has to enter the channel
        # rigid first, which is the thing that does not work: measured, it jams
        # on the mouth and the arm pushes against it until the clock runs out.
        #
        # Everything before this point wants the weld: 2.3 mm of drift across a
        # 450 mm flight against 808 mm on the pads alone. Everything after wants
        # the spring, because a lead-in aligns a part by pushing it.
        mating = torch.zeros_like(transiting)
        mating[ids] = advance[ids] & (leg == 1)
        if MATING_MODE == "rigid":
            mating = torch.zeros_like(mating)
        if bool(mating.any()):
            soften_grapple_latch(task, mating)
            self.latch_softened |= mating
            self.latch_softened_at[mating & (self.latch_softened_at < 0)] = step
        self.waypoint_read[advance] = (self.waypoint_read[advance] - 1).clamp_min(0)
        self.transit_leg_entered[advance] = step

        # **The hand-off contract is the receiving controller's precondition,
        # and this chain's receiver is not the learned policy.**
        #
        # 2.5 mm on the full three-dimensional staging pose is the *insert
        # skill's* reset distribution: that policy starts from one exact rack
        # pose and a weaker condition cannot authorise handing to it, which is
        # why the constant exists and why it stays the contract whenever the
        # policy is the receiver. This chain hands to the guarded controller in
        # ``_step_guarded_insert``, which starts from wherever the module is and
        # advances only while the estimator says it is inside the bay's own
        # catch. Its precondition is therefore that envelope, stated once here
        # rather than inherited from a policy that is not running.
        #
        # Holding the tighter number anyway is not conservatism, it is a
        # different failure: measured, the module reached 0.5804 m against a
        # 0.5829 m target with 0.6 mm of lateral error -- a 2.57 mm norm against
        # a 2.5 mm gate -- and sat there while the clock ran out, because
        # nothing downstream needed the missing 0.07 mm.
        velocity = attached_blade_velocity(task)[ids]
        axial_error = position_error[:, 0].abs()
        lateral_error = torch.linalg.vector_norm(position_error[:, 1:], dim=-1)
        if MATING_MODE == "compliant":
            pose_ready = (axial_error <= INSERTION_AXIAL_DEPTH_TOLERANCE_M) & (
                lateral_error <= SLOT_ENTRY_RAMP_CATCH_M
            )
        else:
            pose_ready = (
                torch.linalg.vector_norm(position_error, dim=-1) <= INSERT_HANDOFF_POSITION_TOLERANCE_M
            )
        arrived = torch.zeros_like(transiting)
        arrived[ids] = (
            (leg <= 0)
            & pose_ready
            & (orientation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD)
            & (torch.linalg.vector_norm(velocity[:, :3], dim=-1) <= INSERTION_LINEAR_VELOCITY_LIMIT_MPS)
            & (torch.linalg.vector_norm(velocity[:, 3:], dim=-1) <= INSERTION_ANGULAR_VELOCITY_LIMIT_RADPS)
        )
        return arrived & grapple_latched(task)  # softened counts: it is still the load path

    def _begin_guarded_insert(self, mask: torch.Tensor, _step: int) -> None:
        """Let the form lock go, and set the axial target the seating will use.

        **The lock is released here, at the mouth, and that is a measurement.**

        The transport case for holding it through the seating is obvious and it
        is wrong. Section 6 of ``docs/service_interface_spec.md`` establishes
        that this rack's lead-in does not assist the insertion, it *performs*
        it: the flares walk the module into a 0.75 mm-per-side channel by
        contact, and two fully trained policies insert nothing without them.
        Contact can only walk a module that is free to be walked. A module
        rigidly locked to a wrist is not, and measured that way it does not move
        at all: the seating phase spent its entire 30-second budget with the
        module at 0.5818 m, having advanced 0.3 mm, while the arm pushed a
        rigid link against a channel.

        So the lock carries the module through free space and hands it to the
        rack, which is the division of work the rack was designed for. The pads
        keep it -- they are what the insert skill was always certified on -- and
        the *hand* opens only after the settled seating re-check.

        The geometric interlock in ``service_latch.release_before_blade_centre_x_m``
        is kept as the outer bound it always was: even if the lock were held
        longer, an engaged jaw enters the slot mouth once the module centre
        passes 0.733 m, and the release below happens 145 mm before that.
        """

        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        # The lock was softened at the end of the squaring leg, where the
        # module's nose reaches the lead-in. This is the backstop for a state
        # that should not occur.
        still_rigid = mask & grapple_latch_rigid(self.task)
        if MATING_MODE == "rigid":
            still_rigid = torch.zeros_like(still_rigid)
        if bool(still_rigid.any()):
            soften_grapple_latch(self.task, still_rigid)
            self.latch_softened |= still_rigid
            self.latch_softened_at[still_rigid & (self.latch_softened_at < 0)] = _step
        self.guarded_insert_target_x[ids] = _blade_centre_x(self.task)[ids]
        engaged_far = LATCH_ENGAGED_FAR_DEPTH_M + self.latch_seek_travel_m[ids]
        self.guarded_insert_release_x[ids] = (
            SLOT_MOUTH_X
            + 0.5 * BLADE_LENGTH_M
            + (LATCH_MODULE_FACE_DEPTH_M - engaged_far)
            - GUARDED_INSERT_RELEASE_MARGIN_M
        )

    def _step_guarded_insert(self, inserting: torch.Tensor, step: int, tool: torch.Tensor, tool_rot: torch.Tensor) -> torch.Tensor:
        """Drive the module into the second bay, guarded, and let go on the way.

        **Why this is not the learned insert policy.** It could have been, and
        the flag to try it is still there. It is not, because the promoted
        two-bay insert checkpoint certifies at 10.50% on the moved workcell --
        0.00% in the first bay and 21.45% in the second
        (``evidence/grapple_insert_two_slot_w65_certification.json``) -- against
        98.60% for the same skill on the cell it was trained on. Handing a
        robot-carried module to a policy measured at 21% would make the chain's
        number a statement about that policy rather than about the interface
        this branch exists to test, and the honest alternative was named in the
        task: a guarded robot-held insertion. This is it, and the report labels
        it scripted.

        **What "guarded" means here, exactly.** The axial target advances by a
        bounded step **only while the deployed estimator says the module is
        inside the lateral and attitude envelope**, and stops advancing when it
        is not. Perception is therefore in the loop and fails closed: a lost
        fiducial stops the insertion rather than continuing blind. The estimate
        comes from ``_payload_feedback``, which is the same RGB-D estimate the
        policies see; simulator truth is used only to score the result.

        **The lock is already off by the time this runs.** It is released at the
        hand-off, in ``_begin_guarded_insert``, and the reason is measured
        there: a module rigidly locked to a wrist cannot be walked into a
        0.75 mm-per-side channel by contact, and contact is what this rack's
        lead-in does. The seating is therefore pad-held, which is what the
        insert skill was always certified on, and the *hand* opens only after
        the settled re-check. The geometric interlock below is the outer bound
        that would have forced the release anyway.
        """

        task = self.task
        ids = torch.nonzero(inserting, as_tuple=False).squeeze(-1)
        estimated_position, estimated_orientation, _velocity = self._payload_feedback()
        module_pos = estimated_position[ids]
        module_rot = estimated_orientation[ids]
        seated = module_pos.new_tensor(SECOND_SLOT_INSERTED_POS)
        staging_rot = self.relocation_staging_rot.unsqueeze(0).expand_as(module_rot)

        lateral_error = torch.linalg.vector_norm(module_pos[:, 1:3] - seated[1:3].unsqueeze(0), dim=-1)
        orientation_error = torch.linalg.vector_norm(
            axis_angle_from_quat(quat_mul(staging_rot, quat_inv(module_rot))), dim=-1
        )
        lateral_tolerance = GUARDED_INSERT_LATERAL_TOLERANCE_M
        orientation_tolerance = GUARDED_INSERT_ORIENTATION_TOLERANCE_RAD
        sensor_ready = torch.ones_like(ids, dtype=torch.bool)
        estimator = getattr(task, "_module_state_estimator", None)
        if estimator is not None and estimator.backend == "fiducial_pnp":
            lateral_tolerance = FIDUCIAL_GUARDED_LATERAL_TOLERANCE_M
            orientation_tolerance = FIDUCIAL_GUARDED_ORIENTATION_TOLERANCE_RAD
            sensor_ready = estimator.fiducial_current_detection[ids]
        clear_to_advance = sensor_ready & (lateral_error <= lateral_tolerance) & (orientation_error <= orientation_tolerance)
        self.guarded_insert_steps[ids] += clear_to_advance.to(torch.long)
        self.guarded_insert_holds[ids] += (~clear_to_advance).to(torch.long)

        # The axial target leads the measured module by a bounded amount, so a
        # step the arm cannot take does not accumulate into a lunge.
        lead = (self.guarded_insert_target_x[ids] - module_pos[:, 0]).clamp(
            -GUARDED_INSERT_MAX_LEAD_M, GUARDED_INSERT_MAX_LEAD_M
        )
        advanced = torch.where(
            clear_to_advance,
            module_pos[:, 0] + lead + GUARDED_INSERT_AXIAL_STEP_M,
            module_pos[:, 0] + lead,
        )
        self.guarded_insert_target_x[ids] = torch.minimum(advanced, seated[0].expand_as(advanced))
        target_module_pos = torch.stack(
            (
                self.guarded_insert_target_x[ids],
                seated[1].expand_as(advanced),
                seated[2].expand_as(advanced),
            ),
            dim=-1,
        )
        target_tool_pos, target_tool_rot = self._rigid_tool_command(
            ids,
            target_module_pos + task.scene.env_origins[ids],
            staging_rot,
            tool_rot[ids],
            module_pos + task.scene.env_origins[ids],
        )
        self._drive_tool_to(
            ids, tool, tool_rot, target_tool_pos, target_tool_rot, self.scales[INSERT], RIGID_TRANSIT_ATTITUDE_AUTHORITY
        )
        # Hold the module firmly while it is being driven between rails. The
        # operating rule this project measured is capture gently, hold hard to
        # move, retain once free; this phase moves.
        self.gripper.retain_latch[inserting] = False

        # The backstop. The lock is released at the hand-off above, so this can
        # only fire if some future change holds it longer; it is simulator truth
        # deliberately, because it is a geometric interlock protecting the rack
        # from a mechanism rather than a perception decision, and it must not be
        # able to fail open because a marker was occluded.
        true_blade_x = _blade_centre_x(task)[ids]
        release_depth = self.guarded_insert_release_x[ids]
        due_to_release = inserting.clone()
        due_to_release[:] = False
        due_to_release[ids] = (true_blade_x >= release_depth) & grapple_latch_rigid(task)[ids]
        if bool(due_to_release.any()):
            release_grapple_latch(task, due_to_release)
            self.latch_released |= due_to_release
            self.latch_released_at[due_to_release & (self.latch_released_at < 0)] = step

        # Seated. The lock lets go completely now, so the 0.70 s re-check that
        # decides the outcome is taken on a module held by nothing but the pads
        # and its own rails -- a spring across that window would make the
        # settling check a statement about the spring.
        seated = grapple_insertion_success_mask(task)
        fired = inserting & seated & (
            torch.ones_like(seated) if MATING_MODE == "rigid" else ~grapple_latch_rigid(task)
        )
        if bool(fired.any()):
            release_grapple_latch(task, fired)
            self.latch_released |= fired
            self.latch_released_at[fired & (self.latch_released_at < 0)] = step
        return fired

    def _front_overhang_x(self, ids: torch.Tensor) -> torch.Tensor:
        """Distance from the module centre to its furthest-forward corner, now.

        Read from the module's measured attitude and its own authored size, so
        a module that comes out of the rails askew is retreated for what it
        actually is rather than for what a square one would be. The grapple pin
        is ignored deliberately: it protrudes from the *rear* face, away from
        the rack, and adding it here would retreat the arm past its own reach
        boundary for a feature that cannot touch anything.
        """

        blade = self.task.scene["spare_blade"]
        size = getattr(getattr(blade.cfg, "spawn", None), "size", None)
        half = blade.data.root_pos_w.new_tensor(
            (0.5 * BLADE_LENGTH_M, 0.080, 0.0175) if size is None else tuple(0.5 * value for value in size)
        )
        signs = torch.tensor(
            [
                (sx, sy, sz)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            device=half.device,
            dtype=half.dtype,
        )
        corners = signs * half
        orientation = blade.data.root_quat_w[ids].unsqueeze(1).expand(-1, corners.shape[0], -1)
        rotated = quat_apply(orientation.reshape(-1, 4), corners.repeat(ids.numel(), 1))
        return rotated.reshape(ids.numel(), corners.shape[0], 3)[..., 0].amax(dim=-1)

    def _plan_lateral_transit(
        self,
        mask: torch.Tensor,
        tool: torch.Tensor,
        tool_rot: torch.Tensor,
        blade_x: torch.Tensor,
    ) -> None:
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
        task = self.task
        # The periodic extraction trace can be up to three control steps old
        # when the success predicate fires.  Under payload load that gap was
        # already 0.13 rad in one joint, so replay could not even reach its
        # first saved target.  Make the actual hand-off state the last sample;
        # the reverse path then starts continuously from where the arm is.
        terminal_slots = self.waypoint_write[ids].clamp(max=self.max_waypoints - 1)
        self.waypoints[terminal_slots, ids] = tool[ids]
        self.extraction_joint_waypoints[terminal_slots, ids] = task.scene["robot"].data.joint_pos[ids][
            :, self.arm_joint_ids
        ]
        blade_asset = task.scene["spare_blade"]
        self.extraction_blade_pose_waypoints[terminal_slots, ids, :3] = (
            blade_asset.data.root_pos_w[ids] - task.scene.env_origins[ids]
        )
        self.extraction_blade_pose_waypoints[terminal_slots, ids, 3:] = blade_asset.data.root_quat_w[ids]
        self.waypoint_write[ids] = (self.waypoint_write[ids] + 1).clamp(max=self.max_waypoints - 1)
        # Capture the actual rigid transform at extraction clear. The final
        # tool target is then solved from the desired bay-1 module pose and the
        # rack-aligned tool attitude. This remains correct when the offset has
        # rotated; adding a world-axis offset measured before the rotation does
        # not.
        blade_world = blade_asset.data.root_pos_w
        if self.base_rail_enabled:
            current_blade_y = blade_world[ids, 1] - task.scene.env_origins[ids, 1]
            # The shuttle's zero is captured at handoff, so its lateral target
            # is the measured bay-centre correction from that pose.
            self.rail_goal_target_m[ids] = (SECOND_SLOT_CENTER_Y - current_blade_y).clamp(
                BASE_STAGE_MIN_TARGET_M[1], BASE_STAGE_MAX_TARGET_M[1]
            )
        blade_relative_to_tool = quat_apply(quat_inv(tool_rot[ids]), blade_world[ids] - tool[ids])
        blade_relative_rot_to_tool = quat_mul(quat_inv(tool_rot[ids]), blade_asset.data.root_quat_w[ids])
        desired_blade_rot = self.relocation_staging_rot.unsqueeze(0).expand_as(tool_rot[ids])
        desired_tool_rot = quat_mul(desired_blade_rot, quat_inv(blade_relative_rot_to_tool))
        desired_blade_world = self.relocation_staging_pos.unsqueeze(0) + self.task.scene.env_origins[ids]
        final_tool_hold = desired_blade_world - quat_apply(tool_rot[ids], blade_relative_to_tool)
        final_tool_aligned = desired_blade_world - quat_apply(desired_tool_rot, blade_relative_to_tool)
        self.relocation_hold_tool_rot[ids] = tool_rot[ids]
        self.relocation_blade_relative_to_tool[ids] = blade_relative_to_tool
        self.relocation_blade_relative_rot_to_tool[ids] = blade_relative_rot_to_tool
        self.relocation_desired_tool_rot[ids] = desired_tool_rot
        self.relocation_final_tool_hold[ids] = final_tool_hold
        self.relocation_final_tool_aligned[ids] = final_tool_aligned
        self.relocation_aligning[ids] = False
        self.relocation_aligned[ids] = False
        self.relocation_joint_replaying[ids] = False
        source_staging = self.relocation_staging_pos.clone()
        source_staging[1] = 0.0
        for env_id in ids.detach().cpu().tolist():
            count = int(self.waypoint_write[env_id])
            if count <= 0:
                raise RuntimeError(f"No extraction joint path was recorded for relocation env {env_id}")
            recorded_positions = self.extraction_blade_pose_waypoints[:count, env_id, :3]
            self.relocation_joint_replay_stop[env_id] = torch.argmin(
                torch.linalg.vector_norm(recorded_positions - source_staging, dim=-1)
            )

        tool_to_blade_x = tool[ids, 0] - blade_x[ids]
        # How far the module's furthest-forward corner really is from its
        # centre, at the attitude the rails have just released it in. For a
        # square module this is half its length and the nominal constant is
        # exact; for a real one it is larger, and the difference is the whole
        # reason the first crossing leg stalled against a lead-in plate.
        overhang = self._front_overhang_x(ids)
        measured_clear_centre_x = FLARE_LEADING_X - overhang - TRANSIT_FLARE_CLEARANCE_M
        clear_centre_x = torch.minimum(
            measured_clear_centre_x,
            measured_clear_centre_x.new_full((), TRANSIT_CLEAR_BLADE_CENTRE_X),
        )
        self.transit_clear_centre_x[ids] = clear_centre_x
        self.transit_front_overhang_m[ids] = overhang
        retreat_x = clear_centre_x + tool_to_blade_x

        back = tool[ids].clone()
        back[:, 0] = retreat_x
        across = back.clone()
        across[:, 1] = final_tool_hold[:, 1]
        approach = final_tool_hold

        # Written in reverse, because the follower walks the buffer downwards.
        self.waypoints[2, ids] = back
        self.waypoints[1, ids] = across
        self.waypoints[0, ids] = approach
        if self.rigid_transit:
            # Four module poses. Legs 3 and 2 keep the attitude the rails
            # released the module in: rotating it while its nose is still
            # between the flares is the one thing the retreat exists to avoid,
            # and rotating it while crossing is what the arm cannot afford.
            module_local = blade_world[ids] - task.scene.env_origins[ids]
            held_rot = blade_asset.data.root_quat_w[ids]
            square_rot = self.relocation_staging_rot.unsqueeze(0).expand(ids.numel(), -1)
            staging = self.relocation_staging_pos.unsqueeze(0).expand_as(module_local)
            crossed = torch.stack((clear_centre_x, staging[:, 1], staging[:, 2]), dim=-1)
            retreated = torch.stack(
                (clear_centre_x, module_local[:, 1], module_local[:, 2]), dim=-1
            )
            # 4: retreat, holding whatever attitude the rails released.
            self.module_leg_pos[4, ids] = retreated
            self.module_leg_rot[4, ids] = held_rot
            # 3: square, at the source bay's retreat depth. Squaring only ever
            # *shortens* the module's forward overhang, so the clearance leg 4
            # bought is not spent by it, and doing it here means the crossing
            # has an attitude to hold rather than one to change.
            self.module_leg_pos[3, ids] = retreated
            self.module_leg_rot[3, ids] = square_rot
            # 2: cross.
            self.module_leg_pos[2, ids] = crossed
            self.module_leg_rot[2, ids] = square_rot
            # 1: square again, at the destination bay, because the crossing is
            # measured to give some of it back.
            self.module_leg_pos[1, ids] = crossed
            self.module_leg_rot[1, ids] = square_rot
            # 0: in, to the pose the insertion begins from.
            self.module_leg_pos[0, ids] = staging
            self.module_leg_rot[0, ids] = square_rot
            self.transit_leg_entered[ids] = 0
        if self.base_rail_enabled:
            # The physical shuttle accepts the ORU as soon as learned
            # extraction clears the rails.  It owns the collision-clear axial
            # retreat, cross-bay move, pose staging, and guarded insertion;
            # making the arm perform another long retreat before handoff wasted
            # almost the entire demonstration clock and added no capability.
            self._engage_payload_stage(ids)
            self.stage_drive_target_m[ids] = 0.0
            self.stage_goal_target_m[ids] = 0.0
            self.stage_rotation_drive_target_deg[ids] = 0.0
            self.relocation_aligned[ids] = True
            # The now-open gripper first retreats to the already planned clear
            # waypoint while the shuttle holds a zero target.  Starting the
            # lateral move with the fingers still around the handle physically
            # blocked the payload and wound the D6 target into its limit.
            self.waypoint_read[ids] = 2
        else:
            self.waypoint_read[ids] = 2
        # The axis each leg is laid out along, so the follower can ask whether
        # the leg is finished rather than whether the tool is at a point. Holding
        # the module's attitude moves the tool off that point on the other two
        # axes, which is correct behaviour and used to stall the follower.
        self.leg_axis[2, ids] = 0  # back: along x
        self.leg_axis[1, ids] = 1  # across: along y
        self.leg_axis[0, ids] = 0  # in again: along x
        if self.rigid_transit:
            # **Last, because the branch above sets it too.** Written before it,
            # the rigid plan's first leg was silently replaced by the pad-held
            # follower's, which starts at the *cross* waypoint -- so the module
            # retreated and crossed at once, cutting the corner diagonally
            # across the source bay's lead-in flare, which is the single thing
            # the retreat leg exists to prevent. It completed anyway, which is
            # exactly why this is worth a comment rather than a fix and a
            # silence.
            self.waypoint_read[ids] = RIGID_TRANSIT_LEGS - 1

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
        all_tool, all_tool_rot = end_effector_pose_world(self.task)
        tool = all_tool[ids]
        tool_rot = all_tool_rot[ids]
        target = self.waypoints[self.waypoint_read[ids], ids]
        distance = torch.linalg.vector_norm(target - tool, dim=-1)
        attitude_distance = torch.linalg.vector_norm(
            axis_angle_from_quat(quat_mul(self.relocation_desired_tool_rot[ids], quat_inv(tool_rot))),
            dim=-1,
        )
        blade_orientation = self.task.scene["spare_blade"].data.root_quat_w[ids]
        blade_attitude_distance = torch.linalg.vector_norm(
            axis_angle_from_quat(
                quat_mul(
                    self.relocation_staging_rot.unsqueeze(0).expand_as(blade_orientation),
                    quat_inv(blade_orientation),
                )
            ),
            dim=-1,
        )
        legs = torch.bincount(self.waypoint_read[ids], minlength=RIGID_TRANSIT_LEGS).tolist()
        # Each conjunct of `arrived` separately, because the first relocation run
        # showed the tool sitting 0.4 mm from its last waypoint and not arriving,
        # and a single boolean cannot say which clause is the one refusing.
        blade_x = _blade_centre_x(self.task)[ids]
        lateral = (self.task.scene["spare_blade"].data.root_pos_w[:, 1] - self.task.scene.env_origins[:, 1])[ids]
        # Is the module still on the tool at all? A tool that flies its whole
        # path while the module does not follow has either lost the grip or is
        # dragging the module against something, and grip error tells the two
        # apart: a lost module's error grows without bound, a snagged one's does
        # not. Reported here because the hand-off trace only samples phase
        # boundaries, and this failure lives in the middle of a phase.
        grip_error, _ = grapple_grip_error_metrics(self.task)
        grip_error = grip_error[ids]
        latch = grapple_latch_diagnostics(self.task)
        return (
            f"  transit: legs_remaining={legs[:RIGID_TRANSIT_LEGS]} "
            f"to_waypoint_m p50={float(distance.median()):.4f} max={float(distance.max()):.4f} "
            f"| last_leg={int((self.waypoint_read[ids] <= 0).sum())} "
            f"blade_x_ok={int((blade_x >= TRANSIT_TARGET_BLADE_X - 0.005).sum())} "
            f"(p50={float(blade_x.median()):.4f} need>={TRANSIT_TARGET_BLADE_X - 0.005:.4f}) "
            f"crossed={int((lateral <= 0.5 * SECOND_SLOT_CENTER_Y).sum())} "
            f"(p50={float(lateral.median()):.4f} need<={0.5 * SECOND_SLOT_CENTER_Y:.4f}) "
            f"| grip_error_m p50={float(grip_error.median()):.4f} max={float(grip_error.max()):.4f} "
            f"latch={int(latch['engaged'][ids].sum())}/{int(ids.numel())} "
            f"latch_rel_m max={float(latch['position_error_m'][ids].max()):.4f} "
            f"latch_rel_rad max={float(latch['orientation_error_rad'][ids].max()):.4f} "
            f"tool_attitude_rad p50={float(attitude_distance.median()):.4f} "
            f"blade_attitude_rad p50={float(blade_attitude_distance.median()):.4f} "
            f"stage_state={int(self.relocation_stage_lateral_done[ids].sum())}/"
            f"{int(self.relocation_stage_translated[ids].sum())}/"
            f"{int(self.relocation_aligning[ids].sum())}/"
            f"{int(self.relocation_aligned[ids].sum())} "
            f"stage_rot_target_deg={[round(float(value), 2) for value in self.stage_rotation_drive_target_deg[ids[0]]]} "
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
        steps = torch.where(self.done_steps > 0, self.done_steps, task.episode_length_buf.to(self.done_steps.dtype)).to(
            torch.float64
        )
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


def _transit_retention_report(driver, arguments) -> dict[str, object]:
    """Summarise what happened to the tool-to-module transform during transit.

    Acceptance gate 3 of this branch's task is that *robot* motion commands, not
    module motion commands, produce the transfer, and that the tool-to-module
    pose stays bounded and is recorded through it. That is this block. It is
    written from accumulators the driver keeps every control step, so it exists
    whether or not a trace file was requested, and it reports the payload-stage
    baseline honestly as a run in which the robot was *not* the carrier.
    """

    carried = not arguments.base_rail_on_relocation
    sampled = driver.transit_samples > 0
    count = int(sampled.sum())

    def summarize(values, mask) -> dict[str, object] | None:
        selected = values[mask]
        if selected.numel() == 0:
            return None
        finite = selected[torch.isfinite(selected)]
        if finite.numel() == 0:
            return None
        ordered = torch.sort(finite).values
        return {
            "count": int(finite.numel()),
            "mean": float(finite.mean()),
            "p50": float(ordered[int(0.50 * (finite.numel() - 1))]),
            "p95": float(ordered[int(0.95 * (finite.numel() - 1))]),
            "max": float(finite.max()),
            "min": float(finite.min()),
        }

    within = sampled & (driver.transit_loss_step < 0)
    return {
        "carrier": ("six_axis_robot" if carried else "world_aligned_payload_stage"),
        "claim_is_that_the_robot_carried_the_module": carried,
        "environments_that_entered_transit": count,
        "environments_retaining_the_planned_transform_throughout": int(within.sum()),
        # Not a chosen tolerance. See TRANSIT_RETENTION_POSITION_LIMIT_M.
        "retention_limit_position_m": TRANSIT_RETENTION_POSITION_LIMIT_M,
        "retention_limit_orientation_rad": TRANSIT_RETENTION_ORIENTATION_LIMIT_RAD,
        "retention_limit_derivation": (
            "the insert hand-off envelope, read backwards: the transit is planned from the "
            "tool-to-module transform measured at rail release, so drift in it lands on the "
            "staging pose one for one"
        ),
        "reference_frame": "tool frame at the instant extraction cleared the rails",
        # The retreat is laid out from the module's measured corners, not from
        # its nominal half-length, because a module leaving the rails is not
        # square to them. Reported so the difference between the two is visible
        # rather than buried in a constant.
        "flare_leading_plane_x_m": FLARE_LEADING_X,
        "nominal_clear_blade_centre_x_m": TRANSIT_CLEAR_BLADE_CENTRE_X,
        "flare_clearance_margin_m": TRANSIT_FLARE_CLEARANCE_M,
        "measured_front_overhang_m": summarize(driver.transit_front_overhang_m, sampled),
        # Which legs the follower had to give up on, and what attitude they were
        # left at. Empty on a run where every leg met its gate.
        "legs": [
            {
                "leg": index,
                "role": ("retreat", "square_at_source", "cross", "square_at_destination", "insert_approach")[
                    RIGID_TRANSIT_LEGS - 1 - index
                ],
                "environments_forced_by_timeout": int(driver.transit_leg_forced[index][sampled].sum()),
                "residual_orientation_rad": summarize(
                    driver.transit_leg_residual_rad[index], sampled & driver.transit_leg_forced[index]
                ),
                "residual_position_m": summarize(
                    driver.transit_leg_residual_m[index], sampled & driver.transit_leg_forced[index]
                ),
            }
            for index in range(RIGID_TRANSIT_LEGS - 1, -1, -1)
        ],
        "retreat_clear_blade_centre_x_m": summarize(driver.transit_clear_centre_x, sampled),
        "max_tool_to_module_position_drift_m": summarize(driver.transit_max_drift_m, sampled),
        "max_tool_to_module_orientation_drift_rad": summarize(driver.transit_max_drift_rad, sampled),
        "terminal_tool_to_module_position_drift_m": summarize(driver.transit_final_drift_m, sampled),
        "terminal_tool_to_module_orientation_drift_rad": summarize(driver.transit_final_drift_rad, sampled),
        "max_grip_error_m": summarize(driver.transit_max_grip_error_m, sampled),
        "max_grip_attitude_rad": summarize(driver.transit_max_grip_attitude_rad, sampled),
        "min_grip_drive_torque_nm": summarize(driver.transit_min_drive_torque_nm, sampled),
        "tool_travel_m": summarize(driver.transit_tool_travel_m, sampled),
        "module_travel_m": summarize(driver.transit_module_travel_m, sampled),
        # A module that travels with the tool has followed it; a module that
        # travels much less has been left behind, and a pooled drift maximum
        # alone cannot tell those apart from a short transit.
        "control_steps_before_retention_was_lost": summarize(
            driver.transit_loss_step.to(torch.float64), sampled & (driver.transit_loss_step >= 0)
        ),
        "observed_per_environment": [
            {
                "env": index,
                "entered_transit": bool(sampled[index]),
                "samples": int(driver.transit_samples[index]),
                "max_position_drift_m": float(driver.transit_max_drift_m[index]),
                "max_orientation_drift_rad": float(driver.transit_max_drift_rad[index]),
                "terminal_position_drift_m": float(driver.transit_final_drift_m[index]),
                "terminal_orientation_drift_rad": float(driver.transit_final_drift_rad[index]),
                "tool_travel_m": float(driver.transit_tool_travel_m[index]),
                "module_travel_m": float(driver.transit_module_travel_m[index]),
                "measured_front_overhang_m": float(driver.transit_front_overhang_m[index]),
                "retreat_clear_blade_centre_x_m": float(driver.transit_clear_centre_x[index]),
                "retained_throughout": bool(within[index]),
                "driver_step_retention_lost": (
                    int(driver.transit_loss_step[index]) if int(driver.transit_loss_step[index]) >= 0 else None
                ),
            }
            for index in range(driver.transit_samples.numel())
        ],
    }


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
        if args.latch_on_release:
            # **Set before configure_robustness, not after, and that ordering is
            # the whole of it.** configure_robustness rebuilds the event set and
            # ends by calling _configure_latch(), which sets
            # events.grapple_latch = None whenever latch_enabled is False. Once
            # the term is None it cannot be re-enabled: the next
            # _configure_latch() dereferences it and raises
            # "AttributeError: 'NoneType' object has no attribute 'params'".
            # That is exactly how the first attempt at this flag failed, and it
            # failed silently into a report file rather than loudly.
            #
            # A modelled latch engaged on *capture* was swept from 10 to 160 N-m
            # and refuted:
            # a restoring torque on a module the rails still hold jams it in the
            # rails and extraction travel collapses from 458 mm to about 25 mm.
            # Engaged the instant the rails let go it has nothing left to jam the
            # module against, and that is the phase that needs it -- the module
            # is unconstrained for the whole crossing and nothing else opposes a
            # moment about the closing axis. Release-time bounded-compliance
            # probes are now implemented and have remained stable, but have not
            # completed relocation; the flag is retained for reproducible design
            # experiments, not enabled in the service preset.
            env_cfg.latch_enabled = True
            env_cfg.latch_engages_on_release = True
            env_cfg.latch_rated_torque_nm = args.latch_rated_torque_nm
            env_cfg.latch_rated_force_n = args.latch_rated_force_n
            env_cfg.latch_position_stiffness_n_per_m = args.latch_position_stiffness_n_per_m
            env_cfg.latch_position_damping_ratio = args.latch_position_damping_ratio
            env_cfg.latch_rotation_stiffness_nm_per_rad = args.latch_rotation_stiffness_nm_per_rad
            env_cfg.latch_rotation_damping_ratio = args.latch_rotation_damping_ratio
            env_cfg.latch_joint_mode = args.latch_joint_mode
            # **A compliance device's maximum force is its stiffness times its
            # stroke, and deriving it is not tidiness.** Fixed at 400 N against
            # a 40 kN/m spring, the cap bound at 10 mm of deflection and the
            # spring became a constant 400 N pull -- which is not a spring, and
            # measured, the module slid 208 mm back toward the bay it came from
            # while the mechanism reported itself engaged the whole way.
            env_cfg.mating_force_cap_n = args.mating_force_cap_n or (
                args.latch_position_stiffness_n_per_m * MATING_TRAVEL_LIMIT_M
            )
            env_cfg.mating_torque_cap_nm = args.latch_rotation_stiffness_nm_per_rad * MATING_ROTATION_LIMIT_RAD
            if args.latch_joint_mode == "fixed":
                env_cfg.scene.release_latch_joint.spawn.break_force_n = args.latch_rated_force_n
                env_cfg.scene.release_latch_joint.spawn.break_torque_nm = args.latch_rated_torque_nm
                # The release latch is a procedurally authored, per-environment
                # joint, and PhysX scene replication copies only the first one.
                # A 32-environment batch therefore came up with a latch prim in
                # every environment and a usable joint in exactly one, and
                # failed at construction rather than silently -- which is the
                # right failure, but it is avoidable. Author every environment
                # independently, as ``configure_base_rail`` already does for the
                # payload stage and as the camera workflow does for its sensors.
                env_cfg.scene.replicate_physics = False
                env_cfg.scene.clone_in_fabric = False
        globals()["MATING_MODE"] = args.mating_mode
        env_cfg.base_rail_enabled = args.base_rail_on_relocation
        env_cfg.configure_robustness(0)
        if args.base_rail_on_relocation:
            env_cfg.configure_base_rail()
        if (
            args.latch_on_release
            and args.workflow == "relocate"
            and not args.base_rail_on_relocation
            and hasattr(env_cfg, "configure_service_destination")
        ):
            # A robot-carried module enters the destination bay from *outside*
            # the rack, which no skill in this repository has ever done: both
            # insertion skills reset with the module already in its channel. The
            # bay therefore needs a lead-in on the vertical axis as well as the
            # lateral one, sized by the attitude a six-axis arm can actually
            # deliver through free space. Installed only on this path, so every
            # task an existing certification describes is unchanged.
            env_cfg.service_destination_channel_relief_m = args.destination_channel_relief_m
            env_cfg.configure_service_destination()
            print(
                "[INFO] Destination bay fitted with its vertical lead-in "
                "(16.6 mm per side, accepting 0.074 rad of delivered attitude error); "
                f"channel relief {args.destination_channel_relief_m * 1000.0:.2f} mm per side",
                flush=True,
            )
        # The full physical stage motion can legitimately outlive the original
        # per-skill episode. Keep the environment alive for the explicitly
        # requested driver horizon; learned phases still retain their own
        # certified deadlines below.
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args.steps / 30.0 + 2.0)
        if args.latch_on_release:
            if env_cfg.events.grapple_latch is None:
                raise RuntimeError(
                    "--latch_on_release did not survive configure_robustness: the task's "
                    "_configure_latch() removed the term. The flag must be set before it runs."
                )
            interface_detail = (
                "two-body fixed joint with reaction wrench"
                if args.latch_joint_mode == "fixed"
                else (
                    f"stiffness {args.latch_position_stiffness_n_per_m} N/m / "
                    f"{args.latch_rotation_stiffness_nm_per_rad} N-m/rad"
                )
            )
            print(
                f"[INFO] Latch ARMED ON RELEASE at {args.latch_rated_force_n} N / "
                f"{args.latch_rated_torque_nm} N-m; {interface_detail} "
                "(engages when the driver retains the grip, not on capture)"
            )
        env_cfg.seed = args.seed
        if hasattr(env_cfg, "perception_backend"):
            env_cfg.perception_backend = args.perception_backend
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
            PHASE_BUDGET_S[index] * (args.transit_slowdown if index == TRANSIT and args.workflow != "relocate" else 1)
            for index in phases
        )
        if args.workflow != "remove":
            # The scripted realign runs inside the seat phase, so it is not in
            # PHASE_BUDGET_S and has to be added to the episode explicitly.
            budget += ALIGN_STEPS / 30.0
        env_cfg.episode_length_s = round(
            max(budget + SETTLE_STEPS / 30.0 + 1.0, args.steps / 30.0 + 2.0),
            2,
        )
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
        if args.video and env_cfg.sim.render_interval > env_cfg.decimation:
            # The vision task normally renders at the 15 Hz sensor cadence
            # (eight physics steps) while control and Gym video advance at
            # 30 Hz (four physics steps).  RecordVideo otherwise receives an
            # unrendered frame every other control step, producing clips that
            # are approximately half black.  Rendering the viewer at control
            # cadence does not change the camera's configured 15 Hz update
            # period or any policy observation; it only makes the artifact
            # faithful to the executed run.
            env_cfg.sim.render_interval = env_cfg.decimation
            print("[INFO] Viewer render cadence raised to 30 Hz for video capture", flush=True)

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
            release_latch_required=args.latch_on_release,
            base_rail_enabled=args.base_rail_on_relocation,
            base_rail_arm_mode=args.base_rail_arm_mode,
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
            if single and int(driver.phase[0]) == TRANSIT and step % 120 == 0:
                print(f"[CHAIN] step {step:5d}{driver.transit_progress()}", flush=True)
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
                f"{trace['handoff'].shape[0]} hand-offs, {trace['transit'].shape[0]} transit samples, "
                f"{trace['settle'].shape[0]} settling rows",
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
            "runtime_source_bindings": [
                {
                    "path": path.as_posix(),
                    "sha256": hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest(),
                }
                for path in WORKFLOW_RUNTIME_SOURCES
            ],
            "loaded_but_not_executed_policies": (
                ["insert"] if args.workflow == "relocate" and args.base_rail_on_relocation else []
            ),
            "learned_phases": {
                "remove": ["capture", "extract"],
                "install": ["capture", "insert"],
                "relocate": (
                    ["capture", "extract"] if args.base_rail_on_relocation else ["capture", "extract", "insert"]
                ),
            }[args.workflow],
            "scripted_phases": ["seat"]
            + (
                ["payload_handoff", "shuttle_retreat", "cross_bay", "guarded_insert"]
                if args.workflow == "relocate" and args.base_rail_on_relocation
                else ["transit"]
                if args.workflow == "relocate"
                else []
            ),
            # Whether the robot carried the module, as a number rather than as
            # a claim. Present on every workflow that has a transit, including
            # the payload-stage baseline, so the two can be read side by side
            # and the baseline cannot quietly borrow this section's language.
            "robot_carried_transit": _transit_retention_report(driver, args),
            "insert_handoff_contract": (
                {
                    "receiver": (
                        "guarded_robot_driven_insertion"
                        if args.latch_on_release and args.mating_mode == "compliant"
                        else "learned_insert_policy_reset_distribution"
                    ),
                    "axial_tolerance_m": (
                        INSERTION_AXIAL_DEPTH_TOLERANCE_M
                        if args.latch_on_release and args.mating_mode == "compliant"
                        else INSERT_HANDOFF_POSITION_TOLERANCE_M
                    ),
                    "lateral_tolerance_m": (
                        SLOT_ENTRY_RAMP_CATCH_M
                        if args.latch_on_release and args.mating_mode == "compliant"
                        else INSERT_HANDOFF_POSITION_TOLERANCE_M
                    ),
                    "orientation_tolerance_rad": INSERTION_ORIENTATION_TOLERANCE_RAD,
                    "note": (
                        "The tolerance is the receiving controller's own precondition. The learned "
                        "policy's is its reset distribution; the guarded controller's is the bay's "
                        "lead-in catch, which is what it needs to finish the job."
                    ),
                }
                if args.workflow == "relocate"
                else None
            ),
            "transit_planner": (
                {
                    "type": (
                        "physical_payload_shuttle_retreat_cross_align_and_guarded_insert"
                        if args.base_rail_on_relocation
                        else "cartesian_clearance_translate_then_stationary_alignment"
                    ),
                    "lateral_crossing_attitude": (
                        "held_by_world_aligned_six_axis_payload_stage"
                        if args.base_rail_on_relocation
                        else "unconstrained_for_fixed_latch_only"
                    ),
                    "physical_d6_drive_target_step_m": (
                        BASE_RAIL_TARGET_STEP_M if args.base_rail_on_relocation else None
                    ),
                    "physical_d6_drive_target_range_m": (
                        {"minimum_xyz": BASE_STAGE_MIN_TARGET_M, "maximum_xyz": BASE_STAGE_MAX_TARGET_M}
                        if args.base_rail_on_relocation
                        else None
                    ),
                    "kinematic_mount_transform_writes": False,
                    "physical_d6_translation_drive_target_writes": (
                        ["transX", "transY", "transZ"] if args.base_rail_on_relocation else []
                    ),
                    "physical_d6_rotation_drive_target_writes": (
                        ["rotX", "rotY", "rotZ"] if args.base_rail_on_relocation else []
                    ),
                    "physical_d6_rotation_target_limit_deg": (
                        BASE_STAGE_ROTATION_LIMIT_DEG if args.base_rail_on_relocation else None
                    ),
                    "arm_joint_position_target_hold": False if args.base_rail_on_relocation else None,
                    "stage_arm_stiffness_multiplier": (
                        BASE_STAGE_ARM_STIFFNESS_MULTIPLIER if args.base_rail_on_relocation else None
                    ),
                    "physical_payload_stage_handoff": args.base_rail_on_relocation,
                    "arm_mode": (
                        "robot_releases_after_measured_payload_stage_handoff" if args.base_rail_on_relocation else None
                    ),
                    "robot_or_payload_state_writes": False,
                    "destination_interface": (
                        {
                            "straight_rail_total_clearance_mm": 4.5,
                            "floor_lead_in_clearance_mm": 2.0,
                            "passive_flare_collision": False,
                        }
                        if args.base_rail_on_relocation
                        else None
                    ),
                    "rail_force_measured": False if args.base_rail_on_relocation else None,
                }
                if args.workflow == "relocate"
                else None
            ),
            "success_definition": (
                "rack seating (depth, lateral pose, attitude, and velocity) re-checked after a "
                f"{SETTLE_STEPS / 30.0:.2f} s settling window; the robot grasp is intentionally "
                "excluded after physical handoff to the payload shuttle"
                if args.base_rail_on_relocation
                else "the workflow's own condition re-checked after a "
                f"{SETTLE_STEPS / 30.0:.2f} s settling window, not the instant a predicate fired"
            ),
            "capture_interface": {
                "type": (
                    "break_rated_fixed_joint"
                    if args.latch_on_release and args.latch_joint_mode == "fixed"
                    else "bounded_compliant_form_lock"
                    if args.latch_on_release
                    else "friction_grapple_only"
                ),
                "engagement": "after_rail_release" if args.latch_on_release else "disabled",
                "rated_force_n": args.latch_rated_force_n if args.latch_on_release else None,
                "rated_torque_nm": args.latch_rated_torque_nm if args.latch_on_release else None,
                "position_stiffness_n_per_m": (
                    args.latch_position_stiffness_n_per_m
                    if args.latch_on_release and args.latch_joint_mode == "compliant"
                    else None
                ),
                "position_damping_ratio": (
                    args.latch_position_damping_ratio
                    if args.latch_on_release and args.latch_joint_mode == "compliant"
                    else None
                ),
                "rotation_stiffness_nm_per_rad": (
                    args.latch_rotation_stiffness_nm_per_rad
                    if args.latch_on_release and args.latch_joint_mode == "compliant"
                    else None
                ),
                "rotation_damping_ratio": (
                    args.latch_rotation_damping_ratio
                    if args.latch_on_release and args.latch_joint_mode == "compliant"
                    else None
                ),
                "reaction_wrench_on_robot_modelled": (args.latch_on_release and args.latch_joint_mode == "fixed"),
                "states": (
                    (
                        ["rigid_for_transport", "compliant_for_mating", "released_after_seating"]
                        if args.mating_mode == "compliant"
                        else ["rigid_for_transport_and_mating", "released_after_seating"]
                    )
                    if args.latch_on_release
                    else None
                ),
                "mating_mode": args.mating_mode if args.latch_on_release else None,
                "mating_compliance_n_per_m": (
                    args.latch_position_stiffness_n_per_m if args.latch_on_release else None
                ),
                "mating_compliance_nm_per_rad": (
                    args.latch_rotation_stiffness_nm_per_rad if args.latch_on_release else None
                ),
                "mating_stroke_m": MATING_TRAVEL_LIMIT_M if args.latch_on_release else None,
                "mating_force_cap_n": (
                    (args.mating_force_cap_n or args.latch_position_stiffness_n_per_m * MATING_TRAVEL_LIMIT_M)
                    if args.latch_on_release
                    else None
                ),
                "destination_channel_relief_m": args.destination_channel_relief_m,
                "load_measurement": (
                    "break_threshold_only_reaction_magnitude_not_exposed"
                    if args.latch_on_release and args.latch_joint_mode == "fixed"
                    else "commanded_external_wrench"
                    if args.latch_on_release
                    else None
                ),
            },
        }
        result["capture_interface"]["observed_per_environment"] = [
            {
                "env": index,
                "ever_engaged": bool(driver.latch_ever_engaged[index]),
                "first_engagement_episode_step": (
                    int(driver.latch_first_engagement_episode_step[index])
                    if bool(driver.latch_ever_engaged[index])
                    else None
                ),
                "softened_for_mating": bool(driver.latch_softened[index]),
                "softened_at_driver_step": (
                    int(driver.latch_softened_at[index]) if bool(driver.latch_softened[index]) else None
                ),
                "released_after_seating": bool(driver.latch_released[index]),
                "hand_opened_after_settling_verification": bool(driver.gripper_released[index]),
                "hand_opened_at_driver_step": (
                    int(driver.gripper_released_at[index]) if bool(driver.gripper_released[index]) else None
                ),
                "released_at_driver_step": (
                    int(driver.latch_released_at[index]) if bool(driver.latch_released[index]) else None
                ),
                "carriage_seek_travel_m": float(driver.latch_seek_travel_m[index]),
                "engagements_refused_out_of_seek_travel": int(driver.latch_seek_refusals[index]),
                "max_relative_position_error_m": float(driver.latch_max_position_error_m[index]),
                "max_relative_orientation_error_rad": float(driver.latch_max_orientation_error_rad[index]),
                "max_applied_force_n": (
                    None if args.latch_joint_mode == "fixed" else float(driver.latch_max_applied_force_n[index])
                ),
                "max_applied_torque_nm": (
                    None if args.latch_joint_mode == "fixed" else float(driver.latch_max_applied_torque_nm[index])
                ),
                "force_saturated": (
                    None if args.latch_joint_mode == "fixed" else bool(driver.latch_force_saturation_steps[index] > 0)
                ),
                "torque_saturated": (
                    None if args.latch_joint_mode == "fixed" else bool(driver.latch_torque_saturation_steps[index] > 0)
                ),
                "force_saturation_control_steps": (
                    None if args.latch_joint_mode == "fixed" else int(driver.latch_force_saturation_steps[index])
                ),
                "torque_saturation_control_steps": (
                    None if args.latch_joint_mode == "fixed" else int(driver.latch_torque_saturation_steps[index])
                ),
            }
            for index in range(task.num_envs)
        ]
        if args.workflow == "relocate":
            terminal_arm_joints = task.scene["robot"].data.joint_pos[:, driver.arm_joint_ids]
            terminal_blade_position, terminal_blade_orientation = attached_blade_pose_world(task)
            terminal_blade_local = terminal_blade_position - task.scene.env_origins
            terminal_staging_position_error = terminal_blade_local - driver.relocation_staging_pos.unsqueeze(0)
            terminal_staging_orientation_error = axis_angle_from_quat(
                quat_mul(
                    driver.relocation_staging_rot.unsqueeze(0).expand_as(terminal_blade_orientation),
                    quat_inv(terminal_blade_orientation),
                )
            )
            terminal_replay_targets = driver.extraction_joint_waypoints[
                driver.relocation_joint_replay_index,
                torch.arange(task.num_envs, device=task.device),
            ]
            terminal_replay_errors = (terminal_arm_joints - terminal_replay_targets).abs().amax(dim=-1)
            result["transit_planner"]["observed_per_environment"] = [
                {
                    "env": index,
                    "rail_commanded_control_steps": int(driver.rail_commanded_steps[index]),
                    "rail_terminal_drive_target_m": float(driver.rail_drive_target_m[index]),
                    "rail_goal_drive_target_m": float(driver.rail_goal_target_m[index]),
                    "stage_terminal_drive_target_xyz_m": [float(value) for value in driver.stage_drive_target_m[index]],
                    "stage_terminal_rotation_drive_target_xyz_deg": [
                        float(value) for value in driver.stage_rotation_drive_target_deg[index]
                    ],
                    "payload_stage_engaged": bool(driver.payload_stage_engaged[index]),
                    "payload_stage_control_steps": int(driver.payload_stage_control_steps[index]),
                    "payload_stage_last_staging_error_world_m": [
                        float(value) for value in driver.payload_stage_last_error_world[index]
                    ],
                    "payload_stage_last_staging_error_stage_frame_m": [
                        float(value) for value in driver.payload_stage_last_error_stage[index]
                    ],
                    "stage_retreat_done": bool(driver.relocation_stage_retreat_done[index]),
                    "stage_lateral_done": bool(driver.relocation_stage_lateral_done[index]),
                    "stage_attitude_done": bool(driver.relocation_stage_attitude_done[index]),
                    "stage_pose_staged": bool(driver.relocation_stage_translated[index]),
                    "stage_aligned": bool(driver.relocation_aligned[index]),
                    "stage_aligning": bool(driver.relocation_aligning[index]),
                    "terminal_waypoint_read": int(driver.waypoint_read[index]),
                    "terminal_driver_phase": PHASE_NAMES[int(driver.phase[index])],
                    "terminal_blade_position_xyz_m": [float(value) for value in terminal_blade_local[index]],
                    "terminal_staging_position_error_xyz_m": [
                        float(value) for value in terminal_staging_position_error[index]
                    ],
                    "terminal_staging_orientation_error_axis_angle_rad": [
                        float(value) for value in terminal_staging_orientation_error[index]
                    ],
                    "joint_replay_stop_index": int(driver.relocation_joint_replay_stop[index]),
                    "joint_replay_terminal_index": int(driver.relocation_joint_replay_index[index]),
                    "joint_replay_control_steps": int(driver.relocation_joint_replay_steps[index]),
                    "joint_replay_terminal_max_joint_error_rad": float(terminal_replay_errors[index]),
                    "joint_replay_terminal_current_joints_rad": [float(value) for value in terminal_arm_joints[index]],
                    "joint_replay_terminal_target_joints_rad": [
                        float(value) for value in terminal_replay_targets[index]
                    ],
                    "max_mount_translation_deflection_m": float(driver.rail_max_mount_deflection_m[index]),
                    "max_mount_translation_axis_m": float(driver.rail_max_mount_translation_axis_m[index]),
                    "max_mount_rotation_axis_rad": float(driver.rail_max_mount_rotation_axis_rad[index]),
                    "max_payload_stage_tracking_error_norm_m": float(driver.rail_max_mount_deflection_m[index]),
                    "max_payload_stage_tracking_error_axis_m": float(driver.rail_max_mount_translation_axis_m[index]),
                    "max_payload_stage_rotation_from_capture_axis_rad": float(
                        driver.rail_max_mount_rotation_axis_rad[index]
                    ),
                }
                for index in range(task.num_envs)
            ]
        estimator = getattr(task, "_module_state_estimator", None)
        if estimator is not None:
            terminal_pose = estimator.pose_wxyz()[0]
            terminal_occupancy = estimator.occupancy_probabilities()
            fiducial_backend = getattr(estimator, "backend", "pose_head") == "fiducial_pnp"
            result["perception"] = {
                "position_m": [float(value) for value in terminal_pose[:3]],
                "quaternion_wxyz": [float(value) for value in terminal_pose[3:7]],
                # PnP confidence is a bounded detector/geometric quality score,
                # not a learned probability. The CNN has no calibrated
                # uncertainty output, so it remains null.
                "confidence": float(estimator.confidence[0]) if fiducial_backend else None,
                "source": {
                    "deployment": ("rgb_fiducial_calibrated_pnp" if fiducial_backend else "rgb_pose_head"),
                    "oracle": "simulator_oracle_control",
                    "blind": "configured_pose_prior_control",
                }.get(estimator.mode, estimator.mode),
                "pose_error_mm": float(estimator.diagnostic_position_error_m()[0] * 1_000.0),
                "pose_error_is_privileged_simulation_diagnostic": True,
                "cnn_evaluation_count": estimator.cnn_evaluation_count,
                "reprojection_error_px": (float(estimator.reprojection_error_px[0]) if fiducial_backend else None),
                "detector_availability": (estimator.fiducial_detection_statistics if fiducial_backend else None),
                "frame": "environment_local",
                "terminal_bay_occupancy_scores": (
                    None if terminal_occupancy is None else [float(value) for value in terminal_occupancy[0]]
                ),
            }
            if args.workflow == "relocate":
                checked = bool(driver.plan_checked[0])
                result["planning"] = {
                    "request": {"source_bay": 0, "destination_bay": 1},
                    "source_occupied_destination_clear": bool(driver.plan_passed[0]) if checked else None,
                    "initial_bay_occupancy_scores": (
                        [float(value) for value in driver.initial_occupancy_scores[0]] if checked else None
                    ),
                    "decision_threshold": OCCUPANCY_PLAN_THRESHOLD,
                    "scores_are_calibrated_confidence": False,
                    "used_to_gate_execution": True,
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
            result["seated_conditions_still_held_after_settling"] = bool(driver.outcome[0])
            result["all_conditions_including_released_gripper"] = bool(driver.all_conditions[0])
            # Kept for service-parser compatibility. A shuttle workflow judges
            # the module seated after handoff; the separate all-conditions key
            # still shows that the intentionally released robot grasp is false.
            result["conditions_still_held_after_settling"] = bool(driver.outcome[0])
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
