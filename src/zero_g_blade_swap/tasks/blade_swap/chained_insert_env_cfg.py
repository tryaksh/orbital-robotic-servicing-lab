"""Insert, trained inside the chain.

The installation chain scores 84.38% while the insert skill it uses certifies at
95.57% alone, and the whole of that gap is the insert phase: measured in the
chain it runs at about 80%. Four attempts to close it by reconstructing the
hand-off as a *reset distribution* have been built, gated and refuted:

    per-joint noise box                       0.00%
    measured arm poses, module left nominal  26.32%
    measured arm and module poses, paired    47.17%
    the real chain's insert phase             ~80%

The third of those is the interesting one. Pairing the arm and the module
recovered half the remaining gap and stopped there, and what is left is **not an
initial condition at all**: the chained driver latches the holding closure at
hand-over (``TwoStageRobotiqAction.hold_latch``), so the module is carried under
a different gripper controller than any training task provides. A reset cannot
express a controller state, and a pose cannot express a trajectory.

So this task stops approximating the hand-off and runs it. Every episode:

1. resets the *capture* scene -- the grasp skill's own wide reset, at the
   installation chain's stage;
2. steps the **frozen** capture policy until the chain's hand-off predicate
   fires -- a loaded grip inside ``WORKFLOW_HANDOVER_GRIP_M`` held for
   ``HANDOVER_HOLD_S``;
3. latches ``hold_latch`` and holds still for ``SEAT_STEPS`` while the closure
   drives the pin against its collar;
4. hands the arm to the policy being trained, at the insert skill's own action
   scale, reward set, terminations and clock.

Steps 1 to 3 are a **prologue**: they carry no reward and no insert termination,
because the learning policy did not act in them. They are still real physics on
the real scene, which is the entire point -- the state the policy takes over in
is produced rather than sampled.

Nothing here modifies the single-slot skills. ``ZeroGBladeGrapplePinInsertEnvCfg``
is untouched and insert v6's 95.57% still describes the task in that file.

Two properties of this construction are worth stating because they bound what it
can claim:

*The prologue is uncontrollable, and about one episode in twenty ends in it.*
A capture that overruns its own 10 s budget or shoves the module ends the episode
with exactly zero return, which the policy could not have prevented. That is
variance, not bias: the prologue reward is zero for every episode, so no action
the policy takes is ever credited or charged for a capture outcome.

*The clock is not restarted at hand-over.* ``episode_length_buf`` keeps counting
through the prologue, so the arm action term's settling gate is spent on the
capture exactly as it is in the chain, and the insert phase is deadlined from the
step it begins on. The consequence is that ``control_steps`` in this task's
terminal metrics includes the prologue and is therefore not comparable with the
insert skill's own cycle time.
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: F401  (documented group base)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .assets import CONTACT_INSERTION_STAGE_BLADE_POSE, GRAPPLE_HEAD_ON_ARM_JOINT_POS
from .grapple_pin_env_cfg import (
    GraspActionsCfg,
    GraspEventsCfg,
    InsertRewardsCfg,
    SingleStageCurriculumCfg,
    ZeroGBladeGrapplePinCaptureEnvCfg,
    ZeroGBladeGrapplePinGraspEnvCfg,
)
from .mdp.grapple import (
    WORKFLOW_HANDOVER_GRIP_M,
    capture_established,
    extraction_failure,
    grapple_grip_error_metrics,
    grapple_insertion_success_mask,
    grip_drive_torque,
    grip_finger_angle,
)
from .mdp.insertion import attached_blade_pose_world, attached_blade_velocity
from .robust_insertion_env_cfg import configure_insertion_play_presentation
from .terminal_metrics_env import TerminalMetricsManagerBasedRLEnv
from .workflow_demo_env_cfg import (
    CAPTURE_BUDGET_S,
    GRASP_ACTION_SCALE,
    HANDOVER_HOLD_S,
    INSERT_ACTION_SCALE,
    INSERT_BUDGET_S,
    SEAT_STEPS,
    WorkflowGraspObsCfg,
    WorkflowInsertObsCfg,
)

#: Phases, as integers, because they are driven per environment in parallel.
#: Deliberately the chain's own three install phases with the same names; the
#: transit and extract phases belong to removal and are not in this task.
CAPTURE, SEAT, INSERT = range(3)

#: The installation chain starts with a module presented at the rack mouth,
#: which is curriculum stage 2 -- ``certify_workflow.sh install`` passes
#: ``--curriculum_stage 2``. This task trains on exactly that, so its reset
#: carries one entry and it is the stage-2 one.
INSTALL_STAGE = 2

#: The capture policy this task runs. Promoted capture v5. Overridable through
#: the environment so a retrained capture can be gated without editing a task
#: file, which is how every checkpoint choice in this repository is made.
DEFAULT_CAPTURE_CHECKPOINT = (
    "logs/rl_games/zero_g_blade_insertion_contact/grapple_grasp_l0_seed70_v5/nn/"
    "last_zero_g_blade_insertion_contact_ep_1500_rew__35.348194_.pth"
)


def _steps(seconds: float, env) -> int:
    return max(1, int(round(seconds / float(env.step_dt))))


#: One row per seat-to-insert hand-off, written when ``INSERT_CHAIN_TRACE`` names
#: a path. Deliberately the subset of ``HANDOFF_TRACE_FIELDS`` that describes the
#: state the insert policy takes over in, in the same units and the same order,
#: so a trace from this task and one from ``run_workflow_demo.py --handoff_trace``
#: can be compared column by column. Rule: before believing a reconstructed
#: distribution, measure it against the real one.
INSERT_CHAIN_TRACE_FIELDS = (
    "step",
    "env",
    "grip_error_m",
    "grip_attitude_rad",
    "finger_angle_rad",
    "drive_torque_nm",
    "blade_x_m",
    "blade_y_m",
    "blade_z_m",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
    "arm_joint_0",
    "arm_joint_1",
    "arm_joint_2",
    "arm_joint_3",
    "arm_joint_4",
    "arm_joint_5",
)


# ---------------------------------------------------------------------------
# Phase-gated terminations.
#
# Every one of the insert skill's predicates would fire during the prologue for
# reasons the learning policy had nothing to do with: ``extraction_failure``
# reads a grip error above 30 mm as a lost grip, and an approach starts at 85 mm.
# Gating them on the phase is what lets the prologue exist at all.
#
# The *names* of these terms are the insert skill's, because
# ``TERMINATION_REASONS`` is keyed by name and the evaluator refuses a term it
# cannot categorise.


def chained_insertion_success(env) -> torch.Tensor:
    return grapple_insertion_success_mask(env) & (env.chain_phase == INSERT)


def chained_extraction_failure(env) -> torch.Tensor:
    return extraction_failure(env) & (env.chain_phase == INSERT)


# ``capture_failure`` is deliberately **not** a termination here, and the first
# version of this task had it. The reasoning was that a capture which shoves the
# module cannot recover, so ending the episode saves compute. Gated and refuted
# before any GPU was spent on training, which is what the gate is for:
#
#   insert v6 on this task, with capture_failed     69.27%  (52/192 died capturing)
#   insert v6 on this task, without it              (below)
#   the real install chain, same seed, same policy  88.54% predicate fired
#
# The chain's own numbers say why. ``WorkflowTerminationsCfg`` carries no capture
# failure term at all, and measured over the same 192 episodes at seed 4070 the
# chain overruns its capture phase **once**. A capture that transiently shoves
# the module past the 60 mm ``blade_shoved`` limit keeps going in the chain, and
# the median episode this task was killing had a 12.2 mm grip -- it was
# recovering. Adding a predicate the chain does not have made the training
# distribution not the chain's, which is the exact defect this task exists to
# fix. The capture phase is bounded by its own certified clock and nothing else.


def chained_time_out(env) -> torch.Tensor:
    """Two deadlines, one per phase, each read from the skill that owns it.

    The capture gets the grasp task's certified episode length and the insert
    phase gets the insert task's, measured from the step it takes over on. This
    is the same reconciliation ``PHASE_BUDGET_S`` performs in the chain: a skill
    that is given a longer clock here than its certification grants would make
    the training task easier than the workflow it is being trained for.
    """

    steps = env.episode_length_buf
    return ((env.chain_phase == CAPTURE) & (steps >= env.chain_capture_deadline)) | (
        (env.chain_phase == INSERT) & ((steps - env.chain_insert_started) >= env.chain_insert_deadline)
    )


@configclass
class ChainedInsertObservationsCfg:
    """The learning policy's input, and the frozen capture policy's.

    ``policy`` is exactly ``WorkflowInsertObsCfg`` -- the group the chain already
    feeds insert v6 -- so a checkpoint can be resumed into this task without an
    observation dimension change. ``grasp`` is exactly what the chain feeds the
    capture policy. Both are computed every step; that costs a few tensor
    operations and removes any chance of handing a policy an observation
    assembled in the wrong order.
    """

    policy: WorkflowInsertObsCfg = WorkflowInsertObsCfg()
    grasp: WorkflowGraspObsCfg = WorkflowGraspObsCfg()


@configclass
class ChainedInsertTerminationsCfg:
    """The insert skill's termination set, gated on the phase.

    Exactly the chain's: the two learned predicates apply only where the learning
    policy is driving, and the capture is bounded by its own clock. See the note
    above ``chained_time_out`` for why no capture-failure term is here.

    ``non_finite`` is deliberately *not* gated: a diverging simulation is a
    diverging simulation whichever phase it happens in.
    """

    time_out = DoneTerm(func=chained_time_out, time_out=True)
    insertion_success = DoneTerm(func=chained_insertion_success)
    extraction_failed = DoneTerm(func=chained_extraction_failure)
    non_finite = DoneTerm(func=mdp.insertion_non_finite_state)


@configclass
class ZeroGBladeGrapplePinInsertChainEnvCfg(ZeroGBladeGrapplePinCaptureEnvCfg):
    """Fine-tune the insert skill on the states a real capture hands it."""

    observations: ChainedInsertObservationsCfg = ChainedInsertObservationsCfg()
    # The grasp task's action set, because the capture policy commands the
    # gripper and the chain's action term is the one that carries hold_latch.
    # The learning policy still sees six actions; see ChainedInsertEnv.
    actions: GraspActionsCfg = GraspActionsCfg()
    events: GraspEventsCfg = GraspEventsCfg()
    # Unchanged from the insert skill. The point of this task is the state the
    # policy starts in, not a new objective.
    rewards: InsertRewardsCfg = InsertRewardsCfg()
    terminations: ChainedInsertTerminationsCfg = ChainedInsertTerminationsCfg()
    curriculum: SingleStageCurriculumCfg = SingleStageCurriculumCfg()
    #: Long enough for both phases at their own budgets, so the built-in episode
    #: bound never fires before ``chained_time_out`` does.
    episode_length_s: float = CAPTURE_BUDGET_S + SEAT_STEPS / 30.0 + INSERT_BUDGET_S
    #: The frozen capture policy. Read from the environment if set, so a
    #: retrained capture can be gated without editing this file.
    capture_checkpoint: str = os.environ.get("CAPTURE_CHECKPOINT", DEFAULT_CAPTURE_CHECKPOINT)

    def configure_robustness(self, level: int) -> None:
        super().configure_robustness(level)
        # The grasp task's reset, because the capture in this task is a real
        # approach and has to be the one capture v5 was certified on.
        self.events = GraspEventsCfg()
        # One entry, and it is the installation chain's stage. The curriculum is
        # single-stage, so index 0 is the only one drawn, and writing the stage-2
        # values into it is how "always stage 2" is expressed without inventing a
        # second way to select a stage.
        grasp_defaults = ZeroGBladeGrapplePinGraspEnvCfg()
        grasp_defaults.configure_robustness(level)
        self.events.reset_arm.params["noise_by_stage"] = (
            grasp_defaults.events.reset_arm.params["noise_by_stage"][INSTALL_STAGE],
        )
        self.events.reset_arm.params["poses_by_stage"] = (GRAPPLE_HEAD_ON_ARM_JOINT_POS[INSTALL_STAGE],)
        self.events.reset_blade.params["poses_by_stage"] = (CONTACT_INSERTION_STAGE_BLADE_POSE[INSTALL_STAGE],)
        if level < 2:
            self.events.blade_mass = None
        if level < 3:
            self.events.slot_material = None
            self.events.left_guide_material = None
            self.events.right_guide_material = None
            self.events.randomize_stiction = None
            self.events.rail_stiction_force = None
        if level < 4:
            self.events.clear_mount_wrench = None
            self.events.base_wobble = None
        self._configure_latch()


@configclass
class ChainedInsertAttitudeRewardsCfg(InsertRewardsCfg):
    """The insert objective with one number changed: the attitude weighting.

    Measured on the pre-training gate, 6 of 7 insert-phase overruns end outside
    `grasp_orientation` and 6 of 7 outside `axial_depth`, together, while
    lateral, orientation and both velocity conditions never fail. Successful
    insertions terminate at 0.1927 rad of grip attitude against a 0.20 rad
    tolerance -- the skill lives on that limit.

    What the objective charges for it, in weighted units, at exactly the 0.20 rad
    boundary::

        0.25 * ((0.20 - 0.08) / 0.15)^2 * 0.50 = 0.08

    against `elapsed_time_penalty`'s 0.10. The policy is charged more for taking
    one more control step than for sitting at the attitude that ends the episode.

    **Only ``orientation_weight`` moves, 0.25 -> 1.0.** Everything else is the
    insert task's default, deliberately:

    * ``free_rad`` stays 0.08, because the aim is not to charge for attitudes
      that already succeed;
    * ``orientation_scale`` stays 0.15 rather than taking extraction's 0.06,
      because extraction's parameters were sized for a failure that runs to
      0.35 rad and this one lives at 0.19;
    * ``max_penalty`` stays 25.0, and that is checked rather than assumed -- at
      the 0.35 rad extraction-failure limit the raw cost reaches about 3.2, so
      nothing an episode can visit is saturated. Unlike extraction, the defect
      here is a weight and not a clamp.

    Sized against the measurement rather than guessed: this makes the attitude
    term about four times its current size, roughly 3.8 of episode reward against
    8.2 for progress and 30 for success. Extract v7 is the reason it is not
    larger -- an over-weighted attitude term made standing still cheaper than
    working and cost the removal chain 11 points.
    """

    retention = RewTerm(
        func=mdp.grip_retention_penalty,
        weight=-0.50,
        params={
            "free_m": 0.004,
            "free_rad": 0.08,
            "orientation_scale": 0.15,
            "orientation_weight": 1.0,
            "max_penalty": 25.0,
        },
    )


@configclass
class ZeroGBladeGrapplePinInsertChainAttitudeEnvCfg(ZeroGBladeGrapplePinInsertChainEnvCfg):
    """The chained-insert task with the attitude term reweighted.

    A separate registration so the run trained against the unchanged objective
    stays reproducible and the two remain one change apart. The single-slot
    insert task keeps its own defaults either way, so insert v6's certification
    still describes the task in the file.
    """

    rewards: ChainedInsertAttitudeRewardsCfg = ChainedInsertAttitudeRewardsCfg()


@configclass
class ZeroGBladeGrapplePinInsertChainPlayEnvCfg(ZeroGBladeGrapplePinInsertChainEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


@configclass
class ZeroGBladeGrapplePinInsertChainAttitudePlayEnvCfg(ZeroGBladeGrapplePinInsertChainAttitudeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        configure_insertion_play_presentation(self)


class ChainedInsertEnv(TerminalMetricsManagerBasedRLEnv):
    """The chain's install phase machine, run inside the environment.

    Lifted from ``WorkflowDriver`` in ``scripts/run_workflow_demo.py`` rather
    than rewritten: same hand-off predicate, same seat length, same latch, same
    per-phase action scales, all read from the same constants the driver reads.
    What is different is only where it lives -- inside ``step`` instead of around
    it -- so that the policy being trained receives the hand-off as its initial
    state.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        # Before ``super().__init__``, because the termination terms read these
        # and the managers are built inside it.
        count = int(cfg.scene.num_envs)
        device = cfg.sim.device
        self.chain_phase = torch.full((count,), CAPTURE, dtype=torch.long, device=device)
        self.chain_held = torch.zeros(count, dtype=torch.long, device=device)
        self.chain_seat_until = torch.zeros(count, dtype=torch.long, device=device)
        self.chain_insert_started = torch.zeros(count, dtype=torch.long, device=device)
        self.chain_capture_deadline = 0
        self.chain_insert_deadline = 0
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.chain_capture_deadline = _steps(CAPTURE_BUDGET_S, self)
        self.chain_insert_deadline = _steps(INSERT_BUDGET_S, self)
        self._chain_required_hold = _steps(HANDOVER_HOLD_S, self)
        self._chain_arm = self.action_manager.get_term("arm")
        self._chain_gripper = self.action_manager.get_term("gripper")
        # Per phase, in phase order. Read from the tasks the two policies were
        # certified on; a policy driven at another skill's scale is a different
        # policy, and this project has measured that as a full certification.
        self._chain_scales = torch.tensor(
            [GRASP_ACTION_SCALE, GRASP_ACTION_SCALE, INSERT_ACTION_SCALE],
            device=self.device,
        )
        self._chain_action_dim = self.action_manager.total_action_dim
        # The action term's own joint ids, so a trace records the joints the
        # reset writes rather than the first six of whatever order the scene has.
        arm_joint_ids = getattr(self._chain_arm, "_joint_ids", None)
        self._chain_arm_joint_ids = list(range(6)) if arm_joint_ids is None else list(arm_joint_ids)
        self._chain_trace_path = os.environ.get("INSERT_CHAIN_TRACE") or None
        self._chain_trace_rows: list = []

        checkpoint = Path(self.cfg.capture_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"The chained-insert task needs a frozen capture policy; {checkpoint} does not exist. "
                "Set CAPTURE_CHECKPOINT to the promoted capture checkpoint."
            )
        # Imported here rather than at module import time: the loader pulls in
        # torch only, but keeping it local documents that the task configuration
        # is importable on a machine with no checkpoints at all.
        from zero_g_blade_swap.checkpoint_policy import CheckpointPolicy

        self._chain_capture_policy = CheckpointPolicy(checkpoint, self.device)
        print(
            f"[INFO] Chained insert: frozen capture policy {checkpoint.name} "
            f"(epoch {self._chain_capture_policy.epoch}, sha256 {self._chain_capture_policy.sha256[:16]})",
            flush=True,
        )

        # The learning policy commands the arm only, six values, exactly as the
        # insert skill does. The gripper is driven by the phase machine, so a
        # checkpoint trained on the insert task resumes here without an action
        # dimension change -- which is a standing rule in this repository.
        self.single_action_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(6,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    # -- phase machine ------------------------------------------------------

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        self.chain_phase[env_ids] = CAPTURE
        self.chain_held[env_ids] = 0
        self.chain_seat_until[env_ids] = 0
        self.chain_insert_started[env_ids] = 0

    def _advance_phase(self) -> None:
        """Resolve every hand-off due on the state the previous step produced."""

        capturing = self.chain_phase == CAPTURE
        if bool(capturing.any()):
            # The chain's hand-off condition, unchanged: hand over on the *next*
            # skill's precondition rather than on the capture's own success
            # criterion. capture_established alone accepts 20 mm, and handing
            # over there puts the receiving policy 10 mm out of distribution.
            grip_error, _ = grapple_grip_error_metrics(self)
            qualifying = capturing & capture_established(self) & (grip_error <= WORKFLOW_HANDOVER_GRIP_M)
            self.chain_held = torch.where(
                qualifying,
                self.chain_held + 1,
                torch.where(capturing, torch.zeros_like(self.chain_held), self.chain_held),
            )
            promote = capturing & (self.chain_held >= self._chain_required_hold)
            if bool(promote.any()):
                # Latch the holding closure, per environment, exactly as the
                # driver does. This is the part of the hand-off no reset
                # distribution can express, and it is why this task exists.
                self._chain_gripper.hold_latch[promote] = True
                self.chain_phase[promote] = SEAT
                self.chain_seat_until[promote] = self.episode_length_buf[promote] + SEAT_STEPS

        seated = (self.chain_phase == SEAT) & (self.episode_length_buf >= self.chain_seat_until)
        if bool(seated.any()):
            if self._chain_trace_path is not None:
                self._record_handoff(seated)
            self.chain_phase[seated] = INSERT
            self.chain_insert_started[seated] = self.episode_length_buf[seated]
            # The insert phase is the episode as far as the objective is
            # concerned, so the per-term episode sums start here. Without this
            # the tfevents under summaries/ would mix a prologue the policy did
            # not act in into every reward term, and per-term diagnosis is how
            # most of this task's defects were found.
            self.reward_manager.reset(torch.nonzero(seated, as_tuple=False).squeeze(-1))

    def _record_handoff(self, mask: torch.Tensor) -> None:
        """Record the state the insert policy is about to take over in."""

        grip_error, grip_attitude = grapple_grip_error_metrics(self)
        blade_position, _ = attached_blade_pose_world(self)
        blade_local = blade_position - self.scene.env_origins
        velocity = attached_blade_velocity(self)
        joints = self.scene["robot"].data.joint_pos[:, self._chain_arm_joint_ids]
        columns = (
            self.episode_length_buf.to(torch.float64),
            torch.arange(self.num_envs, device=self.device, dtype=torch.float64),
            grip_error.to(torch.float64),
            grip_attitude.to(torch.float64),
            grip_finger_angle(self).to(torch.float64),
            grip_drive_torque(self).to(torch.float64),
            blade_local[:, 0].to(torch.float64),
            blade_local[:, 1].to(torch.float64),
            blade_local[:, 2].to(torch.float64),
            torch.linalg.vector_norm(velocity[:, :3], dim=-1).to(torch.float64),
            torch.linalg.vector_norm(velocity[:, 3:], dim=-1).to(torch.float64),
            *(joints[:, index].to(torch.float64) for index in range(joints.shape[1])),
        )
        rows = torch.stack(columns, dim=-1)
        self._chain_trace_rows.append(rows[mask].cpu().numpy())

    def close(self) -> None:
        if self._chain_trace_path is not None and self._chain_trace_rows:
            import numpy as np

            path = Path(self._chain_trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                handoff=np.concatenate(self._chain_trace_rows),
                handoff_fields=np.asarray(INSERT_CHAIN_TRACE_FIELDS),
            )
            print(f"[INFO] Chained insert: wrote {path}", flush=True)
            self._chain_trace_rows = []
        super().close()

    def _compose_action(self, action: torch.Tensor) -> torch.Tensor:
        """Build the full arm-and-gripper command for the phase each env is in."""

        command = torch.zeros((self.num_envs, self._chain_action_dim), device=self.device)
        capturing = self.chain_phase == CAPTURE
        if bool(capturing.any()):
            captured = self._chain_capture_policy.act(self.obs_buf["grasp"])
            command[capturing] = captured[capturing]
        acting = ~capturing
        if bool(acting.any()):
            # ``--smoke`` passes the action manager's full width; PPO passes the
            # six the learning policy owns. Accept both rather than make the
            # smoke contract a special case.
            arm = action[:, :6].to(self.device)
            command[:, :6] = torch.where(acting.unsqueeze(-1), arm, command[:, :6])
            # The seat is scripted: hold the arm still and let the closure drive
            # the pin onto the collar. The module is still in its rails here, so
            # the rails absorb the wedge thrust that makes an idle pause on a
            # free module catastrophic.
            command[:, :6] = torch.where((self.chain_phase == SEAT).unsqueeze(-1), 0.0, command[:, :6])
            # Everything past the capture keeps commanding closure, so the
            # two-stage action term holds the pin rather than relaxing to the
            # capture command.
            command[:, 6] = torch.where(acting, torch.ones_like(command[:, 6]), command[:, 6])
        return command

    def step(self, action: torch.Tensor):
        self._advance_phase()
        prologue = self.chain_phase != INSERT
        command = self._compose_action(action)
        self._chain_arm._scale[:] = self._chain_scales[self.chain_phase]
        observations, reward, terminated, truncated, extras = super().step(command)
        # The learning policy did not act in the prologue, so it is neither paid
        # nor charged for it. Every episode carries the same zero there, so this
        # adds variance to the return and no bias to the gradient.
        reward = torch.where(prologue, torch.zeros_like(reward), reward)
        self.reward_buf = reward
        return observations, reward, terminated, truncated, extras


__all__ = [
    "CAPTURE",
    "DEFAULT_CAPTURE_CHECKPOINT",
    "INSERT",
    "INSTALL_STAGE",
    "SEAT",
    "ChainedInsertAttitudeRewardsCfg",
    "ChainedInsertEnv",
    "ChainedInsertObservationsCfg",
    "ChainedInsertTerminationsCfg",
    "ZeroGBladeGrapplePinInsertChainAttitudeEnvCfg",
    "ZeroGBladeGrapplePinInsertChainAttitudePlayEnvCfg",
    "ZeroGBladeGrapplePinInsertChainEnvCfg",
    "ZeroGBladeGrapplePinInsertChainPlayEnvCfg",
]
