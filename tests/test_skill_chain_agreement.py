"""A skill must be trained on the problem the chain hands it, and this checks it.

Three policies are trained in isolation and then run inside one continuous
episode. That only means anything if each skill's task is the same problem the
chain presents at that phase. Nothing checked it, and the consequence was not
subtle: capture and extraction agreed with the chain on every dimension, and the
insert skill agreed on **none**.

What the insert skill trained on, against what the chain ran it in, before
2026-08-24:

| | Skill task | Chain's insert phase |
| --- | --- | --- |
| Bay | drew both, one of them a 505 mrad stretch | the second, robot parked opposite it |
| Vertical entry lead-in | absent | fitted |
| Channel relief | none | 4.6125 mm per side |
| Destination surfaces | production friction, 0.8/0.65 | low-friction pairing |
| Load path | pads on the pin, no lock | bounded spring-damper on the lock |
| Module at reset | one pose, 362 mm from the hand-off | at the mouth |
| Seated goal | 0.75 m in bay 1 | 0.676 m, the depth release permits |
| Axial action scale | 45 mm/s, sized for 167 mm | the same field, sized for 529 mm |

The bay's own configuration function records what the first four of those cost:
a module delivered from outside "cocked to 36 mrad, exactly the 2c/L the channel
admits, and then did not move for six thousand control steps of pushing, under
every mating variant tried". The skill was being asked for something the geometry
forbids, which is why it certified at 0.00% while holding the grip perfectly.

These are source-level assertions on purpose: they run in CI with no GPU and no
simulator, which is this repository's rule for anything that has to keep being
checked.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap"
DRIVER = ROOT / "scripts" / "run_workflow_demo.py"


def _read(*parts: str) -> str:
    return (TASKS.joinpath(*parts)).read_text(encoding="utf-8")


def _insert_task() -> str:
    """The two-bay insert task, which is the one the chain seats with."""

    source = _read("two_slot_env_cfg.py")
    return source.split("class ZeroGBladeGrapplePinInsertTwoSlotEnvCfg", 1)[1].split(
        "class ZeroGBladeGrapplePinInsertTwoSlotPlayEnvCfg", 1
    )[0]


def test_the_insert_skill_fits_the_destination_bay_the_chain_fits() -> None:
    """The lead-in, the relief and the friction, or the module cannot go in.

    ``configure_service_destination`` does all three and was called by the
    driver alone. A bay without it is one this project has already measured a
    robot-delivered module cannot enter.
    """

    assert "self.configure_service_destination()" in _insert_task()
    # And the driver still calls it, so the two paths install the same bay.
    assert "env_cfg.configure_service_destination()" in DRIVER.read_text(encoding="utf-8")


def test_the_insert_skill_parks_the_robot_where_the_rail_parks_it() -> None:
    """Reaching a bay is not the same problem as being parked opposite it.

    Solved both ways, the arm's configuration parked opposite the second bay is
    the one it has at the first to 0.0000 mrad; reaching the second from the
    first bay's base differs by 505 mrad. The chain moves the base, so a skill
    that stretches is training a pose the chain never presents.
    """

    task = _insert_task()
    assert "self.scene.robot.init_state.pos = (" in task
    assert "GRAPPLE_ROBOT_ROOT_POS[1] + SECOND_SLOT_CENTER_Y," in task
    # The mount anchor follows the base, or the authored compliance is anchored
    # to a point the robot no longer stands on.
    assert 'mount = getattr(self.scene, "mount_anchor", None)' in task


def test_the_insert_skill_declares_the_chains_mating_and_the_one_gap_left() -> None:
    """The load path is the last dimension that still differs, and it is recorded.

    The chain seats with the lock softened to a bounded spring-damper, because a
    lead-in aligns a part by pushing it and a part welded to a wrist cannot be
    pushed. The skill cannot simply switch that on: the lock's joint is authored
    between the wrist and the module at their spawn poses, and this task's reset
    writes the module anywhere along 436 mm of stroke, so PhysX snaps them
    together. Measured with the lock as the only change, same checkpoint and
    seed: 125 of 128 episodes dead inside ten control steps and 247.6 mrad of
    roll about the pin, against 0 dead and 9.4 mrad with it off.

    Closing it needs ``mdp.GrappleLatch`` to re-anchor after a reset, which is
    code rather than configuration. This test exists so the gap stays named --
    every other dimension in this file is now equal on both sides.
    """

    grapple = _read("grapple_pin_env_cfg.py")
    insert = grapple.split("class ZeroGBladeGrapplePinInsertEnvCfg", 1)[1]
    # The mating numbers the chain uses are declared, so closing the gap is a
    # one-line change rather than a rediscovery.
    assert 'latch_joint_mode: str = "compliant"' in insert
    assert "latch_position_stiffness_n_per_m: float = 40_000.0" in insert
    assert "latch_rotation_stiffness_nm_per_rad: float = 20_000.0" in insert
    assert "mating_force_cap_n: float = 1_000.0" in insert
    assert "service_destination_channel_relief_m: float = 0.0046125" in insert
    # And the measurement that says why it is off travels with it.
    assert "125 of 128 episodes dead inside ten control steps" in insert
    assert "latch_enabled: bool = False" in insert


def test_the_insert_skill_starts_where_the_chain_hands_over() -> None:
    """One pose is not a distribution, and it was the wrong pose.

    The chain hands the seating phase the module at the mouth. The reset placed
    it 362 mm deeper, so every state the chain produced was outside the
    distribution the policy trained on.
    """

    grapple = _read("grapple_pin_env_cfg.py")
    assert "class InsertStrokeEventsCfg" in grapple
    assert "func=mdp.reset_grapple_insert_stroke," in grapple
    insert = grapple.split("class ZeroGBladeGrapplePinInsertEnvCfg", 1)[1]
    assert "self.events = InsertStrokeEventsCfg()" in insert

    bank = _read("insert_reset_bank.py")
    assert "INSERT_STROKE_ARM_JOINT_POS" in bank
    assert "do not hand-edit" in bank


def test_the_insert_skill_seats_where_the_interface_allows_release() -> None:
    """Bay 1's goal was 74 mm past the depth this interface may let go at.

    ``SERVICE_DESTINATION_SEATED_X`` is derived from the latch's own geometry:
    past that depth an engaged jaw enters the slot mouth. A goal beyond it is a
    goal no robot on this interface can deliver, and it is six times the
    insertion's own 12 mm axial tolerance away.
    """

    two_slot = _read("two_slot_env_cfg.py")
    assert "class DestinationBayCommandsCfg" in two_slot
    assert "goal_pos=SECOND_SLOT_INSERTED_POS" in two_slot
    assert "commands: DestinationBayCommandsCfg = DestinationBayCommandsCfg()" in _insert_task()

    assets = _read("assets.py")
    for name in ("FIRST_SLOT_INSERTED_POS", "SECOND_SLOT_INSERTED_POS"):
        block = assets.split(f"{name} = (", 1)[1].split(")", 1)[0]
        assert "SERVICE_DESTINATION_SEATED_X" in block, name


def test_the_chain_reads_each_skills_own_scale_and_clock() -> None:
    """The chain must not restate a skill's action scale or budget.

    Both are read off the task configuration, so a skill and the phase that runs
    it cannot disagree about how fast it may move or how long it has. This is
    the mechanism that made the action-scale correction propagate for free.
    """

    workflow = _read("workflow_demo_env_cfg.py")
    assert "INSERT_ACTION_SCALE = _certified_action_scale(ZeroGBladeGrapplePinInsertEnvCfg)" in workflow
    assert "def certified_episode_length_s" in workflow
    driver = DRIVER.read_text(encoding="utf-8")
    assert "INSERT_ACTION_SCALE," in driver


def test_extraction_agrees_with_the_chain_and_is_the_reason_it_works() -> None:
    """The control for all of the above.

    Extraction's latch is off in the task, and off in the chain too for the
    whole pull: the chain arms the lock only once the driver says the module is
    clear of the rails, which is the moment extraction ends. Capture is the
    same. That agreement is why those two skills transfer and the insertion did
    not, and it should break loudly if either side moves.
    """

    grapple = _read("grapple_pin_env_cfg.py")
    assert "latch_enabled: bool = False" in grapple
    extract = grapple.split("class ZeroGBladeGrapplePinExtractEnvCfg", 1)[1].split("@configclass", 1)[0]
    assert "latch_enabled" not in extract

    driver = DRIVER.read_text(encoding="utf-8")
    assert "env_cfg.latch_engages_on_release = True" in driver
    assert 'self.events.grapple_latch.params["require_armed"] = self.latch_engages_on_release' in grapple


def test_the_insert_objective_is_scaled_to_the_channel_it_must_enter() -> None:
    """A skill's *objective* has to describe the chain's problem, not just its scene.

    Seven of the eight dimensions above are about the task's geometry and
    physics. This is the ninth and it is about the reward, because a task can
    present the right scene and still ask for the wrong thing.

    ``insertion_misalignment_penalty`` normalises orientation error, and its
    default of 0.15 rad is the *seated* tolerance -- what a module must satisfy
    once the channel is already holding it square. Entry is the binding
    constraint and it is four times tighter: a rigid part of length ``L``
    entering a channel with ``c`` of relief per side fits only while its tilt
    stays under ``2c/L``, which for the shipped relief is
    ``SERVICE_DELIVERED_ATTITUDE_RAD`` = 20.5 mrad.

    Measured at the default, over 512 held-out episodes on a policy trained to
    convergence: orientation ends at a median of 84.6 mrad with a 5th percentile
    of 56.1. Not one episode in five hundred ended inside the angle at which the
    module could enter at all, and the objective charged 0.08 a step for it --
    less than the 0.50 the same episode paid for 7.1 mm of lateral. The skill
    was being paid to prefer a survivable offset over a fatal attitude.

    So the number has to come from the rack rather than be chosen, and it has to
    keep coming from the rack. ``evidence/insert_attitude_diagnosis.json`` holds
    the measurement.
    """

    grapple = _read("grapple_pin_env_cfg.py")
    insert_rewards = grapple.split("class InsertRewardsCfg", 1)[1].split("class InsertTerminations", 1)[0]
    assert '"orientation_scale_rad": SERVICE_DELIVERED_ATTITUDE_RAD' in insert_rewards, (
        "The insert objective must scale orientation by the destination channel's own "
        "admittance. A literal here is a tuned constant standing in for a derived one."
    )
    assert "SERVICE_DELIVERED_ATTITUDE_RAD," in grapple, "the constant must be imported, not redefined"

    # The default stays put so every task that already quotes a number under it
    # is bit-identical. Changing it would silently move published evidence.
    insertion = (TASKS / "mdp" / "insertion.py").read_text(encoding="utf-8")
    assert "orientation_scale_rad: float = 0.15" in insertion, (
        "The default is what every previously published insertion number was measured under."
    )


def test_the_channel_admittance_is_derived_from_the_relief_it_comes_from() -> None:
    """``2c/L`` is an identity here, not a coincidence, and it should stay one.

    ``SERVICE_DESTINATION_CHANNEL_RELIEF_M`` is defined as
    ``0.5 * BLADE_LENGTH_M * SERVICE_DELIVERED_ATTITUDE_RAD``, so inverting it
    gives ``2c/L`` back exactly. That is what makes it legitimate to use the
    delivered-attitude constant as the entry limit in the reward: the two are
    the same quantity written from opposite ends.
    """

    assets = _read("assets.py")
    assert (
        "SERVICE_DESTINATION_CHANNEL_RELIEF_M = 0.5 * BLADE_LENGTH_M * SERVICE_DELIVERED_ATTITUDE_RAD"
        in assets
    ), "the relief must stay derived from the delivered attitude, or 2c/L stops being an identity"
