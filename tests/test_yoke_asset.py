"""Contracts on the anti-yaw yoke as an *asset*, not just as a table of numbers.

``tests/test_grapple_geometry.py`` defends the dimensions. This file defends two
things that live one layer down and that a dimensional test cannot see:

* the walls are authored only when the flag asks for them, so the plain pin
  every earlier certification was produced against is still buildable;
* the clearance and the mouth's catch are *derived* from the measured 13.5 mm
  finger half-width rather than copied next to it, so a re-measurement of the
  gripper moves them instead of silently disagreeing with them.

The second is the mistake this project has made twice: a number read off a body
origin, written into a constant, and then defended by a test that asserted the
constant against itself.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from zero_g_blade_swap import grapple_geometry
from zero_g_blade_swap.grapple_geometry import (
    GRAPPLE_PIN_WEDGE_X,
    GRAPPLE_YOKE_HALF_GAP_M,
    GRAPPLE_YOKE_MOUTH_HALF_GAP_M,
    GRAPPLE_YOKE_PARALLEL_X,
    GRAPPLE_YOKE_WALL_THICKNESS_M,
    GRAPPLE_YOKE_X,
    PAD_HALF_WIDTH_M,
    yoke_free_yaw_rad,
    yoke_lead_in_catch_m,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/zero_g_blade_swap/tasks/blade_swap/assets.py"


def _function(source: str, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_the_yoke_is_authored_only_behind_its_flag() -> None:
    """Every call to ``_define_yoke`` sits inside ``if cfg.anti_yaw_yoke``.

    The plain pin is what ``evidence/grapple_pin_axial_pull_gate.json`` and both
    railed yaw probes describe. Authoring the walls unconditionally would make
    those files unreproducible, which is the same as deleting them.
    """

    spawn = _function(ASSETS.read_text(encoding="utf-8"), "spawn_blade_with_grapple_pin")
    guarded = [
        node
        for node in ast.walk(spawn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "anti_yaw_yoke"
    ]
    assert guarded, "the yoke call is not behind an anti_yaw_yoke guard"
    calls_inside = {
        node.func.id
        for guard in guarded
        for node in ast.walk(guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_define_yoke" in calls_inside
    all_calls = [
        node
        for node in ast.walk(spawn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_define_yoke"
    ]
    assert len(all_calls) == 1, "one guarded call, or the guard proves nothing"


def test_both_walls_and_both_flares_are_authored() -> None:
    """Four prims: a yoke with one wall is a deflector, not a constraint."""

    define_yoke = _function(ASSETS.read_text(encoding="utf-8"), "_define_yoke")
    sides = {
        (node.elts[0].value, ast.literal_eval(node.elts[1]))
        for node in ast.walk(define_yoke)
        if isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and isinstance(node.elts[0], ast.Constant)
        and isinstance(node.elts[0].value, str)
    }
    assert ("Left", 1.0) in sides and ("Right", -1.0) in sides, sides
    source = ast.unparse(define_yoke)
    assert "Yoke{label}Wall" in source and "Yoke{label}Flare" in source
    # A wall alone constrains nothing on the approach: the mouth is what turns a
    # 1.5 mm slot the capture has to hit blind into a 5.1 mm catch.
    assert "_define_yoke_flare" in source


def test_a_wall_inner_face_lands_exactly_on_the_half_gap() -> None:
    """The authored box must put its *inner* face at the clearance, not its centre.

    A wall centred on ``yoke_half_gap`` would leave only half its thickness of
    clearance and eat 1.5 mm of a 1.5 mm gap. Replicated here from the same
    centre-and-size arithmetic ``_define_yoke`` uses.
    """

    for sign in (1.0, -1.0):
        centre_y = sign * (GRAPPLE_YOKE_HALF_GAP_M + 0.5 * GRAPPLE_YOKE_WALL_THICKNESS_M)
        inner_face = centre_y - sign * 0.5 * GRAPPLE_YOKE_WALL_THICKNESS_M
        assert inner_face == pytest.approx(sign * GRAPPLE_YOKE_HALF_GAP_M)
    source = ast.unparse(_function(ASSETS.read_text(encoding="utf-8"), "_define_yoke"))
    assert "cfg.yoke_half_gap + 0.5 * cfg.yoke_wall_thickness" in source


def test_the_parallel_section_is_what_spans_the_pads() -> None:
    """The walls have to be long enough to bear on a 57 mm pad's whole grip.

    The parallel run ends on the collar face, which is where a seated pad's
    leading edge is, so the engaged length is the section itself.
    """

    parallel_length = GRAPPLE_YOKE_PARALLEL_X[1] - GRAPPLE_YOKE_PARALLEL_X[0]
    assert parallel_length == pytest.approx(0.024)
    mouth_length = GRAPPLE_YOKE_PARALLEL_X[0] - GRAPPLE_YOKE_X[0]
    assert mouth_length == pytest.approx(0.010)
    assert GRAPPLE_YOKE_X[0] > GRAPPLE_PIN_WEDGE_X[0], "the yoke must not reach past the wedge's free end"


def test_clearance_and_catch_trace_to_the_measured_finger_half_width(monkeypatch) -> None:
    """Re-measure the gripper and both derived numbers must move with it.

    This is the test that would have caught the two retracted claims. It does
    not assert a value; it asserts a *dependency*, by moving the measured pad
    half-width and requiring the derived quantities to follow.
    """

    assert pytest.approx(0.0015) == GRAPPLE_YOKE_HALF_GAP_M - PAD_HALF_WIDTH_M
    assert yoke_lead_in_catch_m() == pytest.approx(GRAPPLE_YOKE_MOUTH_HALF_GAP_M - PAD_HALF_WIDTH_M)

    baseline_catch = yoke_lead_in_catch_m()
    baseline_yaw = yoke_free_yaw_rad()
    monkeypatch.setattr(grapple_geometry, "PAD_HALF_WIDTH_M", PAD_HALF_WIDTH_M + 0.001)
    assert grapple_geometry.yoke_lead_in_catch_m() == pytest.approx(baseline_catch - 0.001)
    # A wider finger in the same slot has less room to rotate, so free yaw falls.
    assert grapple_geometry.yoke_free_yaw_rad() < baseline_yaw
    importlib.reload(grapple_geometry)


def test_free_yaw_is_the_slot_formula_and_not_a_stored_number() -> None:
    """``2c / L`` for a shaft of half-width ``a`` in a slot of half-width ``a+c``."""

    clearance = GRAPPLE_YOKE_HALF_GAP_M - PAD_HALF_WIDTH_M
    length = GRAPPLE_YOKE_PARALLEL_X[1] - GRAPPLE_YOKE_PARALLEL_X[0]
    assert yoke_free_yaw_rad() == pytest.approx(2.0 * clearance / length)
    # The decision the handover records: if extract still certifies at zero with
    # the attitude improved, the fallback is a 1.0 mm-per-side gap. Check that
    # the number quoted for it is the one this formula gives.
    tightened = 2.0 * (0.0145 - PAD_HALF_WIDTH_M) / length
    assert tightened == pytest.approx(0.0833, abs=5e-4)


@pytest.mark.isaac
def test_the_yoke_prims_exist_in_the_stage_only_when_the_flag_is_on() -> None:
    """The same contract as the AST test, checked against a real USD stage.

    Marked ``isaac`` because it needs a simulator. The AST test above is the one
    that runs on every commit; this one is what proves the AST test is asking
    about something real.
    """

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    try:
        from isaaclab.sim.utils import get_current_stage
        from isaacsim.core.utils.stage import create_new_stage

        from zero_g_blade_swap.tasks.blade_swap.assets import (
            GrapplePinBladeCfg,
            spawn_blade_with_grapple_pin,
        )

        wall_paths = (
            "/World/Blade/GrapplePin/YokeLeftWall",
            "/World/Blade/GrapplePin/YokeRightWall",
            "/World/Blade/GrapplePin/YokeLeftFlare",
            "/World/Blade/GrapplePin/YokeRightFlare",
        )
        for enabled in (False, True):
            create_new_stage()
            cfg = GrapplePinBladeCfg(size=(0.45, 0.16, 0.06), anti_yaw_yoke=enabled)
            spawn_blade_with_grapple_pin("/World/Blade", cfg)
            stage = get_current_stage()
            assert stage.GetPrimAtPath("/World/Blade/GrapplePin/Wedge").IsValid()
            for path in wall_paths:
                assert stage.GetPrimAtPath(path).IsValid() is enabled, (path, enabled)
    finally:
        app_launcher.app.close()
