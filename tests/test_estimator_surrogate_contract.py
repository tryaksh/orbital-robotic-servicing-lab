"""The noised skill tasks must differ from the published ones in exactly one way.

The whole value of training a skill against the estimator's error is that the
result can be read as *the estimator's cost, removed*. That reading only holds
if the noised task and the published task differ in the observation and in
nothing else -- not in the reward, not in the tolerances, not in the reset
distribution, not in the load path. A silent second difference would make the
comparison uninterpretable in exactly the way this repository has already paid
for once, when the insert skill's task disagreed with the chain it was deployed
into on eight dimensions at the same time.

Source-level on purpose: these run in CI with no GPU and no simulator, which is
this repository's rule for anything that has to keep being checked. The
alternative -- importing the config -- needs Isaac Lab and would therefore never
run in CI, which is where a drift like this actually gets caught.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap"
NOISED = TASKS / "estimator_noise_env_cfg.py"
SURROGATE = TASKS / "mdp" / "estimator_surrogate.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _class_body(source: str, name: str) -> str:
    body = source.split(f"class {name}", 1)[1]
    return body.split("\n@configclass", 1)[0]


def test_the_noised_cfgs_subclass_the_published_ones() -> None:
    """Inheritance is the mechanism that keeps the second difference from existing."""

    source = _read(NOISED)
    for noised, published in (
        ("ZeroGBladeGrapplePinGraspNoisedEnvCfg", "ZeroGBladeGrapplePinGraspEnvCfg"),
        ("ZeroGBladeGrapplePinExtractNoisedEnvCfg", "ZeroGBladeGrapplePinExtractEnvCfg"),
        ("ZeroGBladeGrapplePinInsertNoisedEnvCfg", "ZeroGBladeGrapplePinInsertEnvCfg"),
    ):
        assert re.search(rf"class {noised}\({published},", source), f"{noised} must extend {published}"


def test_the_noised_cfgs_override_the_observations_and_nothing_else() -> None:
    """Any field other than ``observations`` on these classes is a second change."""

    source = _read(NOISED)
    for name in (
        "ZeroGBladeGrapplePinGraspNoisedEnvCfg",
        "ZeroGBladeGrapplePinExtractNoisedEnvCfg",
        "ZeroGBladeGrapplePinInsertNoisedEnvCfg",
    ):
        body = _class_body(source, name)
        assignments = re.findall(r"^\s{4}(\w+)\s*:", body, flags=re.MULTILINE)
        assert assignments == ["observations"], f"{name} assigns {assignments}, expected only observations"


def test_every_module_derived_term_is_noised_and_no_other_term_is() -> None:
    """The four terms the deployed vision groups replace are exactly these four."""

    source = _read(NOISED)
    replaced = set(re.findall(r"^\s{4}(\w+) = ObsTerm\(func=mdp\.Noised", source, flags=re.MULTILINE))
    assert replaced == {"grip_error", "blade_velocity", "remaining_travel", "blade_goal_error"}

    # And the deployed contract agrees on which those are: the audit that fails
    # a vision run closed lists the same four names.
    deployment = _read(TASKS / "mdp" / "perception.py")
    required = deployment.split("_REQUIRED_DEPLOYMENT_TERMS", 1)[1].split("def audit_vision", 1)[0]
    for term in replaced:
        assert f'"{term}"' in required, f"{term} is noised here but is not a deployed replacement"


def test_the_velocity_scale_survives_the_substitution() -> None:
    """A different scale on the noised channel would be a second change."""

    published = _read(TASKS / "grapple_pin_env_cfg.py")
    noised = _read(NOISED)
    assert "blade_velocity = ObsTerm(func=mdp.attached_blade_velocity, scale=0.10)" in published
    assert "blade_velocity = ObsTerm(func=mdp.NoisedModuleVelocity, scale=0.10)" in noised


def test_the_velocity_filter_matches_the_deployed_one() -> None:
    """The surrogate must filter the way the estimator it stands in for does."""

    noised = _read(NOISED)
    deployed = _read(TASKS / "vision_two_slot_env_cfg.py")
    declared = re.search(r"DEPLOYED_VELOCITY_FILTER_TIME_CONSTANT_S = ([0-9.]+)", noised)
    shipped = re.search(r"perception_velocity_filter_time_constant_s: float = ([0-9.]+)", deployed)
    assert declared is not None and shipped is not None
    assert float(declared.group(1)) == float(shipped.group(1))


def test_the_surrogate_samples_at_the_deployed_camera_period() -> None:
    """The staircase is the model; a surrogate at the control rate would not be one."""

    surrogate = _read(SURROGATE)
    assert "from zero_g_blade_swap.servicing_camera import CAMERA_UPDATE_PERIOD_S" in surrogate
    assert "estimator_noise_camera_period_s" in _read(NOISED)


def test_the_surrogate_never_reads_live_velocity() -> None:
    """Velocity must be manufactured by differencing, not read from the simulator.

    Sampling a velocity would have been easier and would have removed the one
    property that makes this channel hard: the camera period is twice the
    control period, so a differenced estimate is zero on one step and a full
    jump on the next.
    """

    surrogate = _read(SURROGATE)
    for forbidden in ("root_lin_vel", "root_ang_vel", "root_vel_w", "attached_blade_velocity"):
        assert forbidden not in surrogate, f"the surrogate reads {forbidden}"


def test_the_surrogate_reads_its_constants_from_a_certificate() -> None:
    """No sigma is written in this code; it is inverted from published evidence."""

    surrogate = _read(SURROGATE)
    assert "EstimatorNoiseModel.from_certification" in surrogate
    certificate = re.search(r'DEFAULT_ESTIMATOR_CERTIFICATION = "([^"]+)"', surrogate)
    assert certificate is not None
    assert (ROOT / certificate.group(1)).is_file(), "the named certification must exist in evidence/"


def test_the_noised_tasks_are_registered_separately() -> None:
    """Both arms stay runnable, which is what makes the difference a measurement."""

    registry = _read(TASKS / "__init__.py")
    for task in (
        "Isaac-ZeroG-Blade-GrapplePin-GraspNoised-v0",
        "Isaac-ZeroG-Blade-GrapplePin-GraspNoised-Play-v0",
        "Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-v0",
        "Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-Play-v0",
    ):
        assert f'"{task}"' in registry
    for published in (
        "Isaac-ZeroG-Blade-GrapplePin-Grasp-v0",
        "Isaac-ZeroG-Blade-GrapplePin-Extract-v0",
    ):
        assert f'"{published}"' in registry, "the exact-state arm must not have been replaced"
