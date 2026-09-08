import copy

import pytest

from assembly_recovery.retry_controller import ActorRetryController, RetrySettings
from assembly_recovery.reward_ledger import discounted_job_return
from scripts.compare_recovery import recovery_witness


def observation(height=0.032, load=7.0):
    return [0.0, 0.0, height, 0.0, 1.0, 0.0, 0.0] + [0.0] * 6 + [0.0, 0.0, load, 8.0] + [0.0] * 6 + [-1.0]


def controller(retry=True):
    return ActorRetryController(observation(), step_dt=1 / 15, position_bounds=[0.05] * 3,
                                seated_height_m=0.007392, retry=retry)


def test_actor_only_pair_has_identical_actions_until_noisy_stall_trigger():
    recovery, continued = controller(), controller(False)
    for step in range(60):
        assert recovery.act(observation(), step / 15)[0] == continued.act(observation(), step / 15)[0]
    recovered_action, recovered_state = recovery.act(observation(), 4.0)
    continued_action, continued_state = continued.act(observation(), 4.0)
    assert recovery.trigger == continued.trigger
    assert recovered_state["phase"] == "withdraw"
    assert continued_state["phase"] == "insert"
    assert recovered_action[2] > continued_action[2]


def test_noisy_free_space_does_not_trigger_recovery():
    policy = controller()
    for step in range(150):
        policy.act(observation(load=1.0), step / 15)
    assert policy.trigger is None


def test_withdrawal_requires_measured_rise_before_search():
    policy = controller()
    for step in range(120):
        _, state = policy.act(observation(), step / 15)
    assert state["phase"] == "withdraw"
    for step in range(120, 125):
        action, state = policy.act(observation(height=0.045, load=1.0), step / 15)
    assert state["phase"] == "search"
    assert all(-1 <= x <= 1 for x in action)


def test_oscillation_cannot_renew_insertion_progress():
    policy = controller()
    for step in range(100):
        policy.act(observation(height=0.032 + (0.002 if step % 20 < 10 else 0)), step / 15)
    assert policy.trigger is not None


def test_reward_cutoff_excludes_post_terminal_success_and_partial_interval():
    rows = [{"step": i + 1, "reward": [r], "reward_terms": {"base": [r]}} for i, r in enumerate([1., 2., 1000.])]
    result = discounted_job_return(rows, 0, terminal_physics_step=17, decimation=8, gamma=0.5)
    assert result["discounted_return"] == 2.0
    assert result["included_control_endpoints"] == 2
    assert result["terminal_partial_physics_steps_without_endpoint_reward"] == 1
    endpoint = discounted_job_return(rows, 0, terminal_physics_step=24, decimation=8, gamma=0.5)
    assert endpoint["discounted_return"] == 252.0
    corrupt = copy.deepcopy(rows)
    corrupt[0]["reward_terms"]["base"] = [0.0]
    with pytest.raises(ValueError, match="components"):
        discounted_job_return(corrupt, 0, 17, 8)
    with pytest.raises(ValueError, match="consecutive"):
        discounted_job_return(rows[1:], 0, 17, 8)


@pytest.mark.parametrize("missing", [None, "contact", "withdrawal", "rise", "pretrigger_stall"])
def test_recovery_claim_requires_separate_contact_and_withdrawal_witnesses(missing):
    report = {"jobs": [{"job_id": "test", "first_stall_step": 100, "success": True,
                        "recovered_after_stall": True, "outcome": "success", "forbidden_events": []}],
              "criteria": {"physics_dt": 0.01, "stall_window_s": 1.0},
              "controllers": [{"trigger": {"time_s": 1.2}}],
              "initial_fixed_pos": [[0.0, 0.0, 0.0]], "geometry": {"hole_height_m": 0.025}}
    controls = [{"step": i, "phase": ["insert" if i <= 15 else "withdraw"],
                 "active_before_step": [True], "part_pos": [[0.0, 0.0, 0.025 if i <= 15 else 0.035]]}
                for i in range(1, 36)]
    physics = {"test": [{"step": i, "peg_fixture_contact_force_n": 7.0 if i <= 120 else 0.0}
                        for i in range(1, 401)]}
    if missing == "contact":
        for row in physics["test"]:
            row["peg_fixture_contact_force_n"] = 0.0
    elif missing == "withdrawal":
        for row in controls:
            row["phase"] = ["insert"]
    elif missing == "rise":
        for row in controls:
            row["part_pos"][0][2] = 0.025
    elif missing == "pretrigger_stall":
        report["jobs"][0]["first_stall_step"] = 125
    assert recovery_witness(report, controls, physics, 0)["witnessed_complete_recovery"] is (missing is None)


def test_realignment_unloads_before_search_and_is_time_bounded():
    policy = ActorRetryController(observation(), step_dt=1 / 15, position_bounds=[0.05] * 3,
                                 seated_height_m=0.007392, retry=True,
                                 settings=RetrySettings(realignment_min_s=2.0, realignment_max_s=4.0))
    for step in range(61):
        policy.act(observation(), step / 15)
    offset = observation(height=0.045, load=1.0)
    offset[0] = 0.006
    for step in range(61, 150):
        action, state = policy.act(offset, step / 15)
    assert state["phase"] == "realign"
    assert action[0] < 0 and action[2] == pytest.approx(0.9)
    for step in range(150, 167):
        _, state = policy.act(offset, step / 15)
    assert state["phase"] == "search"
    assert policy.events[-1]["observed_alignment_within_tolerance"] is False
