import json
from pathlib import Path

from assembly_recovery.retry_controller import ActorRetryController, RetrySettings


def test_frozen_retry_and_continue_share_entire_four_second_prefix():
    settings = RetrySettings(**json.loads((Path(__file__).resolve().parents[1] / "configs/retry_unload_realign_v1.json").read_text()))
    observation = [0.] * 24
    observation[2], observation[13], observation[16] = .03, 10., 20.
    controllers = [ActorRetryController(observation, step_dt=1 / 15, position_bounds=(.05,) * 3,
                   seated_height_m=0., retry=enabled, settings=settings) for enabled in (False, True)]
    for step in range(60):
        actions = [c.act(observation, step / 15)[0] for c in controllers]
        assert actions[0] == actions[1]
        assert all(c.trigger is None for c in controllers)
    actions = [c.act(observation, 4.)[0] for c in controllers]
    assert controllers[0].phase == "insert"
    assert controllers[1].phase == "withdraw"
    assert actions[0] != actions[1]
