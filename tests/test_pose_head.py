"""Contracts for the module-pose regressor, asserted without a simulator.

``pose_head.py`` imports nothing from Isaac Lab so that the head can be trained
on a machine with no simulator and reached by the suite on every commit. These
tests hold that line and pin the two properties that would otherwise be
discovered the expensive way:

* a checkpoint trained before the occupancy branch existed must keep loading and
  behaving exactly as it did, because `evidence/module_pose_head.json` describes
  such a head and that number has to keep describing it;
* a head must never be *described* differently than it was trained, which is why
  both output widths are recovered from the checkpoint rather than passed in.
"""

from __future__ import annotations

import json

import torch

from zero_g_blade_swap.pose_head import (
    MODULE_POSE_DIM,
    POSE_HEAD_ARCHITECTURE_V1,
    POSE_HEAD_ARCHITECTURE_V2,
    POSE_HEAD_LEGACY_GRID_SIZE,
    POSE_HEAD_OVERVIEW_GRID_SIZE,
    SECOND_SLOT_OCCUPANCY_SLOTS,
    ModulePoseHead,
    checkpoint_matches_sha256,
    checkpoint_sha256,
    load_pose_head,
)

BATCH = 3


def _image(size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand((BATCH, size, size, 3), generator=generator)


def test_a_head_without_an_occupancy_branch_is_unchanged() -> None:
    head = ModulePoseHead()
    assert head.occupancy is None
    assert head(_image()).shape == (BATCH, MODULE_POSE_DIM)


def test_reading_a_bay_off_a_head_that_never_saw_two_raises() -> None:
    """Refusing beats inventing. A silent zero here is a perception claim."""

    head = ModulePoseHead()
    try:
        head.forward_with_occupancy(_image())
    except AttributeError as error:
        assert "occupancy branch" in str(error)
    else:
        raise AssertionError("a head with no occupancy branch must refuse to report one")


def test_the_occupancy_branch_shares_the_trunk_and_agrees_with_forward() -> None:
    head = ModulePoseHead(occupancy_slots=SECOND_SLOT_OCCUPANCY_SLOTS).eval()
    image = _image()
    with torch.no_grad():
        pose, logits = head.forward_with_occupancy(image)
        probabilities = head.occupancy_probabilities(image)
        # The pose is the same quantity whichever entry point asked for it, or
        # the two paths are two different heads wearing one name.
        assert torch.allclose(pose, head(image))
    assert pose.shape == (BATCH, MODULE_POSE_DIM)
    assert logits.shape == (BATCH, SECOND_SLOT_OCCUPANCY_SLOTS)
    assert bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


def test_both_output_widths_are_recovered_from_the_checkpoint(tmp_path) -> None:
    for slots in (0, SECOND_SLOT_OCCUPANCY_SLOTS):
        head = ModulePoseHead(occupancy_slots=slots)
        # Label statistics travel inside the checkpoint, so give them values a
        # default-constructed head would not have and check they survive.
        head.label_mean += 0.25
        head.label_std *= 2.0
        path = tmp_path / f"head_{slots}.pt"
        torch.save(head.state_dict(), path)

        loaded = load_pose_head(path, "cpu")
        assert loaded.occupancy_slots == slots
        assert (loaded.occupancy is None) == (slots == 0)
        assert torch.allclose(loaded.label_mean, head.label_mean)
        assert torch.allclose(loaded.label_std, head.label_std)
        assert not any(parameter.requires_grad for parameter in loaded.parameters())
        with torch.no_grad():
            assert torch.allclose(loaded(_image()), head.eval()(_image()))


def test_legacy_v1_grid_is_inferred_when_metadata_is_absent(tmp_path) -> None:
    head = ModulePoseHead(occupancy_slots=SECOND_SLOT_OCCUPANCY_SLOTS).eval()
    legacy_state = head.state_dict()
    legacy_state.pop("_architecture_version")
    legacy_state.pop("_feature_grid_size")
    path = tmp_path / "legacy_v1.pt"
    torch.save(legacy_state, path)

    loaded = load_pose_head(path, "cpu")
    assert loaded.architecture_version == POSE_HEAD_ARCHITECTURE_V1
    assert loaded.feature_grid_size == POSE_HEAD_LEGACY_GRID_SIZE
    with torch.no_grad():
        expected_pose, expected_occupancy = head.forward_with_occupancy(_image())
        actual_pose, actual_occupancy = loaded.forward_with_occupancy(_image())
    assert torch.allclose(actual_pose, expected_pose)
    assert torch.allclose(actual_occupancy, expected_occupancy)


def test_overview_v2_grid_round_trips_and_runs_at_256px(tmp_path) -> None:
    head = ModulePoseHead(
        occupancy_slots=SECOND_SLOT_OCCUPANCY_SLOTS,
        feature_grid_size=POSE_HEAD_OVERVIEW_GRID_SIZE,
        architecture_version=POSE_HEAD_ARCHITECTURE_V2,
    ).eval()
    path = tmp_path / "overview_v2.pt"
    torch.save(head.state_dict(), path)

    loaded = load_pose_head(path, "cpu")
    assert loaded.architecture_version == POSE_HEAD_ARCHITECTURE_V2
    assert loaded.feature_grid_size == POSE_HEAD_OVERVIEW_GRID_SIZE
    with torch.no_grad():
        pose, occupancy = loaded.forward_with_occupancy(_image(256))
    assert pose.shape == (BATCH, MODULE_POSE_DIM)
    assert occupancy.shape == (BATCH, SECOND_SLOT_OCCUPANCY_SLOTS)


def test_loader_rejects_architecture_metadata_that_disagrees_with_weights(tmp_path) -> None:
    head = ModulePoseHead(
        occupancy_slots=SECOND_SLOT_OCCUPANCY_SLOTS,
        feature_grid_size=POSE_HEAD_OVERVIEW_GRID_SIZE,
    )
    state = head.state_dict()
    state["_feature_grid_size"] = torch.tensor(POSE_HEAD_LEGACY_GRID_SIZE)
    path = tmp_path / "contradictory_architecture.pt"
    torch.save(state, path)

    try:
        load_pose_head(path, "cpu")
    except ValueError as error:
        assert "regressor weights require" in str(error)
    else:
        raise AssertionError("loader accepted architecture metadata contradicted by learned tensor widths")


def test_checkpoint_report_hash_identifies_the_exact_weights(tmp_path) -> None:
    checkpoint = tmp_path / "head.pt"
    torch.save(ModulePoseHead().state_dict(), checkpoint)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"checkpoint_sha256": checkpoint_sha256(checkpoint)}), encoding="utf-8")

    recorded = json.loads(report.read_text(encoding="utf-8"))["checkpoint_sha256"]
    assert checkpoint_matches_sha256(checkpoint, recorded)
    assert not checkpoint_matches_sha256(checkpoint, "0" * 64)
    assert not checkpoint_matches_sha256(checkpoint, "not-a-digest")
