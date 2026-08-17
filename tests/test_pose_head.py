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

import torch

from zero_g_blade_swap.pose_head import (
    MODULE_POSE_DIM,
    SECOND_SLOT_OCCUPANCY_SLOTS,
    ModulePoseHead,
    load_pose_head,
)

BATCH = 3


def _image() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand((BATCH, 64, 64, 3), generator=generator)


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
