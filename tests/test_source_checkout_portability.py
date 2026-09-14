"""A Windows evidence record must admit the same Python in a Linux checkout."""
import hashlib
from pathlib import Path

import pytest

from zero_g_blade_swap.service.config import ServiceSettings
from zero_g_blade_swap.service.presets import PresetRegistry, _source_digest_matches, sha256_file


@pytest.mark.parametrize("recorded_eol,checkout_eol", [(b"\r\n", b"\n"), (b"\n", b"\r\n")])
def test_equivalent_python_checkout_keeps_source_admission(tmp_path, recorded_eol, checkout_eol):
    original = recorded_eol.join([b"threshold = 0.0025", b"hold = 0.70", b""])
    relative = Path("src/controller.py")
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(original.replace(recorded_eol, checkout_eol))
    record = {"runtime_source_bindings": [{"path": relative.as_posix(),
               "sha256": hashlib.sha256(original).hexdigest()}]}
    registry = PresetRegistry(ServiceSettings(tmp_path, tmp_path / "runtime", tmp_path / "static", tmp_path / "isaac"))
    assert registry._runtime_binding_reasons(record, (relative,), "workflow") == []
    path.write_bytes(path.read_bytes().replace(b"0.0025", b"0.0030"))
    assert registry._runtime_binding_reasons(record, (relative,), "workflow") == [
        "workflow is stale for src/controller.py"
    ]


def test_binary_and_saved_artifact_hashes_remain_exact(tmp_path):
    path = tmp_path / "policy.pth"
    path.write_bytes(b"opaque\r\nbytes\r\n")
    recorded = sha256_file(path)
    path.write_bytes(b"opaque\nbytes\n")
    assert sha256_file(path) != recorded
    assert not _source_digest_matches(path, recorded)


def test_mixed_historical_digest_requires_exact_bytes(tmp_path):
    mixed = b"first = 1\r\nsecond = 2\n"
    recorded = hashlib.sha256(mixed).hexdigest()
    path = tmp_path / "controller.py"
    path.write_bytes(mixed.replace(b"\r\n", b"\n"))
    assert not _source_digest_matches(path, recorded)
    path.write_bytes(mixed)
    assert _source_digest_matches(path, recorded)
