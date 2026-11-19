"""v4_inputs: the bundle manifest and its refusals (no test file, nothing unlisted, sha256)."""

import pytest

from geoguessr_ai.v4_inputs import dev_subset, verify_bundle, write_manifest


def _bundle(tmp_path):
    b = tmp_path / "b"
    (b / "heads").mkdir(parents=True)
    (b / "dev.csv").write_text("id\n1\n")
    (b / "heads" / "h.npy").write_bytes(b"x")
    return b


def test_manifest_round_trip_and_refusals(tmp_path):
    b = _bundle(tmp_path)
    sha = write_manifest(b)
    assert set(verify_bundle(b, sha)) == {"dev.csv", "heads/h.npy"}
    with pytest.raises(SystemExit, match="not the pre-dev config"):
        verify_bundle(b, "0" * 64)
    (b / "extra.csv").write_text("")
    with pytest.raises(SystemExit, match="extra"):
        verify_bundle(b, None)
    (b / "extra.csv").unlink()
    (b / "dev.csv").write_text("id\n2\n")
    with pytest.raises(SystemExit, match="sha256 differs"):
        verify_bundle(b, None)


def test_a_test_file_is_never_bundled(tmp_path):
    b = _bundle(tmp_path)
    (b / "test.csv").write_text("id\n")
    with pytest.raises(SystemExit, match="test file"):
        write_manifest(b)
    (b / "test.csv").unlink()
    write_manifest(b)
    (b / "MANIFEST.sha256").write_text((b / "MANIFEST.sha256").read_text() + "ab  emb_test/x.npz\n")
    with pytest.raises(SystemExit):
        verify_bundle(b, None)


def test_dev_subsets_are_nested_and_order_free():
    ids = [str(i) for i in range(100)]
    five, two = dev_subset(ids, 50), dev_subset(ids[::-1], 20)
    assert two == five[:20] and len(set(five)) == 50
