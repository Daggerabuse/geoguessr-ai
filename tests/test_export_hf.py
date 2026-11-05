"""export_hf: torch-free parts (model card, configs, results reading, token handling)."""

import json

import pytest

from geoguessr_ai import export_hf as e
from geoguessr_ai import train as t

META = {"classes": ["US", "GB", "DE"], "model": "openai/clip-vit-base-patch16",
        "template": "A street view photo taken in {}.", "img_size": 224, "epoch": 10}
RUNS = {"base": (0.111, 0.222, 30), "ft": (0.333, 0.444, 30), "tta": (0.555, 0.666, 30)}
SECRET = "hf_this_is_not_a_real_token_123"


@pytest.fixture
def results_dir(tmp_path):
    root = tmp_path / "results"
    for run, (top1, top5, n) in RUNS.items():
        d = root / run
        d.mkdir(parents=True)
        (d / "test_metrics.json").write_text(json.dumps(
            {"split": "test", "n": n, "n_missing": 0, "top1": top1, "top5": top5}))
        (d / "run_config.json").write_text(json.dumps({"classes": META["classes"]}))
    return root


def _rows(results_dir):
    return e.results_rows(results_dir, [("Base", "base"), ("FT", "ft"), ("TTA", "tta")])


def test_display_names_drop_leading_article():
    n = e.display_names(["US", "GB", "DE", "NL"])
    assert n == {"US": "United States", "GB": "United Kingdom", "DE": "Germany",
                 "NL": "Netherlands"}


def test_inference_config_matches_training_preprocessing():
    cfg = e.inference_config(META, 100.0)
    assert cfg["image_size"] == 224 and cfg["n_classes"] == 3
    assert cfg["mean"] == pytest.approx(t.CLIP_MEAN.tolist())
    assert cfg["std"] == pytest.approx(t.CLIP_STD.tolist())
    assert cfg["tta_crops"] == ["left", "center", "right"]
    assert cfg["logit_scale"] == 100.0 and cfg["base_model"] == META["model"]
    assert cfg["display_names"]["US"] == "United States"
    json.dumps(cfg)


def test_results_rows_read_from_files(results_dir):
    rows = _rows(results_dir)
    assert [(r["top1"], r["top5"], r["n"]) for r in rows] == list(RUNS.values())


def test_missing_metrics_raise(results_dir):
    with pytest.raises(FileNotFoundError):
        e.read_metrics(results_dir, "does_not_exist")


def test_check_run_classes(results_dir):
    e.check_run_classes(results_dir, "ft", META["classes"])
    with pytest.raises(ValueError):
        e.check_run_classes(results_dir, "ft", ["US", "GB"])


def test_model_card_frontmatter_license_and_numbers(results_dir):
    card = e.model_card(META, _rows(results_dir), "user/repo")
    head = card.split("---")[1]
    assert "license: cc-by-sa-4.0" in head
    assert "osv5m/osv5m" in head and "base_model: openai/clip-vit-base-patch16" in head
    for top1, top5, n in RUNS.values():
        assert f"| {top1:.3f} | {top5:.3f} | {n} |" in card
    assert "CC BY-SA 4.0" in card and "Astruc" in card and e.GITHUB_URL in card
    assert 'snapshot_download("user/repo")' in card
    assert "United States (US)" in card


def test_write_metadata_files(results_dir, tmp_path):
    out = tmp_path / "export"
    e.write_metadata(out, META, _rows(results_dir), "user/repo", 100.0)
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "classes.json", "inference_config.json"]
    assert json.loads((out / "classes.json").read_text()) == META["classes"]


def test_get_token_missing_gives_clear_error():
    with pytest.raises(RuntimeError, match="HF_WRITE_TOKEN"):
        e.get_token({})
    with pytest.raises(RuntimeError):
        e.get_token({"HF_WRITE_TOKEN": "   "})


def test_get_token_reads_env_and_never_leaks(results_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_WRITE_TOKEN", SECRET)
    assert e.get_token() == SECRET
    out = tmp_path / "export"
    e.write_metadata(out, META, _rows(results_dir), "user/repo", 100.0)
    for p in out.iterdir():
        assert SECRET not in p.read_text(encoding="utf-8")
    assert SECRET not in "".join(capsys.readouterr())


def test_check_out_dir(tmp_path):
    e.check_out_dir(tmp_path / "new")  # missing is fine
    (tmp_path / "x").mkdir()
    e.check_out_dir(tmp_path / "x")  # empty is fine
    (tmp_path / "x" / "stale.bin").write_bytes(b"")
    with pytest.raises(FileExistsError):
        e.check_out_dir(tmp_path / "x")


def test_parse_args_defaults(tmp_path):
    ck = tmp_path / "best.pt"
    ck.write_bytes(b"")
    a = e.parse_args(["--checkpoint", str(ck)])
    assert a.push is None and not a.private
    assert str(a.out).replace("\\", "/") == "models/hf_export"
    assert (a.run, a.tta_run, a.baseline_run) == (
        "clip_vitb16_lpft", "clip_vitb16_lpft_tta", "clip_vitb16_zeroshot")
    assert a.logit_scale == 100.0


def test_parse_args_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        e.parse_args(["--checkpoint", str(tmp_path / "missing.pt")])


def test_default_runs_exist_in_repo_results():
    """The committed results/ hold every run the card reads, with matching class lists."""
    from pathlib import Path

    res = Path(__file__).parent.parent / "results"
    classes = json.loads((res / "clip_vitb16_lpft" / "run_config.json").read_text())["classes"]
    for run in ("clip_vitb16_lpft", "clip_vitb16_lpft_tta", "clip_vitb16_zeroshot"):
        e.read_metrics(res, run)
        e.check_run_classes(res, run, classes)
    assert len(classes) == 33
