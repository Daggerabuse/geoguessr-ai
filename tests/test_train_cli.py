import argparse
import json

import pytest

from geoguessr_ai import train as t


def _ckpt(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"not a real checkpoint")  # parse_args only checks that the file exists
    return str(p)


def test_eval_only_requires_checkpoint(capsys):
    with pytest.raises(SystemExit) as e:
        t.parse_args(["--eval-only", "--run-name", "x"])
    assert e.value.code == 2
    assert "--eval-only requires --checkpoint" in capsys.readouterr().err


def test_eval_only_requires_existing_checkpoint_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        t.parse_args(["--eval-only", "--checkpoint", str(tmp_path / "missing.pt"), "--run-name", "x"])
    assert e.value.code == 2
    assert "checkpoint not found" in capsys.readouterr().err


def test_eval_only_requires_explicit_run_name(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        t.parse_args(["--eval-only", "--checkpoint", _ckpt(tmp_path)])
    assert e.value.code == 2
    assert "explicit --run-name" in capsys.readouterr().err


def test_checkpoint_without_eval_only_is_rejected(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        t.parse_args(["--checkpoint", _ckpt(tmp_path)])
    assert e.value.code == 2
    assert "only valid together with --eval-only" in capsys.readouterr().err


def test_valid_eval_only_arguments(tmp_path):
    ck = _ckpt(tmp_path)
    a = t.parse_args(["--root", "r", "--eval-only", "--checkpoint", ck, "--tta",
                      "--run-name", "clip_vitb16_lpft_tta"])
    assert a.eval_only and a.tta
    assert str(a.checkpoint) == ck
    assert a.run_name == "clip_vitb16_lpft_tta"


def test_training_defaults_are_unchanged():
    a = t.parse_args([])
    assert a.run_name == "clip_vitb16_lpft"
    assert not a.eval_only and a.checkpoint is None and not a.tta and not a.smoke
    assert (a.epochs, a.probe_epochs, a.batch_size, a.patience, a.seed) == (10, 30, 64, 3, 42)
    assert (a.lr_backbone, a.lr_head, a.probe_lr, a.weight_decay) == (1e-5, 1e-4, 1e-3, 0.05)
    assert a.model == "openai/clip-vit-base-patch16"
    assert a.root == "data/subset" and a.results_dir == "results" and a.models_dir == "models"


def test_check_checkpoint_classes():
    t.check_checkpoint_classes(["A", "B"], ["A", "B"])
    with pytest.raises(ValueError, match="differ"):
        t.check_checkpoint_classes(["A", "B"], ["A", "C"])
    with pytest.raises(ValueError):
        t.check_checkpoint_classes(["A"], ["A", "B"])


def test_eval_run_config_prefers_checkpoint_metadata(tmp_path):
    args = t.parse_args(["--eval-only", "--checkpoint", _ckpt(tmp_path), "--run-name", "r",
                         "--model", "other/model", "--template", "ignored {}"])
    meta = {"classes": ["DE"], "model": "ckpt/model", "template": "Photo in {}.",
            "img_size": 224, "epoch": 10, "val_top1": 0.5}
    cfg = t.eval_run_config(args, "r", "cpu", meta, {"val": 0, "test": 1})
    assert cfg["eval_only"] is True
    assert cfg["model"] == "ckpt/model" and cfg["template"] == "Photo in {}."
    assert cfg["prompts"] == ["Photo in Germany."]
    assert cfg["checkpoint_epoch"] == 10 and cfg["checkpoint_val_top1"] == 0.5
    assert cfg["n_missing"] == {"val": 0, "test": 1}
    assert set(cfg["versions"]) == {"torch", "transformers"}
    json.dumps(cfg, default=str)  # must be serialisable


def test_eval_run_config_without_optional_checkpoint_fields(tmp_path):
    args = argparse.Namespace(checkpoint=tmp_path / "c.pt", tta=True)
    meta = {"classes": ["DE"], "model": "m", "template": "{}", "img_size": 32}
    cfg = t.eval_run_config(args, "r", "cpu", meta, {})
    assert cfg["checkpoint_epoch"] is None and cfg["checkpoint_val_top1"] is None
    assert cfg["tta"] is True
