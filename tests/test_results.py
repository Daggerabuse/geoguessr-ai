import argparse
import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from geoguessr_ai import baseline_clip as b
from tests.conftest import CLASSES


def test_evaluate_top1_topk_and_predictions():
    logits = np.array([[3, 1, 0], [0, 2, 1], [1, 0, 5.0]])
    labels = np.array([0, 2, 2])
    hit1, hitk, pred, topk = b.evaluate(logits, labels, k=2)
    assert hit1.tolist() == [True, False, True]
    assert hitk.tolist() == [True, True, True]
    assert pred.tolist() == [0, 1, 2]
    assert topk.shape == (3, 2)


def test_evaluate_clips_k_to_number_of_classes():
    _, _, _, topk = b.evaluate(np.zeros((2, 3)), np.array([0, 1]), k=5)
    assert topk.shape == (2, 3)


def test_confusion_rows_are_true_columns_are_predicted():
    cm = b.confusion(np.array([0, 2, 2]), np.array([0, 1, 2]), 3)
    assert cm.tolist() == [[1, 0, 0], [0, 0, 0], [0, 1, 1]]


def test_country_names_overrides_and_unknown_code():
    n = b.country_names(["US", "TR", "DE", "XX", "GB", "KR"])
    assert n["US"] == "the United States"
    assert n["TR"] == "Turkey"
    assert n["DE"] == "Germany"
    assert n["XX"] == "XX"  # unknown codes fall back to the code itself
    assert n["KR"] == "South Korea"


def test_build_prompts_uses_template():
    assert b.build_prompts(["DE"], "Photo in {}.") == ["Photo in Germany."]


# --- as_embedding: tensor (old transformers) vs ModelOutput (new transformers)


def test_as_embedding_passes_tensor_through():
    a = np.zeros((4, 512))
    assert b.as_embedding(a, "text", 512) is a


def test_as_embedding_prefers_projected_embeds_over_pooler():
    a = np.zeros((4, 512))
    out = NS(text_embeds=a, pooler_output=np.zeros((4, 768)))
    assert b.as_embedding(out, "text", 512) is a


def test_as_embedding_falls_back_to_pooler_output():
    a = np.zeros((4, 512))
    assert b.as_embedding(NS(pooler_output=a), "image", 512) is a
    assert b.as_embedding(NS(image_embeds=np.zeros((2, 768))), "image", 768).shape == (2, 768)


@pytest.mark.parametrize(
    "bad",
    [
        NS(pooler_output=np.zeros((4, 768))),  # wrong dim: projection not applied
        NS(last_hidden_state=np.zeros((4, 512))),  # never used
        NS(text_embeds=None, pooler_output=None),
    ],
)
def test_as_embedding_rejects_unprojected_or_missing(bad):
    with pytest.raises(ValueError):
        b.as_embedding(bad, "text", 512)


def test_as_embedding_rejects_unknown_kind():
    with pytest.raises(ValueError):
        b.as_embedding(np.zeros((1, 512)), "audio", 512)


def test_library_versions_has_both_keys():
    v = b.library_versions()
    assert set(v) == {"torch", "transformers"}
    assert all(x is None or isinstance(x, str) for x in v.values())


# --- result writer and end-to-end run with a fake encoder


def test_write_split_results_files_and_metrics(tmp_path):
    logits = np.random.default_rng(0).normal(size=(6, 3))
    labels = np.array([0, 0, 1, 1, 2, 2])
    m = b.write_split_results(tmp_path, "val", list("abcdef"), labels, logits, CLASSES, 1)
    assert m["n"] == 6 and m["n_missing"] == 1
    assert m["top1"] <= m["top5"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "val_confusion.csv", "val_confusion.png", "val_metrics.json",
        "val_per_country.csv", "val_predictions.csv",
    ]
    cm = pd.read_csv(tmp_path / "val_confusion.csv", index_col=0)
    assert cm.to_numpy().sum() == 6
    assert json.loads((tmp_path / "val_metrics.json").read_text())["split"] == "val"


class FakeEncoder:
    logit_scale = 100.0
    device = "fake"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def encode_texts(self, texts):
        return self.rng.normal(size=(len(texts), 16))

    def encode_images(self, images):
        return self.rng.normal(size=(len(images), 16))


def test_baseline_run_end_to_end_with_missing_image(subset_root, tmp_path):
    (subset_root / "images/test/AU_test_0.jpg").unlink()
    args = argparse.Namespace(
        root=str(subset_root), out=str(tmp_path / "out"), splits=["test", "val"],
        model="fake", template=b.DEFAULT_TEMPLATE, batch_size=4, workers=2, seed=42,
    )
    res = b.run(args, enc=FakeEncoder())
    assert res["test"]["n_missing"] == 1 and res["test"]["n"] == 8
    assert res["val"]["n_missing"] == 0 and res["val"]["n"] == 9
    cfg = json.loads((tmp_path / "out" / "run_config.json").read_text())
    assert set(cfg["versions"]) == {"torch", "transformers"}
    assert cfg["prompts"][0] == "A street view photo taken in Australia."
    cm = pd.read_csv(tmp_path / "out" / "test_confusion.csv", index_col=0)
    assert cm.to_numpy().sum() == 8


# --- TTA (left/center/right crops, logits averaged as in train.py)


def test_mean_over_views_averages_consecutive_rows():
    logits = np.arange(12, dtype=float).reshape(6, 2)  # 2 images x 3 views
    m = b.mean_over_views(logits, 3)
    assert m.shape == (2, 2)
    assert m.tolist() == [[2.0, 3.0], [8.0, 9.0]]
    with pytest.raises(ValueError):
        b.mean_over_views(np.zeros((4, 2)), 3)


class CountingEncoder(FakeEncoder):
    image_size = 32

    def __init__(self):
        super().__init__()
        self.sizes = []

    def encode_images(self, images):
        self.sizes += [im.size for im in images]
        return super().encode_images(images)


def _args(root, out, **kw):
    a = b.parse_args(["--root", str(root), "--out", str(out), "--batch-size", "4",
                      "--workers", "2", "--model", "fake"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_baseline_tta_encodes_three_crops_per_image(subset_root, tmp_path):
    enc = CountingEncoder()
    res = b.run(_args(subset_root, tmp_path / "out", tta=True), enc=enc)
    n_images = res["test"]["n"] + res["val"]["n"]
    assert (res["test"]["n"], res["val"]["n"]) == (9, 9)
    assert len(enc.sizes) == 3 * n_images
    assert set(enc.sizes) == {(32, 32)}  # crops at the encoder's image size
    cfg = json.loads((tmp_path / "out" / "run_config.json").read_text())
    assert cfg["tta"] is True and cfg["image_size"] == 32


def test_baseline_without_tta_passes_full_images(subset_root, tmp_path):
    enc = CountingEncoder()
    res = b.run(_args(subset_root, tmp_path / "out"), enc=enc)
    assert len(enc.sizes) == res["test"]["n"] + res["val"]["n"]
    assert set(enc.sizes) == {(96, 64)}  # untouched: the processor does resize + crop
    assert json.loads((tmp_path / "out" / "run_config.json").read_text())["tta"] is False


def test_baseline_cli_defaults_are_unchanged():
    a = b.parse_args([])
    assert not a.tta
    assert a.out == "results/clip_zeroshot" and a.root == "data/subset"
    assert a.model == b.DEFAULT_MODEL and a.template == b.DEFAULT_TEMPLATE
    assert (a.splits, a.batch_size, a.workers, a.seed) == (["test", "val"], 128, 4, 42)
