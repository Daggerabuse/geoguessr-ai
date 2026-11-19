"""v4_train (V4 LoRA trainer) on the tiny random CLIP, CPU: a fake bundle + archive (make_world)."""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from geoguessr_ai import geocells as gc
from geoguessr_ai import v4_train as vt
from geoguessr_ai.v4_inputs import file_sha256, write_manifest
from tests.conftest import write_jpeg

REPO = Path(__file__).resolve().parents[1]
N_TRAIN, N_DEV, N_DEV_SEEN, CELLS_MAX = 24, 10, 6, 6


def _rows(prefix: str, n: int, rng) -> pd.DataFrame:
    return pd.DataFrame({
        "id": [f"{prefix}{i:04d}" for i in range(n)],
        "latitude": rng.uniform(-60, 60, n), "longitude": rng.uniform(-170, 170, n),
        "country": ["AU"] * n, "sequence": [f"{prefix}s{i}" for i in range(n)],
        "creator_id": [f"{prefix}c{i % 4}" for i in range(n)], "label": [0] * n})


def make_world(root: Path, n_train: int = N_TRAIN) -> dict:
    """parts/ (2 archive parts + manifest.csv), bundle/ (a v4_inputs-style bundle whose identity
    embeddings come from the tiny frozen encoder itself), config.json; returns the CLI args."""
    from io import BytesIO

    from PIL import Image

    from geoguessr_ai.embed import clip_center_crop, load_encoder
    from geoguessr_ai.train import to_tensor_array

    rng = np.random.default_rng(0)
    train = _rows("t", n_train, rng).assign(bal_rank=lambda d: np.arange(len(d))[::-1])
    dev, dev_seen = _rows("d", N_DEV, rng), _rows("s", N_DEV_SEEN, rng)
    parts, bundle = root / "parts", root / "bundle"
    parts.mkdir(parents=True)
    ids = [*train["id"], *dev["id"], *dev_seen["id"]]
    man = []
    for k in range(2):
        with zipfile.ZipFile(parts / f"part_{k:03d}.zip", "w", zipfile.ZIP_STORED) as zf:
            for j, i in enumerate(ids[k::2]):
                jpg = root / "jpg" / f"{i}.jpg"
                write_jpeg(jpg, 60 + 7 * (j % 3), 40, seed=ids.index(i))
                zf.write(jpg, f"{i}.jpg")
                man.append({"id": i, "part": f"part_{k:03d}.zip"})
    pd.DataFrame(man).to_csv(parts / "manifest.csv", index=False)

    bundle.mkdir()
    train.sort_values("bal_rank").to_csv(bundle / "train.csv", index=False)
    dev.to_csv(bundle / "dev.csv", index=False)
    dev_seen.to_csv(bundle / "dev_seen.csv", index=False)
    tree = gc.build(train["latitude"], train["longitude"], CELLS_MAX)
    gsha = gc.save(tree, bundle / "cells_5000.json")
    torch.manual_seed(0)  # = build_model's smoke encoder
    enc, size, dim = load_encoder("smoke", True, torch.device("cpu"))
    dev_ids = vt.dev_subset(dev["id"], 4, 42)
    imgs = []
    for i in dev_ids:
        with zipfile.ZipFile(parts / {m["id"]: m["part"] for m in man}[i]) as zf:
            with Image.open(BytesIO(zf.read(f"{i}.jpg"))) as im:
                imgs.append(to_tensor_array(clip_center_crop(im.convert("RGB"), size)))
    with torch.no_grad():
        e = torch.nn.functional.normalize(enc(pixel_values=torch.from_numpy(np.stack(imgs))).image_embeds, dim=-1)
    for bb in ("streetclip", "clip_l14_336"):
        (bundle / "heads" / bb).mkdir(parents=True)
        for n in {n_train, 12}:
            np.save(bundle / "heads" / bb / f"bal_{n}_cell_zeroshot.npy",
                    np.random.default_rng(n).standard_normal((gc.n_cells(tree), dim)).astype(np.float32))
        (bundle / "identity").mkdir(exist_ok=True)
        np.savez(bundle / "identity" / f"{bb}_dev2k.npz", ids=np.array(dev_ids), emb=e.numpy())

    cfg = json.loads((REPO / "configs" / "v4_predev.json").read_text())
    cfg["train"].update(effective_batch=4, micro_batch=2, workers=0, eval_batch=4)
    cfg["seen"] = {"short": 40, "full": 64}
    cfg.update(ckpt_every_steps=3, log_every_steps=1, n_boot=20)
    cfg["evals"].update(intermediate_n=6, diagnostic_n=4)
    cfg["identity"].update(head_n=n_train)
    cfg["inputs"].update(geocells_sha256=gsha, bundle_manifest_sha256=write_manifest(bundle),
                         archive_manifest_sha256=file_sha256(parts / "manifest.csv"))
    for j in cfg["jobs"]:
        j["n"] = 12 if j["n"] == 500000 else n_train
    (root / "config.json").write_text(json.dumps(cfg))
    return {"config": root / "config.json", "inputs": bundle, "parts": parts, "out": root / "out"}


def _args(w: dict, cmd: str, *extra) -> list[str]:
    return [cmd, "--config", str(w["config"]), "--inputs", str(w["inputs"]), "--parts", str(w["parts"]),
            "--out", str(w["out"]), "--smoke", *extra]


def _call(argv) -> int:
    with pytest.raises(SystemExit) as e:
        vt.main(argv)
    return e.value.code


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = make_world(tmp_path_factory.mktemp("v4"))
    assert _call(_args(w, "identity", "--backbone", "streetclip")) == 0
    return w


def _losses(rdir: Path) -> list:
    rows = [json.loads(s) for s in (rdir / "log.jsonl").read_text().splitlines()]
    return [(r["event"], r.get("step"), r.get("loss"), r.get("train_loss"), r.get("geoscore"))
            for r in rows if r["event"] in ("train", "epoch", "eval", "initial_loss")]


def _weights(rdir: Path) -> dict:
    return torch.load(rdir / "weights_final.pt")


def test_sampler_is_a_resumable_stream_over_epochs():
    full = [x for b in vt.StreamSampler(10, 7, 0, 25, 4) for x in b]
    assert len(full) == 25 and [e for e, _ in full] == [0] * 10 + [1] * 10 + [2] * 5
    assert sorted(r for e, r in full if e == 0) == list(range(10))
    assert [r for e, r in full[:10]] != [r for e, r in full[10:20]]  # a new permutation per epoch
    tail = [x for b in vt.StreamSampler(10, 7, 12, 25, 4) for x in b]
    assert tail == full[12:]
    assert full == [x for b in vt.StreamSampler(10, 7, 0, 25, 4) for x in b]
    assert len(vt.StreamSampler(10, 7, 12, 25, 4)) == 4


def test_device_rule_and_lr_resolution():
    assert [vt.device_for(k, "streetclip") for k in range(4)] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert [vt.device_for(k, "clip_l14_336") for k in range(4)] == ["cuda:1", "cuda:0", "cuda:1", "cuda:0"]
    sel = {"edge": {"run": True, "lr": 0.003}, "selected": None}
    assert vt.job_lr({"k": 3, "lr": "edge"}, "streetclip", sel) == 0.003
    with pytest.raises(SystemExit, match="edge extension"):
        vt.job_lr({"k": 4, "lr": "selected"}, "streetclip", sel)
    with pytest.raises(SystemExit, match="does not run"):
        vt.job_lr({"k": 3, "lr": "edge"}, "x", {"edge": {"run": False}, "selected": {"x": 1e-4}})
    assert vt.job_lr({"k": 4, "lr": "selected"}, "x", {"edge": {"run": False}, "selected": {"x": 1e-4}}) == 1e-4
    with pytest.raises(SystemExit, match="selection file"):
        vt.job_lr({"k": 4, "lr": "selected"}, "x", None)


def test_same_epoch_and_rank_give_the_same_pixels(world):
    where = vt.archive_index(world["parts"], None)
    tr = pd.read_csv(world["inputs"] / "train.csv", dtype={"id": str})
    a = vt.PartImages(world["parts"], where, tr["id"], 32, True, 42, tr["bal_rank"])
    b = vt.PartImages(world["parts"], where, tr["id"][::-1].tolist(), 32, True, 42, tr["bal_rank"][::-1].tolist())
    n = len(tr)
    assert np.array_equal(a[(1, 0)][0], b[(1, n - 1)][0])  # same image, same epoch: same pixels
    assert not np.array_equal(a[(1, 0)][0], a[(2, 0)][0])


def test_identity_passes_and_lora_is_exact_at_init(world):
    res = json.loads((world["out"] / "identity_streetclip.json").read_text())
    assert res["ok"] and res["top1_cell_same"] == 1.0
    cfg = json.loads(world["config"].read_text())
    head = np.random.default_rng(1).standard_normal((5, 16)).astype(np.float32)
    lora, _, _ = vt.build_model(cfg, "smoke", head, "short", 42, True, torch.device("cpu"))
    frozen, tr, _ = vt.build_model(cfg, "smoke", head, "control", 42, True, torch.device("cpu"))
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(lora(x), frozen(x))
    assert list(tr) == ["head"]  # the control trains the head only


def test_train_refuses_without_identity_and_on_the_wrong_device(tmp_path):
    w = make_world(tmp_path)
    with pytest.raises(SystemExit, match="identity check"):
        vt.main(_args(w, "train", "--job", "0", "--backbone", "clip_l14_336"))
    with pytest.raises(SystemExit, match="decision 7"):
        a = vt.parse_args(_args(w, "train", "--job", "1", "--backbone", "streetclip", "--device", "cuda:0"))
        a.smoke = False
        vt.setup(a)


def test_stop_and_resume_equal_an_uninterrupted_run(world, tmp_path):
    """The full chain: a STOP at step 5 (exit 3, RESUME NEEDED), a resume from the checkpoint,
    and the final weights and the loss/eval log are those of an uninterrupted run."""
    one = dict(world, out=tmp_path / "one")
    two = dict(world, out=tmp_path / "two")
    for w in (one, two):
        w["out"].mkdir()
        shutil.copy(world["out"] / "identity_streetclip.json", w["out"])
    job = ["--job", "0", "--backbone", "streetclip"]
    assert _call(_args(one, "train", *job)) == 0
    assert _call(_args(two, "train", *job, "--stop-after-steps", "5")) == 3
    rdir1, rdir2 = (w["out"] / "k0_short_lr1e-4_streetclip" for w in (one, two))
    assert not (rdir2 / "DONE").exists() and torch.load(rdir2 / "ckpt" / "last.pt", weights_only=False)["step"] == 5
    assert _call(_args(two, "train", *job)) == 0
    w1, w2 = _weights(rdir1), _weights(rdir2)
    assert set(w1) == set(w2) and all(torch.equal(w1[k], w2[k]) for k in w1)
    assert _losses(rdir1) == _losses(rdir2)
    res = json.loads((rdir2 / "result.json").read_text())
    assert res["steps"] == 10 and res["images_seen"] == 40 and res["lr"] == 1e-4 and res["head_lr"] == 1e-3
    assert {"dev", "dev_seen"} <= set(res) and res["dev"]["n"] == N_DEV and res["dev_seen"]["n"] == N_DEV_SEEN
    ev = [r for r in map(json.loads, (rdir2 / "log.jsonl").read_text().splitlines()) if r["event"] == "eval"]
    assert [e["kind"] for e in ev] == ["final", "final"]  # short: no intermediate, no diagnostic
    assert not list(rdir2.glob("weights_[0-9]*.pt"))
    assert pd.read_csv(rdir2 / "final" / "dev_predictions.csv").shape[0] == N_DEV
    assert np.load(rdir2 / "final" / "dev_emb.npz")["emb"].shape == (N_DEV, 16)
    assert _call(_args(two, "train", *job)) == 0  # DONE: nothing reruns


def test_diagnostic_evals_never_change_the_weights(world, tmp_path, monkeypatch):
    """The 50K full run (job 4) is the only one with the per-epoch diagnostic; stubbing it out
    leaves the final weights bit-identical."""
    out = tmp_path / "a"
    out.mkdir()
    shutil.copy(world["out"] / "identity_streetclip.json", out)
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"edge": {"run": False}, "selected": {"streetclip": 3e-4, "clip_l14_336": 1e-3}}))
    w = dict(world, out=out)
    rdir = out / "k4_full_50k_streetclip"
    job = ["--job", "4", "--backbone", "streetclip", "--selection", str(sel)]
    assert _call(_args(w, "train", *job)) == 0
    kinds = [r["kind"] for r in map(json.loads, (rdir / "log.jsonl").read_text().splitlines())
             if r["event"] == "eval"]
    assert kinds == ["intermediate", "diagnostic_epoch", "intermediate", "diagnostic_epoch", "final", "final"]
    ref = _weights(rdir)
    shutil.rmtree(rdir)
    calls = []
    real = vt.Eval.run

    def no_diag(self, *a, **k):  # the epoch diagnostic skipped entirely
        calls.append(self.name)
        if self.name == "diag":
            return np.zeros((len(self.ds), 5), dtype=int), None
        return real(self, *a, **k)

    monkeypatch.setattr(vt.Eval, "run", no_diag)
    assert _call(_args(w, "train", *job)) == 0
    new = _weights(rdir)
    assert "diag" in calls and all(torch.equal(ref[k], new[k]) for k in ref)
    calls.clear()  # the seed-2 replicate (job 6) runs no diagnostic
    assert _call(_args(w, "train", "--job", "6", "--backbone", "streetclip", "--selection", str(sel))) == 0
    assert calls and "diag" not in calls


def test_a_changed_setting_refuses_the_old_checkpoint(world, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    shutil.copy(world["out"] / "identity_streetclip.json", out)
    w = dict(world, out=out)
    job = ["--job", "0", "--backbone", "streetclip"]
    assert _call(_args(w, "train", *job, "--stop-after-steps", "2")) == 3
    cfg = json.loads(world["config"].read_text())
    cfg["train"]["grad_clip"] = 2.0
    changed = tmp_path / "c.json"
    changed.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="run key"):
        vt.main(_args(dict(w, config=changed), "train", *job))


def test_control_and_the_large_n_run(world, tmp_path):
    out = tmp_path / "c"
    out.mkdir()
    for bb in ("streetclip", "clip_l14_336"):
        assert _call(_args(dict(world, out=out), "identity", "--backbone", bb)) == 0
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"edge": {"run": False}, "selected": {"streetclip": 3e-4, "clip_l14_336": 1e-3}}))
    w = dict(world, out=out)
    assert _call(_args(w, "train", "--job", "7", "--backbone", "clip_l14_336", "--selection", str(sel))) == 0
    r = json.loads((out / "k7_control_50k_clip_l14_336" / "result.json").read_text())
    assert r["kind"] == "control" and r["head_lr"] == pytest.approx(1e-2) and r["steps"] == 16
    assert set(_weights(out / "k7_control_50k_clip_l14_336")) == {"head"}
    ev = [r["kind"] for r in map(json.loads, (out / "k7_control_50k_clip_l14_336" / "log.jsonl")
                                 .read_text().splitlines()) if r["event"] == "eval"]
    assert ev == ["intermediate", "intermediate", "final", "final"]  # no diagnostic outside job 4
    assert (out / "k7_control_50k_clip_l14_336" / "weights_5.pt").exists()
    assert _call(_args(w, "train", "--job", "5", "--backbone", "streetclip", "--selection", str(sel))) == 0
    r = json.loads((out / "k5_full_500k_streetclip" / "result.json").read_text())
    assert r["n"] == 12 and r["lr"] == 3e-4
