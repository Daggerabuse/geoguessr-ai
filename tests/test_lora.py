"""lora.py (V4) and bench.py's LoRA / archive timing path, on the tiny random CLIP (CPU)."""

import json
import zipfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from geoguessr_ai.lora import LoRALinear, add_lora, lora_parameters
from tests.conftest import write_jpeg


def tiny_vision():
    from transformers import CLIPVisionModelWithProjection

    from geoguessr_ai.train import tiny_vision_config

    torch.manual_seed(0)
    return CLIPVisionModelWithProjection(tiny_vision_config()).eval()


def test_lora_is_identity_at_init_and_only_lora_trains():
    v = tiny_vision()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = v(pixel_values=x).image_embeds
    wrapped = add_lora(v, rank=4)
    assert len(wrapped) == 2 * 4  # 2 blocks x q, k, v, out
    with torch.no_grad():
        after = v(pixel_values=x).image_embeds
    assert torch.equal(before, after)
    trainable = [p for p in v.parameters() if p.requires_grad]
    assert {id(p) for p in trainable} == {id(p) for p in lora_parameters(v)}
    assert sum(p.numel() for p in trainable) == 8 * (4 * 32 + 32 * 4)


def test_lora_last_blocks_only_and_gradients_reach_it():
    v = tiny_vision()
    wrapped = add_lora(v, rank=2, targets=("q_proj", "fc2"), blocks=1)
    assert wrapped == ["layers.1.self_attn.q_proj", "layers.1.mlp.fc2"]
    v(pixel_values=torch.randn(2, 3, 32, 32)).image_embeds.sum().backward()
    assert all(p.grad is not None for p in lora_parameters(v))
    assert v.vision_model.encoder.layers[0].self_attn.q_proj.weight.grad is None
    with pytest.raises(ValueError):
        add_lora(tiny_vision(), rank=2, targets=("nope",))


def test_merged_linear_matches_lora_forward():
    torch.manual_seed(1)
    m = LoRALinear(torch.nn.Linear(6, 5), rank=3, alpha=6)
    torch.nn.init.normal_(m.lora_b)
    x = torch.randn(4, 6)
    with torch.no_grad():
        assert torch.allclose(m(x), m.merged()(x), atol=1e-6)


def test_bench_lora_head_and_archive(tmp_path):
    from geoguessr_ai import bench

    parts = tmp_path / "parts"
    parts.mkdir()
    for k in range(2):  # two ft_archive-style parts: <id>.jpg entries, stored
        with zipfile.ZipFile(parts / f"part_{k:03d}.zip", "w", zipfile.ZIP_STORED) as zf:
            for i in range(6):
                jpg = tmp_path / f"{k}_{i}.jpg"
                write_jpeg(jpg, 60, 40, seed=10 * k + i)
                zf.write(jpg, f"{k}{i:05d}.jpg")
    out = tmp_path / "bench.jsonl"
    res = bench.run(bench.parse_args([
        "--smoke", "--device", "cpu", "--batch-size", "4", "--steps", "2", "--warmup", "1",
        "--lora-rank", "4", "--lora-blocks", "1", "--head-classes", "50", "--grad-checkpointing",
        "--archive", str(parts), "--archive-images", "10", "--workers", "0", "--out", str(out)]))
    assert res["lora_modules"] == 4 and res["head_classes"] == 50
    assert res["trainable_params"] == 4 * (4 * 32 + 32 * 4) + 16 * 50 + 50
    assert res["archive_images"] == 10 and res["archive_parts"] == 2
    for k in ("loader_train_images_per_s", "loader_eval_images_per_s", "e2e_train_images_per_s",
              "e2e_eval_images_per_s"):
        assert res[k] > 0, k
    assert res["ckpt_mb"] > 0 and not list(tmp_path.glob("bench_ckpt_*.pt"))
    assert res["grads_ok"] and res["archive_phases"] == ["loader", "train", "eval", "ckpt"]
    lines = [json.loads(s) for s in out.read_text().splitlines()]
    assert len(lines) == 5 and lines[-1]["complete"] and not lines[0].get("complete")  # synthetic, loader, train, eval, final
    assert "e2e_train_images_per_s" not in lines[1] and "e2e_train_images_per_s" in lines[2]


def test_bench_archive_phase_subset(tmp_path):
    from geoguessr_ai import bench

    parts = tmp_path / "parts"
    parts.mkdir()
    with zipfile.ZipFile(parts / "part_000.zip", "w", zipfile.ZIP_STORED) as zf:
        for i in range(8):
            jpg = tmp_path / f"{i}.jpg"
            write_jpeg(jpg, 60, 40, seed=i)
            zf.write(jpg, f"{i:05d}.jpg")
    res = bench.run(bench.parse_args([
        "--smoke", "--device", "cpu", "--batch-size", "2", "--steps", "1", "--warmup", "1", "--tag", "t",
        "--lora-rank", "2", "--archive", str(parts), "--archive-images", "6", "--archive-phases", "train",
        "--workers", "0", "--out", str(tmp_path / "b.jsonl")]))
    assert res["tag"] == "t" and res["e2e_train_images_per_s"] > 0 and res["e2e_train_seconds"] >= 0
    assert "loader_train_images_per_s" not in res and "ckpt_mb" not in res
