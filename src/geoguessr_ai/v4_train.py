"""V4: resumable LoRA fine-tuning of one backbone for one declared job (docs/V4_FEASIBILITY.md).

One process = one backbone x one job k of configs/v4_predev.json, on one GPU. Two processes (the
backbones) run side by side, StreetCLIP on cuda:(k mod 2) and L/14-336 on the other GPU
(decision 7). scripts/kaggle_v4.sh runs the pair.

  train     LoRA r16 on q,k,v,out_proj of the last 12 blocks (lora.py) + the cosine head initialised
            from V3's bal cell_zeroshot head of the same N; AdamW (head lr = head_lr_mult x lr),
            warmup + cosine to 0, AMP, gradient accumulation to the effective batch. kind=control:
            no LoRA, frozen encoder under no_grad, head only, same augmented schedule.
            train.freeze_head (recipe B, configs/v4_predev_v2.json): the head stays the V3 head
            (no head parameters train, LoRA only); a control run is refused (nothing would train).
            Data order: a flat stream over seeded per-epoch permutations; the augmentation of an
            image depends only on (seed, epoch, bal_rank), so both backbones and both N see the
            same pixels. A resume restarts the stream at its position without decoding.
            Checkpoint (ckpt/last.pt: trainable weights, AdamW, GradScaler, scheduler, position, RNG)
            every ckpt_every_steps, at each eval point and on STOP; pushed to the private runs repo.
            Evals: per-epoch 2K dev_unseen in the jobs of evals.diagnostic_jobs only (the 50K full
            run; diagnostic only: never used for selection, stopping or checkpoint choice), 5K dev_unseen at S/3 and 2S/3 (full and control runs; diagnostic
            only), and at the end the full dev_unseen
            (the selection split) and dev_seen (diagnostic), with per-image predictions, dev
            embeddings (for the later hybrid) and weights_final.pt.
  identity  B = 0 and V3's 50K head on the 2K diagnostic subset, against V3's stored embeddings x
            that head: the top-1 cell must agree on >= identity.top1_same_min. train refuses to
            start until it has passed for its backbone.
  stage     download the archive parts + manifest from the private HF archive repo, sha256-checked.
  hfcheck   the runs repo is private and HF_TOKEN can write to it (a file is put, then deleted).

The official test split has no path here: the inputs bundle (v4_inputs.py) holds no test file.
STOP (--stop-file) is checked before every optimizer step and before the final eval (the per-epoch
and intermediate evals take minutes and always finish, so a resume never skips one): checkpoint,
push, "RESUME NEEDED", exit 3.

  python -m geoguessr_ai.v4_train train --config configs/v4_predev.json --job 4 --backbone streetclip \\
      --inputs <bundle> --parts /kaggle/temp/v4_parts --out /kaggle/working/v4_runs --device auto
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geoguessr_ai.v4_inputs import dev_subset, file_sha256, read_csv, verify_bundle

RESUME_EXIT = 3


# ----------------------------------------------------------------- job and config (no torch)


def device_for(k: int, backbone: str) -> str:
    """Decision 7: StreetCLIP on cuda:(k mod 2), L/14-336 on the other GPU."""
    s = k % 2
    return f"cuda:{s if backbone == 'streetclip' else 1 - s}"


def job_of(cfg: dict, k: int) -> dict:
    jobs = [j for j in cfg["jobs"] if j["k"] == k]
    if len(jobs) != 1:
        raise SystemExit(f"job {k} is not declared in the config")
    return jobs[0]


def job_lr(job: dict, backbone: str, selection: dict | None) -> float:
    """A numeric lr as declared, or the one v4_select recorded ("edge", "selected")."""
    lr = job["lr"]
    if isinstance(lr, (int, float)):
        return float(lr)
    if selection is None:
        raise SystemExit(f"job {job['k']} needs the committed selection file (lr = {lr!r})")
    if lr == "edge":
        if not selection["edge"].get("run"):
            raise SystemExit("the edge rule did not ask for an extension: job 3 does not run")
        return float(selection["edge"]["lr"])
    if lr == "selected":
        if selection.get("selected") is None:
            raise SystemExit("no final lr yet (the edge extension must run first)")
        return float(selection["selected"][backbone])
    raise SystemExit(f"job {job['k']}: unknown lr {lr!r}")


def schedule(cfg: dict, job: dict) -> dict:
    t = cfg["train"]
    eff, micro = t["effective_batch"], t["micro_batch"]
    if eff % micro:
        raise SystemExit("effective_batch must be a multiple of micro_batch")
    total = cfg["seen"][job["kind"] if job["kind"] == "short" else "full"] // eff
    # intermediate 5K evals in the full and control runs only (as budgeted; short runs: final only)
    inter = [] if job["kind"] == "short" else [total * i // 3 for i in (1, 2)]
    return {"eff": eff, "micro": micro, "accum": eff // micro, "total": total,
            "warmup": int(t["warmup_frac"] * total), "intermediate": inter}


def run_key(cfg: dict, job: dict, backbone: str, lr: float, commit: str, hashes: dict) -> str:
    from geoguessr_ai.phase3_eval import cache_key

    return cache_key({"config": {k: v for k, v in cfg.items() if k != "note"}, "job": job,
                      "backbone": backbone, "lr": lr, "code_commit": commit, **hashes})


def git_checks(config: Path, selection: Path | None, smoke: bool) -> str:
    """Code commit; the config (and the selection file if used) committed and unmodified."""
    if smoke:
        return "smoke"
    from geoguessr_ai.phase3_final import code_commit
    from geoguessr_ai.prereg import git_state

    git_state(config)
    if selection is not None:
        git_state(selection)
    return code_commit(config)


# ----------------------------------------------------------------- data (no torch)


class StreamSampler:
    """Micro-batches of (epoch, row) from global position `start` to `stop`: epoch e is the
    permutation default_rng([seed, e]) of n rows, and the epochs follow one another as one stream
    (no partial batch at an epoch end). Positions are counted in images seen."""

    def __init__(self, n: int, seed: int, start: int, stop: int, batch: int):
        self.n, self.seed, self.start, self.stop, self.batch = n, seed, start, stop, batch
        self._perm: dict[int, np.ndarray] = {}

    def item(self, p: int) -> tuple[int, int]:
        e = p // self.n
        if e not in self._perm:
            self._perm = {e: np.random.default_rng([self.seed, e]).permutation(self.n)}
        return e, int(self._perm[e][p % self.n])

    def __iter__(self):
        for s in range(self.start, self.stop, self.batch):
            yield [self.item(p) for p in range(s, min(s + self.batch, self.stop))]

    def __len__(self) -> int:
        return -(-(self.stop - self.start) // self.batch)


class PartImages:
    """Archive images by id (ft_archive part_*.zip, raw OSV-5M bytes; one zip handle per worker).
    train: index (epoch, row) -> train.train_view seeded by (seed, epoch, rank[row]); else index
    row -> V3's eval crop (embed.clip_center_crop). Returns (image array, row)."""

    def __init__(self, parts: Path, where: dict[str, str], ids, size: int, train: bool,
                 seed: int = 0, ranks=None):
        missing = [i for i in ids if i not in where]
        if missing:
            raise SystemExit(f"{len(missing)} ids are not in the archive manifest, e.g. {missing[:3]}")
        self.parts, self.ids = Path(parts), list(ids)
        self.files = [where[i] for i in self.ids]
        self.ranks = None if ranks is None else np.asarray(ranks)
        self.size, self.train, self.seed, self._zips = size, train, seed, {}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key):
        from geoguessr_ai.embed import clip_center_crop
        from geoguessr_ai.train import to_tensor_array, train_view

        epoch, row = key if self.train else (0, key)
        f = self.files[row]
        zf = self._zips.get(f) or self._zips.setdefault(f, zipfile.ZipFile(self.parts / f))
        with Image.open(io.BytesIO(zf.read(f"{self.ids[row]}.jpg"))) as im:
            im = im.convert("RGB")
            x = (train_view(im, self.size, np.random.default_rng([self.seed, epoch, int(self.ranks[row])]))
                 if self.train else to_tensor_array(clip_center_crop(im, self.size)))
        return x, row


def archive_index(parts: Path, want_sha: str | None) -> dict[str, str]:
    man = Path(parts) / "manifest.csv"
    if not man.exists():
        raise SystemExit(f"{man} missing: stage the archive first")
    if want_sha is not None and file_sha256(man) != want_sha:
        raise SystemExit(f"{man}: sha256 is not the pre-dev config's archive manifest")
    m = pd.read_csv(man, dtype={"id": str, "part": str})
    return dict(zip(m["id"], m["part"]))


# ----------------------------------------------------------------- model (torch)


def build_model(cfg: dict, model_name: str, head: np.ndarray, kind: str, seed: int, smoke: bool,
                device):
    """(model, trainable named params, image size). kind control: no LoRA, encoder frozen.
    train.freeze_head: the head is not trained."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    from geoguessr_ai.baseline_clip import as_embedding
    from geoguessr_ai.embed import load_encoder
    from geoguessr_ai.lora import add_lora

    if smoke:
        torch.manual_seed(0)  # the tiny random encoder must be the same in every process
    frozen_head = cfg["train"].get("freeze_head", False)
    if frozen_head and kind == "control":
        raise SystemExit("a control run with a frozen head trains nothing")
    vision, size, dim = load_encoder(model_name, smoke, device)
    torch.manual_seed(seed)  # LoRA A init
    lo = cfg["lora"]
    if kind == "control":
        for p in vision.parameters():
            p.requires_grad_(False)
    else:
        add_lora(vision, lo["rank"], lo["alpha"], tuple(lo["targets"]), lo["blocks"])
    scale = cfg["train"]["logit_scale"]

    class Geo(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision, self.control = vision, kind == "control"
            self.head = nn.Parameter(torch.from_numpy(np.asarray(head, dtype=np.float32)).to(device),
                                     requires_grad=not frozen_head)

        def embed(self, px):
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.control):
                out = as_embedding(self.vision(pixel_values=px), "image", dim)
            return F.normalize(out.float(), dim=-1)

        def logits(self, e):  # geo_eval.probe_logits' cosine head
            return scale * e @ F.normalize(self.head, dim=-1).T

        def forward(self, px):
            return self.logits(self.embed(px))

    model = Geo()
    model.vision.train()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    return model, trainable, size


def make_loader(ds, sampler, workers: int):
    import torch
    from torch.utils.data import DataLoader

    # an explicit generator: creating a loader never draws from the global torch RNG
    return DataLoader(ds, batch_sampler=sampler, num_workers=workers,
                      generator=torch.Generator().manual_seed(0),
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


class Eval:
    """Top-5 cells (+ embeddings) of a model over a list of rows of one split."""

    def __init__(self, name: str, df: pd.DataFrame, ds, batch: int, workers: int):
        self.name, self.df, self.ds, self.batch, self.workers = name, df, ds, batch, workers

    def run(self, model, device, amp: bool, keep_emb: bool = False):
        import torch

        from geoguessr_ai.phase3_eval import top_cells

        was = model.training
        model.eval()
        tops, embs = [], []
        sampler = [list(range(s, min(s + self.batch, len(self.ds))))
                   for s in range(0, len(self.ds), self.batch)]
        with torch.no_grad():
            for x, _ in make_loader(self.ds, sampler, self.workers):
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    e = model.embed(x.to(device, non_blocking=True))
                tops.append(top_cells(model.logits(e).float().cpu().numpy(), 5))
                if keep_emb:
                    embs.append(e.float().cpu().numpy())
        model.train(was)
        return np.concatenate(tops), (np.concatenate(embs) if keep_emb else None)


def score(df: pd.DataFrame, name: str, top: np.ndarray, geo: dict, n_boot: int, seed: int):
    """GeoScore summary (creator clusters for dev) + per-image predictions."""
    from geoguessr_ai import geo_eval as g
    from geoguessr_ai import geocells as gc

    cent = geo["cent"]
    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    km = g.haversine(lat, lon, *cent[top[:, 0]].T)
    res = g.summarize(g.per_image_geo(km), km, g.clusters_of(name, df), n_boot, seed)
    res["top1_cell_acc"] = float((top[:, 0] == gc.assign(geo["tree"], lat, lon)).mean())
    pred = pd.DataFrame({"id": df["id"].to_numpy(), **{f"c{j}": top[:, j] for j in range(top.shape[1])},
                         "pred_lat": cent[top[:, 0], 0], "pred_lon": cent[top[:, 0], 1], "km": km})
    return res, pred


# ----------------------------------------------------------------- checkpoint + hub


def save_atomic(obj, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def weights(trainable: dict) -> dict:
    return {n: p.detach().cpu().clone() for n, p in trainable.items()}


class Runs:
    """The private HF runs repo (None: local only, e.g. --smoke and tests)."""

    def __init__(self, repo: str | None, run_id: str, log_path: Path):
        self.hub = None
        if repo:
            from geoguessr_ai.ft_archive import make_hub

            self.hub, self.token = make_hub(repo, f"{run_id}/", 5, 30, log_path)
            self.hub.open()

    def put(self, local: Path, name: str) -> None:
        if self.hub:
            self.hub.put(local, name)

    def get_ckpt(self, dst: Path) -> Path | None:
        if not self.hub:
            return None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(self.hub.call("download ckpt", lambda: hf_hub_download(
                self.hub.repo, self.hub.prefix + "ckpt/last.pt", repo_type="dataset",
                token=self.token, local_dir=dst)))
        except EntryNotFoundError:
            return None


def load_resume(local: Path, runs: Runs, key: str):
    """The newer of the local checkpoint (seeded by SEED_FROM) and the runs repo's; refuses a
    checkpoint of another run key."""
    import torch

    cands = [p for p in (local if local.exists() else None,
                         runs.get_ckpt(local.parent / "hub")) if p is not None]
    states = [torch.load(p, weights_only=False) for p in cands]
    for s in states:
        if s["key"] != key:
            raise SystemExit(f"checkpoint run key {s['key']} != this run's {key}: settings, inputs or "
                             "code changed; start a fresh output for the new key")
    return max(states, key=lambda s: s["step"]) if states else None


# ----------------------------------------------------------------- train


class Log:
    def __init__(self, path: Path, keep_upto: int | None):
        self.path = path
        if keep_upto is not None and path.exists():  # drop what the killed session logged after its checkpoint
            rows = [json.loads(s) for s in path.read_text().splitlines() if s]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows if r.get("step", 0) <= keep_upto))
        elif keep_upto is None:
            path.write_text("")

    def __call__(self, **row) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def setup(args):
    """Everything both train and identity need: config, inputs, geo, device."""
    import torch

    from geoguessr_ai import geocells as gc
    from geoguessr_ai.phase3_eval import geo_of

    cfg = json.loads(Path(args.config).read_text())
    if args.backbone not in cfg["backbones"]:
        raise SystemExit(f"unknown backbone {args.backbone}")
    inputs = Path(args.inputs)
    shas = cfg["inputs"]
    if not args.smoke and None in (shas["bundle_manifest_sha256"], shas["archive_manifest_sha256"]):
        raise SystemExit("the pre-dev config has no bundle/archive manifest sha256 yet (draft)")
    verify_bundle(inputs, shas["bundle_manifest_sha256"])
    tree, gsha = gc.load(inputs / "cells_5000.json")
    if gsha != cfg["inputs"]["geocells_sha256"]:
        raise SystemExit(f"partition sha256 {gsha[:16]} is not the frozen one")
    where = archive_index(args.parts, cfg["inputs"]["archive_manifest_sha256"])
    want = device_for(args.job, args.backbone) if args.job is not None else None
    dev = args.device
    if dev == "auto":
        dev = want or "cuda:0"
    if want and dev != want and not args.smoke:
        raise SystemExit(f"job {args.job}: {args.backbone} runs on {want} (decision 7), not {dev}")
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    gpu = None
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        gpu = {"index": device.index, "name": p.name, "uuid": str(getattr(p, "uuid", ""))}
    return cfg, inputs, geo_of(tree, gsha, 5000), where, device, gpu


def eval_sets(cfg, inputs: Path, parts: Path, where: dict, size: int) -> dict[str, Eval]:
    """diag (2K, per epoch), inter (5K, at S/3 and 2S/3; the 2K are its first rows), dev (all of
    dev_unseen, the selection split), dev_seen (diagnostic)."""
    dev = read_csv(inputs / "dev.csv").set_index("id", drop=False)
    sub, t = cfg["evals"], cfg["train"]
    ids5k = dev_subset(dev["id"], sub["intermediate_n"], sub["subset_seed"])
    out = {}
    for name, df in (("diag", dev.loc[ids5k[: sub["diagnostic_n"]]]), ("inter", dev.loc[ids5k]),
                     ("dev", dev), ("dev_seen", read_csv(inputs / "dev_seen.csv"))):
        df = df.reset_index(drop=True)
        out[name] = Eval(name, df, PartImages(parts, where, df["id"], size, False),
                         t["eval_batch"], t["workers"])
    return out


def run_train(args) -> int:
    import torch
    import torch.nn.functional as F

    from geoguessr_ai import geocells as gc
    from geoguessr_ai.train import lr_factor

    cfg, inputs, geo, where, device, gpu = setup(args)
    job = job_of(cfg, args.job)
    sel_path = Path(args.selection) if isinstance(job["lr"], str) else None
    selection = json.loads(sel_path.read_text()) if sel_path else None
    commit = git_checks(Path(args.config), sel_path, args.smoke)
    lr = job_lr(job, args.backbone, selection)
    out = Path(args.out)
    ident = out / f"identity_{args.backbone}.json"
    if not (ident.exists() and json.loads(ident.read_text())["ok"]):
        raise SystemExit(f"{ident}: the zero-init identity check has not passed for {args.backbone}")

    t, sch = cfg["train"], schedule(cfg, job)
    head_file = inputs / "heads" / args.backbone / f"bal_{job['n']}_cell_zeroshot.npy"
    hashes = {"bundle": file_sha256(inputs / "MANIFEST.sha256"),
              "archive_manifest": file_sha256(Path(args.parts) / "manifest.csv"),
              "head": file_sha256(head_file), "geocells": geo["sha"]}
    key = run_key(cfg, job, args.backbone, lr, commit, hashes)
    run_id = f"k{job['k']}_{job['name']}_{args.backbone}"
    rdir = out / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "DONE").exists():
        done_key = json.loads((rdir / "result.json").read_text())["key"]
        if done_key != key:
            raise SystemExit(f"{run_id}: DONE with run key {done_key} != this run's {key}: use a fresh output")
        print(f"{run_id}: DONE already", flush=True)
        return 0
    runs = Runs(None if args.smoke else cfg["runs_repo"], run_id, rdir / "hub.log")

    train_df = read_csv(inputs / "train.csv")
    train_df = train_df[train_df["bal_rank"] < job["n"]].sort_values("bal_rank").reset_index(drop=True)
    if len(train_df) != job["n"]:
        raise SystemExit(f"train has {len(train_df)} rows with bal_rank < {job['n']}")
    labels = torch.from_numpy(gc.assign(geo["tree"], train_df["latitude"], train_df["longitude"])).long()

    seed, amp = job["seed"], device.type == "cuda"
    model, trainable, size = build_model(cfg, cfg["backbones"][args.backbone], np.load(head_file),
                                         job["kind"], seed, args.smoke, device)
    groups = [{"params": [p for n, p in trainable.items() if n != "head"], "lr": lr}]
    if "head" in trainable:
        groups.append({"params": [trainable["head"]], "lr": lr * t["head_lr_mult"]})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=t["weight_decay"], betas=tuple(t["adam_betas"]),
                            eps=t["adam_eps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, sch["total"], sch["warmup"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ckpt = rdir / "ckpt" / "last.pt"

    state = load_resume(ckpt, runs, key)
    step, acc = 0, {"loss_sum": 0.0, "steps": 0, "epoch": 0}
    if state:
        for n, p in trainable.items():
            p.data.copy_(state["weights"][n].to(p.device))
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        step, acc = state["step"], state["acc"]
        print(f"{run_id}: resumed at step {step} of {sch['total']}", flush=True)
    log = Log(rdir / "log.jsonl", step if state else None)
    meta = {"run_id": run_id, "key": key, "k": job["k"], "job": job, "backbone": args.backbone,
            "model": cfg["backbones"][args.backbone], "lr": lr,
            "head_lr": None if "head" not in trainable else lr * t["head_lr_mult"],
            "head_frozen": "head" not in trainable,
            "seed": seed, "n": job["n"], "kind": job["kind"], "steps": sch["total"],
            "images_seen": sch["total"] * sch["eff"], "code_commit": commit, "gpu": gpu, **hashes}
    if not state:
        log(step=0, event="start", **meta)
    else:
        log(step=step, event="resume", gpu=gpu)

    def save_ckpt():
        save_atomic({"key": key, "step": step, "weights": weights(trainable), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict(), "sched": sched.state_dict(), "acc": acc,
                     "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}}, ckpt)
        runs.put(ckpt, "ckpt/last.pt")
        runs.put(rdir / "log.jsonl", "log.jsonl")

    stop_file = Path(args.stop_file) if args.stop_file else None

    def stop_now() -> bool:
        return bool((stop_file and stop_file.exists())
                    or (args.stop_after_steps and step == args.stop_after_steps))

    def stopped() -> int:
        save_ckpt()
        print(f"RESUME NEEDED: {run_id} stopped at step {step} of {sch['total']}", flush=True)
        return RESUME_EXIT

    evs = eval_sets(cfg, inputs, Path(args.parts), where, size)
    n_boot, eseed = cfg["n_boot"], cfg["seed"]

    def quick_eval(kind: str, which: str):
        top, _ = evs[which].run(model, device, amp)
        res, _ = score(evs[which].df, "dev", top, geo, n_boot, eseed)  # "dev": creator clusters
        log(step=step, event="eval", kind=kind, split="dev_unseen", n=len(top),
            geoscore=res["geoscore"]["mean"], median_km=res["median_km"],
            top1_cell_acc=res["top1_cell_acc"], images_seen=step * sch["eff"])
        return res

    train_ds = PartImages(args.parts, where, train_df["id"], size, True, seed, train_df["bal_rank"])
    sampler = StreamSampler(job["n"], seed, step * sch["eff"], sch["total"] * sch["eff"], sch["micro"])
    batches = iter(make_loader(train_ds, sampler, t["workers"]))
    model.train()
    t0, n0 = time.time(), step
    while step < sch["total"]:
        if stop_now():
            return stopped()
        tot = 0.0
        for _ in range(sch["accum"]):
            x, rows = next(batches)
            y = labels[rows].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x.to(device, non_blocking=True))
            loss = F.cross_entropy(logits.float(), y, label_smoothing=t["label_smoothing"])
            if step == 0 and tot == 0.0:
                log(step=0, event="initial_loss", loss=loss.item())
            scaler.scale(loss / sch["accum"]).backward()
            tot += loss.item() / sch["accum"]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), t["grad_clip"])
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= before:  # an fp16 overflow skips the optimizer step: skip the scheduler too
            sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        acc["loss_sum"] += tot
        acc["steps"] += 1
        epoch = step * sch["eff"] // job["n"]
        if step % cfg["log_every_steps"] == 0:
            log(step=step, event="train", loss=tot, lr=opt.param_groups[0]["lr"],
                img_per_s=round((step - n0) * sch["eff"] / max(time.time() - t0, 1e-9), 2))
        if epoch > acc["epoch"]:  # an epoch boundary was crossed in this step
            log(step=step, event="epoch", epoch=epoch, train_loss=acc["loss_sum"] / acc["steps"])
            acc.update(loss_sum=0.0, steps=0, epoch=epoch)
            if job["k"] in cfg["evals"]["diagnostic_jobs"]:
                # diagnostic only: never used for selection, stopping or checkpoint choice
                quick_eval("diagnostic_epoch", "diag")
        if step in sch["intermediate"]:
            quick_eval("intermediate", "inter")
            save_atomic(weights(trainable), rdir / f"weights_{step}.pt")
            runs.put(rdir / f"weights_{step}.pt", f"weights_{step}.pt")
            save_ckpt()
        elif step % cfg["ckpt_every_steps"] == 0:
            save_ckpt()
    if stop_now():
        return stopped()
    save_ckpt()

    fin = rdir / "final"
    fin.mkdir(exist_ok=True)
    res = dict(meta)
    for split, name in (("dev", "dev"), ("dev_seen", "dev_seen")):
        top, emb = evs[name].run(model, device, amp, keep_emb=True)
        r, pred = score(evs[name].df, split, top, geo, n_boot, eseed)
        pred.to_csv(fin / f"{split}_predictions.csv", index=False)
        np.savez(fin / f"{split}_emb.npz", ids=evs[name].df["id"].to_numpy(), emb=emb)
        res[split] = r
        log(step=step, event="eval", kind="final", split="dev_unseen" if split == "dev" else split,
            n=len(top), geoscore=r["geoscore"]["mean"], median_km=r["median_km"],
            top1_cell_acc=r["top1_cell_acc"])
    save_atomic(weights(trainable), rdir / "weights_final.pt")
    res["weights_final_sha256"] = file_sha256(rdir / "weights_final.pt")
    (rdir / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    for f in ("weights_final.pt", "result.json", "log.jsonl", "final/dev_predictions.csv",
              "final/dev_seen_predictions.csv"):
        runs.put(rdir / f, f)
    (rdir / "DONE").write_text(res["weights_final_sha256"] + "\n")
    print(f"{run_id}: done; dev_unseen GeoScore {res['dev']['geoscore']['mean']:.1f}", flush=True)
    return 0


def run_identity(args) -> int:
    from geoguessr_ai.phase3_final import apply_head

    cfg, inputs, geo, where, device, gpu = setup(args)
    idn = cfg["identity"]
    z = np.load(inputs / "identity" / f"{args.backbone}_dev2k.npz")
    head = np.load(inputs / "heads" / args.backbone / f"bal_{idn['head_n']}_cell_zeroshot.npy")
    model, _, size = build_model(cfg, cfg["backbones"][args.backbone], head, "short", cfg["seed"],
                                 args.smoke, device)
    dev = read_csv(inputs / "dev.csv").set_index("id", drop=False)
    df = dev.loc[[str(i) for i in z["ids"]]].reset_index(drop=True)
    ev = Eval("dev", df, PartImages(args.parts, where, df["id"], size, False), cfg["train"]["eval_batch"],
              cfg["train"]["workers"])
    top, emb = ev.run(model, device, device.type == "cuda", keep_emb=True)
    stored = apply_head(z["emb"].astype(np.float32), head, cfg["train"]["logit_scale"])
    res = {"backbone": args.backbone, "n": len(top), "gpu": gpu,
           "top1_cell_same": float((top[:, 0] == stored[:, 0]).mean()),
           "cos_mean": float((emb * z["emb"]).sum(1).mean()), "min": idn["top1_same_min"]}
    res["ok"] = res["top1_cell_same"] >= idn["top1_same_min"]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"identity_{args.backbone}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0 if res["ok"] else 1


def run_stage(args) -> int:
    """Download every part + manifest.csv of the archive repo into --parts, sha256-checked."""
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(args.config).read_text())
    repo, parts = cfg["inputs"]["archive_repo"], Path(args.parts)
    tok = os.environ.get("HF_TOKEN")
    man = Path(hf_hub_download(repo, "manifest.csv", repo_type="dataset", token=tok, local_dir=parts))
    want = cfg["inputs"]["archive_manifest_sha256"]
    if want and file_sha256(man) != want:
        raise SystemExit(f"{repo} manifest.csv is not the pre-dev config's")
    m = pd.read_csv(man, dtype={"id": str, "part": str, "sha256": str})
    for part in sorted(m["part"].unique()):
        sha_f = Path(hf_hub_download(repo, Path(part).with_suffix(".sha256").name, repo_type="dataset",
                                     token=tok, local_dir=parts))
        p = Path(hf_hub_download(repo, part, repo_type="dataset", token=tok, local_dir=parts))
        if file_sha256(p) != sha_f.read_text().strip():
            raise SystemExit(f"{part}: sha256 differs after download")
        print(f"staged {part} ({p.stat().st_size / 1e9:.2f} GB)", flush=True)
    (parts / "STAGED").write_text(f"{len(m)} images\n")  # written last: a killed staging reruns
    return 0


def run_hfcheck(args) -> int:
    """The runs repo exists, is private, and HF_TOKEN can write to it."""
    cfg = json.loads(Path(args.config).read_text())
    from geoguessr_ai.ft_archive import make_hub

    hub, _ = make_hub(cfg["runs_repo"], "preflight/", 3, 5, None)
    hub.open()  # create private / refuse a public repo
    f = Path(args.out) / "hf_write_check.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}.txt"
    f.write_text("V4 preflight write check\n")
    hub.put(f, name)
    hub.call("delete", lambda: hub.api.delete_file(hub.prefix + name, hub.repo, repo_type="dataset",
                                                   commit_message="preflight write check"))
    print(f"{cfg['runs_repo']}: private, HF_TOKEN can write (put + delete {name})", flush=True)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("train", "identity", "stage", "hfcheck"))
    ap.add_argument("--config", default="configs/v4_predev.json")
    ap.add_argument("--selection", default="configs/v4_selection.json")
    ap.add_argument("--job", type=int)
    ap.add_argument("--backbone")
    ap.add_argument("--inputs", help="the V4 inputs bundle (v4_inputs.py)")
    ap.add_argument("--parts", help="dir of the staged archive part_*.zip + manifest.csv")
    ap.add_argument("--out", default="v4_runs")
    ap.add_argument("--device", default="auto", help="auto = decision 7 for --job")
    ap.add_argument("--stop-file")
    ap.add_argument("--stop-after-steps", type=int, default=0, help=argparse.SUPPRESS)  # tests only
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random encoder, no git checks, no HF (CPU tests)")
    a = ap.parse_args(argv)
    if a.cmd == "train" and (a.job is None or not a.backbone):
        ap.error("train needs --job and --backbone")
    if a.cmd == "identity" and not a.backbone:
        ap.error("identity needs --backbone")
    return a


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    run = {"train": run_train, "identity": run_identity, "stage": run_stage, "hfcheck": run_hfcheck}[a.cmd]
    sys.exit(run(a))


if __name__ == "__main__":
    main()
