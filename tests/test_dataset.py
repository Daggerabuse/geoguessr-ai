import numpy as np
import pandas as pd

from geoguessr_ai.train import GeoDataset


def _df(root):
    return pd.read_csv(root / "manifest.csv", dtype={"id": str})


def test_missing_images_are_skipped_and_counted(subset_root):
    df = _df(subset_root)
    train = df[df["split"] == "train"].reset_index(drop=True)
    (subset_root / train.loc[0, "path"]).unlink()
    ds = GeoDataset(subset_root, train, 32, train=True)
    assert ds.n_missing == 1
    assert len(ds) == len(train) - 1
    assert len(ds.df) == len(ds.labels) == len(ds)


def test_train_and_eval_item_shapes(subset_root):
    df = _df(subset_root)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    x, y = GeoDataset(subset_root, tr, 32, train=True)[0]
    assert x.shape == (3, 32, 32) and isinstance(y, int)
    x, _ = GeoDataset(subset_root, tr, 32, train=False)[0]
    assert x.shape == (1, 3, 32, 32)
    x, _ = GeoDataset(subset_root, tr, 32, train=False, tta=True)[0]
    assert x.shape == (3, 3, 32, 32)


def test_augmentation_changes_with_epoch_but_not_between_calls(subset_root):
    df = _df(subset_root)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    ds = GeoDataset(subset_root, tr, 32, train=True)
    ds.set_epoch(1)
    a, b = ds[0][0], ds[0][0]
    assert np.array_equal(a, b)
    ds.set_epoch(2)
    assert not np.array_equal(a, ds[0][0])
