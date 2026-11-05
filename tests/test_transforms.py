import numpy as np
from PIL import Image

from geoguessr_ai import train as t
from tests.conftest import gradient_image


def test_train_view_shape_and_dtype():
    rng = np.random.default_rng(0)
    x = t.train_view(gradient_image(), 224, rng)
    assert x.shape == (3, 224, 224)
    assert x.dtype == np.float32


def test_train_view_never_flips_horizontally():
    # With jitter/blur/erasing off, a left-dark/right-bright gradient must stay increasing.
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = t.train_view(gradient_image(), 224, rng, jitter=False, erase_p=0.0, blur_p=0.0)
        row = x[0, 112]
        assert row[-1] > row[0]


def test_train_view_is_deterministic_per_seed_epoch_index():
    im = gradient_image()
    a = t.train_view(im, 64, np.random.default_rng([42, 1, 7]))
    b = t.train_view(im, 64, np.random.default_rng([42, 1, 7]))
    c = t.train_view(im, 64, np.random.default_rng([42, 2, 7]))
    assert (a == b).all()
    assert not (a == c).all()


def test_eval_views_center_and_tta():
    im = gradient_image()
    single = t.eval_views(im, 224, tta=False)
    three = t.eval_views(im, 224, tta=True)
    assert len(single) == 1 and len(three) == 3
    assert (three[1] == single[0]).all()  # the middle TTA crop is the center crop
    assert not (three[0] == three[2]).all()
    # left < center < right on a left-to-right gradient
    assert three[0][0, 100, 0] < three[1][0, 100, 0] < three[2][0, 100, 0]


def test_eval_views_portrait_and_square_inputs():
    portrait = Image.fromarray(np.zeros((400, 200, 3), np.uint8))
    assert len(t.eval_views(portrait, 64, tta=True)) == 3
    assert t.eval_views(portrait, 64, tta=False)[0].shape == (3, 64, 64)
    square = Image.fromarray(np.zeros((64, 64, 3), np.uint8))
    assert len(t.eval_views(square, 32, tta=True)) == 3
    t.train_view(square, 32, np.random.default_rng(0))  # must not raise


def test_eval_views_are_the_normalized_eval_crops():
    from geoguessr_ai.baseline_clip import eval_crops

    im = gradient_image()
    for tta in (False, True):
        crops = eval_crops(im, 224, tta)
        views = t.eval_views(im, 224, tta)
        assert [c.size for c in crops] == [(224, 224)] * len(views)
        for c, v in zip(crops, views):
            assert (t.to_tensor_array(c) == v).all()
