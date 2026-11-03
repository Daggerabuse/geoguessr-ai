import math

from geoguessr_ai.train import EarlyStopper, lr_factor


def test_lr_factor_warmup_then_cosine_to_zero():
    f = [lr_factor(s, 100, 10) for s in range(100)]
    assert math.isclose(f[0], 0.1)
    assert math.isclose(f[9], 1.0)
    assert f[10] == 1.0
    assert f[-1] < 0.01
    assert all(f[i] >= f[i + 1] for i in range(10, 99))  # monotone decay after warmup


def test_lr_factor_without_warmup_starts_at_one():
    assert lr_factor(0, 50, 0) == 1.0


def test_early_stopper_tracks_best_and_patience():
    es = EarlyStopper(2)
    assert es.update(0.5)  # first value is the best
    assert not es.update(0.4)
    assert not es.should_stop
    assert not es.update(0.4)
    assert es.should_stop
    assert es.update(0.9)  # improvement resets the counter
    assert not es.should_stop
