import numpy as np
import pandas as pd

from geoguessr_ai.zip_audit import randomness


def _meta(grouped: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 6000
    seq = np.repeat(np.arange(n // 3), 3)  # 3 frames per sequence
    df = pd.DataFrame({"id": np.arange(n).astype(str), "sequence": seq.astype(str),
                       "creator_id": (seq // 20).astype(str),
                       "country": np.array(list("ABCDEF"))[seq % 6]})
    if grouped:  # zips filled country by country, sequences kept together
        df = df.sort_values(["country", "sequence"])
    else:
        df = df.sample(frac=1.0, random_state=0)
    df["shard"] = np.arange(n) // 500
    df["pos"] = np.arange(n)
    return df.reset_index(drop=True).iloc[rng.permutation(n)]  # row order must not matter


def test_random_zips_look_like_the_baseline_and_grouped_zips_do_not():
    rand, grp = randomness(_meta(False), 3, 1), randomness(_meta(True), 3, 1)
    for k, v in rand.items():  # within a small margin of the baseline range
        span = v["random_max"] - v["random_min"]
        assert v["random_min"] - span - 0.02 <= v["observed"] <= v["random_max"] + span + 0.02, k
    assert grp["country_tvd_mean"]["observed"] > 0.5 > grp["country_tvd_mean"]["random_max"]
    assert grp["consecutive_same_sequence"]["observed"] > 0.5
    m = grp["multi_frame_sequences_in_one_zip"]
    assert m["observed"] > 0.95 and m["random_max"] < 0.2  # a few sequences straddle zip borders
