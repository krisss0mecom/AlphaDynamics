import numpy as np

from alphadynamics.metrics import canonical_jsd, visual_jsd


def test_canonical_metadata_has_no_leakage():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 0.2, size=(200, 4, 2)).astype(np.float32)
    val = rng.normal(0, 0.2, size=(100, 4, 2)).astype(np.float32)
    rollout = rng.normal(0, 0.2, size=(100, 4, 2)).astype(np.float32)
    rep = canonical_jsd(rollout, val, n_bins=24)
    assert rep.mode == "canonical"
    assert rep.gt_source == "held_out_val"
    assert rep.smoothing == "none"
    assert rep.includes_train_in_target is False
    assert rep.n_bins == 24
    vis = visual_jsd(rollout, train, val, n_bins=24)
    assert vis.mode == "visual"
    assert vis.includes_train_in_target is True


def test_jsd_detects_bad_rollout():
    rng = np.random.default_rng(1)
    val = rng.normal(0, 0.15, size=(400, 2, 2)).astype(np.float32)
    close = rng.normal(0, 0.15, size=(400, 2, 2)).astype(np.float32)
    bad = rng.normal(2.0, 0.15, size=(400, 2, 2)).astype(np.float32)
    assert canonical_jsd(close, val).mean_jsd < canonical_jsd(bad, val).mean_jsd

