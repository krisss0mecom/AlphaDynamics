import pytest
import torch

from alphadynamics.models import AlphaDynamicsModel


def test_model_forward_backward_and_sample():
    model = AlphaDynamicsModel(n_osc=8, n_components=3, hidden=16, rk_steps=2)
    x = torch.randn(4, 5, 2)
    y = torch.randn(4, 5, 2)
    loss = model.nll(x, y)
    loss.backward()
    sample = model.sample_next(x)
    assert sample.shape == x.shape
    assert torch.isfinite(loss)
    assert torch.isfinite(sample).all()


def test_residue_ids_accepts_both_1d_and_2d_shapes():
    """Regression test: build_features must accept residue_ids in either
    (N,) form (one sequence broadcast across the batch) or (B, N) form
    (per-batch-item sequences). Reported by independent third-party test
    on 2026-05-01."""
    model = AlphaDynamicsModel(
        n_osc=8, n_components=3, hidden=16, rk_steps=2, use_sequence=True,
    )
    angles = torch.zeros(2, 5, 2)

    # 1-D: one sequence broadcast across the batch
    rid_1d = torch.tensor([0, 1, 2, 3, 4])
    log_pi, mu, kappa = model(angles, residue_ids=rid_1d)
    assert log_pi.shape == (2, 5, 3)
    assert mu.shape == (2, 5, 3, 2)
    assert kappa.shape == (2, 5, 3, 2)

    # 2-D: per-batch-item sequences (different chains in the batch)
    rid_2d = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    log_pi, mu, kappa = model(angles, residue_ids=rid_2d)
    assert log_pi.shape == (2, 5, 3)
    assert mu.shape == (2, 5, 3, 2)
    assert kappa.shape == (2, 5, 3, 2)
    assert torch.isfinite(log_pi).all()

    # Bad shape must raise
    with pytest.raises(ValueError, match="residue_ids must have shape"):
        model(angles, residue_ids=torch.zeros(2, 5, 1, dtype=torch.long))

