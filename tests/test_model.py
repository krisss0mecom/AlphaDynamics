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

