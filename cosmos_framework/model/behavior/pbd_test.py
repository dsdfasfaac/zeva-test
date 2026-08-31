import torch

from cosmos_framework.model.behavior.pbd import (
    ActionPriorConfig,
    ActionPriorNetwork,
    BehaviorActionAdapter,
    BriefInteractionTrace,
    gaussian_prior_nll,
)


def test_pbd_shapes_nll_and_zero_projection() -> None:
    pbd = ActionPriorNetwork(ActionPriorConfig(horizon=32, action_dim=8))
    effect = torch.randn(3, 4, 128)
    valid = torch.tensor([[False, False, False, False], [False, True, True, True], [True, True, True, True]])
    mean, std = pbd(torch.randn(3, 256), torch.randn(3, 128), effect, valid)
    assert mean.shape == std.shape == (3, 32, 8)
    assert torch.isfinite(gaussian_prior_nll(torch.randn_like(mean), mean, std))
    adapter = BehaviorActionAdapter(action_dim=8, hidden_dim=16)
    adapter.init_weights()
    assert not hasattr(adapter, "residual_gate")
    assert torch.equal(adapter(mean, action_length=33), torch.zeros(3, 33, 16))


def test_zero_projection_has_direct_action_gradient() -> None:
    """The faithful zero projection must grow without a scalar-gate warmup."""
    adapter = BehaviorActionAdapter(action_dim=8, hidden_dim=16)
    adapter.init_weights()
    prior = torch.randn(2, 32, 8, requires_grad=True)
    target = torch.randn(2, 33, 16)
    loss = (adapter(prior, action_length=33) - target).square().mean()
    loss.backward()
    weight_grad = adapter.prior_to_action_embedding.weight.grad
    assert weight_grad is not None
    assert torch.count_nonzero(weight_grad) > 0


def test_brief_interaction_trace_keeps_legacy_tensor_path_exact() -> None:
    torch.manual_seed(7)
    pbd = ActionPriorNetwork(ActionPriorConfig(horizon=4, action_dim=2))
    global_behavior = torch.randn(2, 256)
    phase = torch.randn(2, 128)
    effects = torch.randn(2, 4, 128)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    legacy = pbd(global_behavior, phase, effects, valid)
    paper_named = pbd(
        global_behavior,
        phase,
        BriefInteractionTrace(effects=effects, valid=valid),
    )
    assert torch.equal(legacy[0], paper_named[0])
    assert torch.equal(legacy[1], paper_named[1])
