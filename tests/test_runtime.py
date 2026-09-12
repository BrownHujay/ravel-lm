import torch

from ravel_lm.runtime import _global_l2_clip_


def test_global_l2_clip_matches_pytorch():
    torch.manual_seed(19)
    reference = [torch.nn.Parameter(torch.randn(7, 5)), torch.nn.Parameter(torch.randn(11))]
    actual = [torch.nn.Parameter(param.detach().clone()) for param in reference]
    for ref, got in zip(reference, actual):
        grad = torch.randn_like(ref)
        ref.grad = grad.clone()
        got.grad = grad.clone()

    expected_norm = torch.nn.utils.clip_grad_norm_(reference, 0.37, foreach=False)
    actual_norm = _global_l2_clip_(tuple(param.grad for param in actual), 0.37)

    torch.testing.assert_close(actual_norm, expected_norm)
    for ref, got in zip(reference, actual):
        torch.testing.assert_close(got.grad, ref.grad)
