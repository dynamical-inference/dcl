import pytest
import torch
from dcl.criterions.contrastive import euclidean_similarity
from dcl.criterions.contrastive import infonce_full_denominator
from dcl.criterions.contrastive import infonce_full_denominator_old


def setup_data(num_positives, num_negatives, dim):
    # set seed
    torch.manual_seed(42)
    ref = torch.randn(num_positives, dim).float()
    pos = torch.randn(num_positives, dim).float()
    neg = torch.randn(num_negatives, dim).float()
    return ref, pos, neg


def setup_dist(ref, pos, neg):
    pos_dist, neg_dist = euclidean_similarity(ref, pos, neg)
    pos_dist.requires_grad_(True)
    neg_dist.requires_grad_(True)
    return pos_dist, neg_dist


def _compute_grads(output, inputs):
    for input_ in inputs:
        input_.grad = None
        assert input_.requires_grad
    output.backward()
    return [input_.grad for input_ in inputs]


def assert_loss(
    loss,
    align,
    uniform,
    ref_loss,
    ref_align,
    ref_uniform,
):
    assert torch.allclose(loss, ref_loss)
    assert torch.allclose(align, ref_align)
    assert torch.allclose(uniform, ref_uniform)


def assert_grads(
    grads,
    ref_grads,
):
    for grad, ref_grad in zip(grads, ref_grads):
        assert grad is not None
        assert ref_grad is not None
        assert torch.allclose(grad, ref_grad)


def _test_loss_impl(num_positives, num_negatives, dim, ref_fn, loss_fn):

    ref, pos, neg = setup_data(num_positives, num_negatives, dim)

    pos_dist, neg_dist = setup_dist(ref, pos, neg)
    ref_loss, ref_align, ref_uniform = ref_fn(pos_dist, neg_dist)
    ref_grads = _compute_grads(ref_loss, [pos_dist, neg_dist])

    pos_dist, neg_dist = setup_dist(ref, pos, neg)
    loss, align, uniform = loss_fn(pos_dist, neg_dist)
    grads = _compute_grads(loss, [pos_dist, neg_dist])

    assert torch.allclose(loss, ref_loss)
    assert not torch.allclose(align, ref_align)
    assert not torch.allclose(uniform, ref_uniform)
    assert_grads(grads, ref_grads)


@pytest.mark.parametrize("num_samples", [(100, 100), (10, 1000), (1000, 10)])
@pytest.mark.parametrize("dim", [5, 10])
def test_infonce_full_denominator(num_samples, dim):
    num_positives, num_negatives = num_samples
    _test_loss_impl(
        dim=dim,
        num_positives=num_positives,
        num_negatives=num_negatives,
        ref_fn=infonce_full_denominator_old,
        loss_fn=infonce_full_denominator,
    )
