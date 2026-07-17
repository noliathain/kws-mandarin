import torch
import torch.nn.functional as F

from kws_mandarin.decode import ctc_forward_logprob
from kws_mandarin.loss import CTCLoss


def test_ctc_loss_shapes_and_finite():
    torch.manual_seed(0)
    B, T, V = 4, 30, 20
    logits = torch.randn(B, T, V)
    targets = torch.randint(1, V, (B, 7))
    input_lengths = torch.full((B,), T, dtype=torch.long)
    target_lengths = torch.full((B,), 7, dtype=torch.long)
    loss = CTCLoss()(logits, targets, input_lengths, target_lengths)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_forward_logprob_matches_torch_ctc():
    # Our exact CTC forward must equal -F.ctc_loss (per sample, unnormalized).
    torch.manual_seed(1)
    T, V, U = 15, 8, 4
    log_probs = torch.randn(T, V).log_softmax(-1)
    target = torch.randint(1, V, (U,))

    mine = ctc_forward_logprob(log_probs, target.tolist(), blank=0)
    ref = F.ctc_loss(
        log_probs.unsqueeze(1),                       # (T, 1, V)
        target.unsqueeze(0),                          # (1, U)
        torch.tensor([T]),
        torch.tensor([U]),
        blank=0,
        reduction="none",
        zero_infinity=True,
    )[0]
    assert torch.allclose(mine, -ref, atol=1e-4), f"{mine.item()} vs {-ref.item()}"


def test_ctc_loss_backward():
    logits = torch.randn(2, 20, 12, requires_grad=True)
    targets = torch.randint(1, 12, (2, 5))
    il = torch.full((2,), 20, dtype=torch.long)
    tl = torch.full((2,), 5, dtype=torch.long)
    CTCLoss()(logits, targets, il, tl).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
