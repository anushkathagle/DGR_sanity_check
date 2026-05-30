"""
Projected Gradient Descent (PGD) perturbation step.

Restricted under the ℓ∞-norm with bound ε.
Used inside the PACOL outer loop (Algorithm 1, line 8).
"""

import torch


def pgd_step(
    x_adv:    torch.Tensor,          # current adversarial samples  (N, C, H, W)
    x_orig:   torch.Tensor,          # clean originals for projection
    grad:     torch.Tensor,          # gradient of objective w.r.t. x_adv
    alpha:    float,                  # step size
    epsilon:  float,                  # l-inf perturbation bound
    x_min:    float = None,           # valid range lower bound (None = no clip)
    x_max:    float = None,           # valid range upper bound (None = no clip)
) -> torch.Tensor:
    """
    One PGD update step (sign-gradient method, l-inf projection).

    Returns a new x_adv that is:
      1. Moved by α · sign(grad) in the gradient direction.
      2. Clipped to the ε-ball around x_orig  (l-inf constraint).
      3. Optionally clipped to [x_min, x_max]  (valid image range).

    Note: pass x_min=0.0, x_max=1.0 only when data is NOT mean/std
    normalised (i.e. raw pixel values in [0, 1]).  For normalised
    tensors the valid range is dataset-specific and the ε-ball clip
    is sufficient.
    """
    x_adv = x_adv.detach() + alpha * grad.sign()
    # Project back into ε-ball
    x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
    # Optional clip to valid image range
    if x_min is not None and x_max is not None:
        x_adv = x_adv.clamp(x_min, x_max)
    return x_adv.detach()
