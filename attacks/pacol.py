"""
PACOL – Poisoning Attack against Continual Learners (Algorithm 1).

Core idea
---------
The adversary wants to craft perturbed inputs X^adv_{τ+n} (for non-targeted
task τ+n) such that the gradient update they produce on the model mimics
the gradient update that *label-flipped* samples from the targeted task τ
would produce.  This forces the model to "forget" task τ without ever
injecting mislabelled samples (clean-label attack).

Algorithm 1 (from the paper)
-----------------------------
Input: θ, D̃_τ = {X_τ, Y^adv_τ}, D̃_{τ+n} = {X_{τ+n}, Y_{τ+n}}, ε, K, S, α

Initiate: θ₀ = θ,  X^adv = X_{τ+n}

while k ≤ K:
    Δ^lf = ∇_θ L( f(X_τ; θ_k), Y^adv_τ )          # label-flipped gradient
    s = 0
    while s ≤ S:
        Δ^adv = ∇_θ L( f(X^adv; θ_k), Y_{τ+n} )    # poisoning gradient
        H = dist( Δ^lf, Δ^adv )                      # gradient matching loss
        X^adv ← clip_ε( X^adv + α · sign(∇_X H) )   # PGD update on inputs
        s += 1
    θ_{k+1} ← Adam( L( f(X^adv; θ_k), Y_{τ+n} ) )  # update model params
    k += 1

Output: D̃_{τ+n} = {X^adv, Y_{τ+n}}

Distance metrics
----------------
    'l2'     : ‖ Δ^adv − Δ^lf ‖²  (MNIST variants per paper)
    'cosine' : − cosine_sim( Δ^adv, Δ^lf )  (SVHN / CIFAR per paper)

Attack boxes
------------
    white-box : uses the actual CL model and actual target task data
    gray-box  : uses a surrogate model (same arch, different init) +
                auxiliary target-task data
    black-box : uses a surrogate model (different arch) +
                auxiliary target-task data
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.label_flip import create_label_flip_poison
from attacks.pgd import pgd_step


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def _l2_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """‖p − q‖²  (both vectors flattened)."""
    return ((p - q) ** 2).sum()


def _neg_cosine(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity (minimising = maximising alignment)."""
    return -F.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0))


_DIST_FNS = {'l2': _l2_distance, 'cosine': _neg_cosine}


# ---------------------------------------------------------------------------
# Helper: flatten all parameter gradients into a single vector
# ---------------------------------------------------------------------------

def _flat_grad(
    loss:          torch.Tensor,
    params,
    create_graph:  bool = False,
    retain_graph:  bool = False,
) -> torch.Tensor:
    """
    Compute ∇_θ loss and return as a flat 1-D tensor.
    Uses create_graph=True when we need to differentiate *through* this
    gradient (i.e., for the poisoning gradient ∇_X H).
    """
    grads = torch.autograd.grad(
        loss, params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    # Replace None grads (unused params) with zeros
    flat = []
    for g, p in zip(grads, params):
        flat.append(g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
    return torch.cat(flat)


# ---------------------------------------------------------------------------
# Main PACOL class
# ---------------------------------------------------------------------------

class PACOL:
    """
    Parameters
    ----------
    model        : the model to attack (or surrogate for gray/black-box)
    num_classes  : number of classes in the targeted task
    epsilon      : ℓ∞ perturbation budget
    K            : outer loop iterations (model parameter updates)
    S            : inner PGD iterations
    alpha        : PGD step size
    distance     : 'l2' or 'cosine'
    device       : torch.device
    lr           : Adam learning rate for outer-loop model updates
    """

    def __init__(
        self,
        model:       nn.Module,
        num_classes: int,
        epsilon:     float,
        K:           int   = 10,
        S:           int   = 40,
        alpha:       float = None,
        distance:    str   = 'cosine',
        device:      torch.device = None,
        lr:          float = 1e-4,
    ):
        if distance not in _DIST_FNS:
            raise ValueError(f"distance must be one of {list(_DIST_FNS)}, got '{distance}'")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if alpha is None:
            alpha = 2.0 * epsilon / max(S, 1)

        self.model_orig  = model          # keep original reference
        self.num_classes = num_classes
        self.epsilon     = epsilon
        self.K           = K
        self.S           = S
        self.alpha       = alpha
        self.dist_fn     = _DIST_FNS[distance]
        self.device      = device
        self.lr          = lr

    # ------------------------------------------------------------------

    def craft_poison(
        self,
        target_data:      torch.utils.data.Dataset,
        nontarget_x:      torch.Tensor,
        nontarget_y:      torch.Tensor,
        seed:             int   = 0,
        x_min:            float = None,
        x_max:            float = None,
        craft_batch_size: int   = 128,
        verbose:          bool  = True,
    ) -> torch.Tensor:
        """
        Run Algorithm 1 and return adversarial inputs X^adv_{τ+n}.

        Structure faithful to Algorithm 1:
          - Target label-flip data is sampled once before the K outer loop.
          - The K outer loop wraps the mini-batch iteration so the model
            evolves exactly K times (one Adam step per outer iteration),
            not K × num_batches times as in the old per-batch structure.
          - Inner PGD sweeps all non-target mini-batches per outer iteration.
        """
        model  = copy.deepcopy(self.model_orig).to(self.device)
        model.train()

        x_orig = nontarget_x.clone()   # CPU; reference for ε-ball projection
        x_adv  = nontarget_x.clone()   # output accumulator (CPU)

        opt    = torch.optim.Adam(model.parameters(), lr=self.lr)
        params = list(model.parameters())
        n      = len(nontarget_x)

        # Sample target (label-flipped) data once — held fixed across all K steps.
        # A batch of craft_batch_size samples gives a stable gradient signal.
        n_target = min(craft_batch_size, len(target_data))
        x_t, y_t_adv = create_label_flip_poison(
            target_data,
            n_poison=n_target,
            num_classes=self.num_classes,
            seed=seed,
        )
        x_t     = x_t.to(self.device)
        y_t_adv = y_t_adv.to(self.device)

        for k in range(self.K):
            # ── Label-flipped gradient (once per outer iteration) ──────────
            model.zero_grad()
            loss_lf = F.cross_entropy(model(x_t), y_t_adv)
            grad_lf = _flat_grad(loss_lf, params, create_graph=False).detach()

            # ── Inner PGD: sweep all non-target mini-batches ───────────────
            h_first_start = None   # H at PGD step 0 of first batch (for diag)
            h_first_end   = None   # H at PGD step S-1 of first batch

            for batch_start in range(0, n, craft_batch_size):
                batch_end = min(batch_start + craft_batch_size, n)

                xb_adv  = x_adv[batch_start:batch_end].to(self.device)
                xb_orig = x_orig[batch_start:batch_end].to(self.device)
                yb_nt   = nontarget_y[batch_start:batch_end].to(self.device)

                for s in range(self.S):
                    xb_adv = xb_adv.detach().requires_grad_(True)

                    model.zero_grad()
                    loss_adv = F.cross_entropy(model(xb_adv), yb_nt)
                    grad_adv = _flat_grad(
                        loss_adv, params,
                        create_graph=True,
                        retain_graph=True,
                    )

                    H      = self.dist_fn(grad_adv, grad_lf)
                    grad_X = torch.autograd.grad(H, xb_adv)[0]

                    # Record H for first batch only (representative sample)
                    if batch_start == 0:
                        if s == 0:
                            h_first_start = H.item()
                        if s == self.S - 1:
                            h_first_end = H.item()

                    with torch.no_grad():
                        xb_adv = pgd_step(xb_adv, xb_orig, grad_X,
                                          self.alpha, self.epsilon, x_min, x_max)

                x_adv[batch_start:batch_end] = xb_adv.detach().cpu()

            # ── One model update per outer iteration (Algorithm 1 step 11) ─
            # Uses the first mini-batch of the now-updated adversarial data.
            mb_end = min(craft_batch_size, n)
            xb_update = x_adv[:mb_end].to(self.device)
            yb_update = nontarget_y[:mb_end].to(self.device)

            opt.zero_grad()
            loss_step = F.cross_entropy(model(xb_update.detach()), yb_update)
            loss_step.backward()
            opt.step()

            if verbose and h_first_start is not None:
                pert = (x_adv - x_orig).abs()
                print(
                    f'  [PACOL k={k+1:2d}/{self.K}] '
                    f'H {h_first_start:.4f}→{h_first_end:.4f}  '
                    f'pert max={pert.max():.4f} mean={pert.mean():.5f}',
                    flush=True,
                )

        return x_adv

    # ------------------------------------------------------------------
    # Convenience wrappers for box variants
    # ------------------------------------------------------------------

    @classmethod
    def white_box(
        cls,
        model:        nn.Module,
        num_classes:  int,
        epsilon:      float,
        K: int, S: int, alpha: float, distance: str, device: torch.device,
    ) -> 'PACOL':
        """
        White-box: use the actual CL model.
        Caller passes the real model; no surrogate needed.
        """
        return cls(model, num_classes, epsilon, K, S, alpha, distance, device)

    @classmethod
    def gray_box(
        cls,
        surrogate:    nn.Module,    # same arch, different init, pre-trained on aux data
        num_classes:  int,
        epsilon:      float,
        K: int, S: int, alpha: float, distance: str, device: torch.device,
    ) -> 'PACOL':
        return cls(surrogate, num_classes, epsilon, K, S, alpha, distance, device)

    @classmethod
    def black_box(
        cls,
        surrogate:    nn.Module,    # different arch, pre-trained on aux data
        num_classes:  int,
        epsilon:      float,
        K: int, S: int, alpha: float, distance: str, device: torch.device,
    ) -> 'PACOL':
        return cls(surrogate, num_classes, epsilon, K, S, alpha, distance, device)


# ---------------------------------------------------------------------------
# Surrogate training helper (gray/black-box)
# ---------------------------------------------------------------------------

def train_surrogate(
    surrogate:   nn.Module,
    aux_dataset: torch.utils.data.Dataset,
    device:      torch.device,
    n_iters:     int   = 2000,
    lr:          float = 1e-4,
    batch_size:  int   = 128,
) -> nn.Module:
    """
    Train a surrogate model on auxiliary target-task data.
    Used to set up gray-box and black-box attacks.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)
    surrogate = surrogate.to(device).train()
    opt = torch.optim.Adam(surrogate.parameters(), lr=lr)

    it = _infinite(loader)
    for _ in range(n_iters):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(surrogate(x), y).backward()
        opt.step()
    return surrogate


def _infinite(loader):
    while True:
        for b in loader:
            yield b
