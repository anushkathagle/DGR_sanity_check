"""
Label-Flipping Attack for continual learners.

Injects mislabelled samples from the targeted task into non-targeted tasks
to cause catastrophic forgetting of the targeted task.

Two strategies (paper §3.2):
  - Random flip: randomly select samples and flip their labels.
  - Adversarial flip: uses the adversarial label-flip formulation
    (we implement random flip here, as used in the paper's experiments).

Label flipping rule (paper eq. 3):
  binary        : Y_adv = 1 − Y         (equivalent to −1 × Y for {−1,1})
  multi-class   : Y_adv = (Y + z) % C   where z ~ Uniform{1, …, C−1}
"""

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Tuple


def flip_labels(
    labels:      torch.Tensor,
    num_classes: int,
    rng:         np.random.Generator = None,
) -> torch.Tensor:
    """
    Return adversarial labels for a batch.

    Parameters
    ----------
    labels      : (N,) int tensor  – original class indices
    num_classes : number of classes in the targeted task
    rng         : numpy RNG (optional, for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()

    labels_adv = labels.clone()
    if num_classes == 2:
        labels_adv = 1 - labels_adv
    else:
        z = torch.from_numpy(
            rng.integers(1, num_classes, size=len(labels))
        ).to(labels.device)
        labels_adv = (labels + z) % num_classes

    return labels_adv.long()


def create_label_flip_poison(
    target_dataset:  Dataset,
    n_poison:        int,
    num_classes:     int,
    seed:            int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `n_poison` examples from `target_dataset`, flip their labels,
    and return (adv_inputs, adv_labels).

    These tensors are then injected into non-targeted task data via
    data.task_splitter.inject_label_flip_poison.
    """
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(target_dataset), size=n_poison, replace=(n_poison > len(target_dataset)))

    inputs, labels = [], []
    for i in idxs:
        x, y = target_dataset[int(i)]
        inputs.append(x)
        labels.append(y)

    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    adv_labels = flip_labels(labels, num_classes, rng)

    return inputs, adv_labels
