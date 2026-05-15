"""
Sanity-check experiment: apply PACOL attack to kuc2477 DGR.

Purpose
-------
Verify that PACOL causes measurable T0 accuracy drops on this weaker DGR
(2000-iter GAN default; no optimizer reset; eval() previous scholar).
If it works here but not on our improved DGR (PACOL_rebuild), that
confirms our implementation is genuinely more robust, not broken.

Paper settings (PACOL §4.3):
    generator_iterations = 8000  (MNIST)
    solver_iterations    = 5000  (MNIST)
    K=15, S=40, ε=16/255
    Target task  = task 1 (index 0)
    Non-target   = tasks 4 & 5 (indices 3 & 4)

Usage
-----
    python run_pacol_experiment.py --attack clean      --runs 3
    python run_pacol_experiment.py --attack white      --ratio 0.03 --runs 1
    python run_pacol_experiment.py --attack label_flip --ratio 0.03 --runs 1
    python run_pacol_experiment.py --attack gray       --ratio 0.03 --runs 3
    python run_pacol_experiment.py --attack black      --ratio 0.03 --runs 3
"""

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
from torch import optim, nn

from data import get_rotated_mnist_tasks, DATASET_CONFIGS
from dgr import Scholar
from models import WGAN, CNN
from train import train
import utils

sys.path.insert(0, os.path.dirname(__file__))
from attacks.pacol import PACOL, train_surrogate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PACOL paper settings
N_TASKS         = 5
TARGET_TASK_ID  = 1          # 1-indexed (task 1 = first rotation)
NONTARGET_IDS   = [4, 5]     # 1-indexed (tasks 4 and 5)
K               = 15
S               = 40
EPS             = 16.0 / 255.0
ALPHA           = 2 * EPS / S
AUX_SIZE        = 1000

# Model / training settings (paper §4.3 MNIST)
GEN_ITERS       = 8000
CLS_ITERS       = 5000
BATCH_SIZE      = 32
LR              = 1e-4

# MNIST test set size (used to cap evaluation; no need to expand test data)
MNIST_TEST_SIZE = 10000


def build_scholar(cuda):
    cfg = DATASET_CONFIGS['rmnist']
    cnn = CNN(
        image_size=cfg['size'],
        image_channel_size=cfg['channels'],
        classes=cfg['classes'],
        depth=5,
        channel_size=1024,
        reducing_layers=3,
    )
    wgan = WGAN(
        z_size=100,
        image_size=cfg['size'],
        image_channel_size=cfg['channels'],
        c_channel_size=64,
        g_channel_size=64,
    )
    scholar = Scholar('rmnist-dgr', generator=wgan, solver=cnn)
    utils.gaussian_intiailize(scholar, std=0.02)
    if cuda:
        scholar.cuda()
    return scholar


def evaluate_all_tasks(solver, test_datasets, cuda):
    accs = []
    for ds in test_datasets:
        acc = utils.validate(
            solver, ds,
            test_size=MNIST_TEST_SIZE,   # cap at 10k; test_datasets may be capacity-expanded
            cuda=cuda,
            verbose=False,
            collate_fn=utils.label_squeezing_collate_fn,
        )
        accs.append(acc)
    return accs


def single_run(attack, ratio, train_datasets, test_datasets, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cuda   = torch.cuda.is_available()
    scholar = build_scholar(cuda)

    pacol_attacker = None
    target_dataset = train_datasets[TARGET_TASK_ID - 1]  # 0-indexed

    if attack not in ('clean', 'label_flip'):
        aux_idx = np.random.default_rng(seed).choice(
            len(target_dataset), min(AUX_SIZE, len(target_dataset)), replace=False
        )
        aux_data = torch.utils.data.Subset(target_dataset, aux_idx)

        if attack == 'white':
            # White-box: attacker uses the CL solver.  The model_orig is updated
            # to θ_{τ+n−1} inside train.py just before craft_poison is called
            # for each non-target task, so we pass the initial model here as a
            # placeholder — it will be replaced before first use.
            atk_model = copy.deepcopy(scholar.solver).to(DEVICE)

        elif attack == 'gray':
            # Gray-box: same architecture as the solver, trained on auxiliary
            # target-task data (adversary has no access to the CL model).
            atk_model = CNN(
                image_size=DATASET_CONFIGS['rmnist']['size'],
                image_channel_size=DATASET_CONFIGS['rmnist']['channels'],
                classes=DATASET_CONFIGS['rmnist']['classes'],
                depth=5, channel_size=1024, reducing_layers=3,
            ).to(DEVICE)
            utils.gaussian_intiailize(atk_model, std=0.02)
            atk_model = train_surrogate(atk_model, aux_data, DEVICE)

        elif attack == 'black':
            # Black-box: different (smaller) architecture, trained on auxiliary data.
            atk_model = CNN(
                image_size=DATASET_CONFIGS['rmnist']['size'],
                image_channel_size=DATASET_CONFIGS['rmnist']['channels'],
                classes=DATASET_CONFIGS['rmnist']['classes'],
                depth=3, channel_size=256, reducing_layers=2,
            ).to(DEVICE)
            atk_model = train_surrogate(atk_model, aux_data, DEVICE)

        pacol_attacker = PACOL(
            atk_model, num_classes=10,
            epsilon=EPS, K=K, S=S, alpha=ALPHA,
            distance='l2', device=DEVICE,
        )

    train(
        scholar, train_datasets, test_datasets,
        replay_mode='generative-replay',
        generator_iterations=GEN_ITERS,
        solver_iterations=CLS_ITERS,
        importance_of_new_task=0.5,   # used only for task 1 (dynamic takes over after)
        dynamic_importance=True,      # use r=1/τ schedule (DGR paper eq. 2)
        batch_size=BATCH_SIZE,
        lr=LR, beta1=0.5, beta2=0.9,
        loss_log_interval=500,
        eval_log_interval=999999,   # suppress mid-training eval
        image_log_interval=999999,
        collate_fn=utils.label_squeezing_collate_fn,
        cuda=cuda,
        pacol_attacker=pacol_attacker,
        target_dataset=target_dataset,
        nontarget_task_ids=NONTARGET_IDS if attack != 'clean' else None,
        poison_ratio=ratio,
        attack_mode='label_flip' if attack == 'label_flip' else 'pacol',
        num_classes=10,
        seed=seed,
    )

    return evaluate_all_tasks(scholar.solver, test_datasets, cuda)


def run_experiment(attack, ratio, n_runs=3):
    capacity = BATCH_SIZE * max(GEN_ITERS, CLS_ITERS)
    # Capacity padding is only needed for training (to keep DataLoader from
    # exhausting the dataset mid-epoch). Test datasets use raw MNIST (10k each).
    train_datasets = get_rotated_mnist_tasks(train=True,  capacity=capacity)
    test_datasets  = get_rotated_mnist_tasks(train=False, capacity=None)

    all_accs = []
    for run in range(n_runs):
        print(f'\n[run {run+1}/{n_runs}]', flush=True)
        accs = single_run(attack, ratio, train_datasets, test_datasets, seed=run)
        all_accs.append(accs)
        row = ' | '.join(f'T{i+1}={a:.1%}' for i, a in enumerate(accs))
        print(f'  {row}', flush=True)

    all_accs = np.array(all_accs) * 100
    means = all_accs.mean(axis=0)
    stds  = all_accs.std(axis=0)
    print('\nFinal results:')
    print(' | '.join(f'{m:.2f}±{s:.2f}' for m, s in zip(means, stds)))
    return means, stds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack',  default='clean',
                        choices=['clean', 'white', 'gray', 'black', 'label_flip'])
    parser.add_argument('--ratio',   default=0.03, type=float)
    parser.add_argument('--runs',    default=3,    type=int)
    args = parser.parse_args()

    print(f'\n[DGR sanity check | rmnist | {args.attack} | {args.ratio:.0%}]')
    run_experiment(args.attack, args.ratio, n_runs=args.runs)
