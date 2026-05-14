"""
Run this ONCE in Colab to apply all changes on top of the kuc2477 base repo.

Usage (in Colab, after cloning kuc2477):
    !git clone https://github.com/kuc2477/pytorch-deep-generative-replay /content/DGR_sanity_check
    %cd /content/DGR_sanity_check
    # paste and run this file, OR:
    # !python setup_colab.py
"""

import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. Fix gan.py ──────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'gan.py'), 'r') as f:
    content = f.read()
content = content.replace(
    'from torch import nn\nfrom torch.nn import functional as F',
    'import torch\nfrom torch import nn\nfrom torch.nn import functional as F'
)
content = content.replace('return F.sigmoid(g)', 'return torch.sigmoid(g)')
with open(os.path.join(BASE, 'gan.py'), 'w') as f:
    f.write(content)
print("✓ gan.py fixed")

# ── 2. Fix utils.py ───────────────────────────────────────────────────────
with open(os.path.join(BASE, 'utils.py'), 'r') as f:
    content = f.read()
content = content.replace(
    'from torch.autograd import Variable\n', ''
)
content = content.replace(
    '        data = Variable(data).cuda() if cuda else Variable(data)\n'
    '        labels = Variable(labels).cuda() if cuda else Variable(labels)',
    '        data = data.cuda() if cuda else data\n'
    '        labels = labels.cuda() if cuda else labels'
)
content = content.replace(
    "        total_correct += (predicted == labels).sum().data[0]",
    "        total_correct += (predicted == labels).sum().item()"
)
content = content.replace('nn.init.xavier_normal(p)', 'nn.init.xavier_normal_(p)')
content = content.replace('nn.init.normal(p, std=std)', 'nn.init.normal_(p, std=std)')
content = content.replace('nn.init.constant(p, 0)', 'nn.init.constant_(p, 0)')
with open(os.path.join(BASE, 'utils.py'), 'w') as f:
    f.write(content)
print("✓ utils.py fixed")

# ── 3. Fix dgr.py ─────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'dgr.py'), 'r') as f:
    content = f.read()
content = content.replace('from torch.autograd import Variable\n', '')
content = content.replace(
    '            x = Variable(x).cuda() if cuda else Variable(x)\n'
    '            y = Variable(y).cuda() if cuda else Variable(y)',
    '            x = x.cuda() if cuda else x\n'
    '            y = y.cuda() if cuda else y'
)
content = content.replace(
    "                x_ = Variable(x_).cuda() if cuda else Variable(x_)\n"
    "                y_ = Variable(y_).cuda() if cuda else Variable(y_)",
    "                x_ = x_.cuda() if cuda else x_\n"
    "                y_ = y_.cuda() if cuda else y_"
)
content = content.replace(
    "        real_prec = (y == real_predicted).sum().data[0] / batch_size",
    "        real_prec = (y == real_predicted).sum().item() / batch_size"
)
content = content.replace(
    "            replay_prec = (y_ == replay_predicted).sum().data[0] / batch_size",
    "            replay_prec = (y_ == replay_predicted).sum().item() / batch_size"
)
content = content.replace(
    "        return {'loss': loss.data[0], 'precision': precision}",
    "        return {'loss': loss.item(), 'precision': precision}"
)
with open(os.path.join(BASE, 'dgr.py'), 'w') as f:
    f.write(content)
print("✓ dgr.py fixed")

# ── 4. Fix models.py ──────────────────────────────────────────────────────
with open(os.path.join(BASE, 'models.py'), 'r') as f:
    content = f.read()
content = content.replace('from torch.autograd import Variable\n', '')
content = content.replace(
    "    def _noise(self, size):\n        z = Variable(torch.randn(size, self.z_size)) * .1",
    "    def _noise(self, size):\n        z = torch.randn(size, self.z_size) * .1"
)
content = content.replace(
    "        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)",
    "        interpolated = (a * x.detach() + (1 - a) * g.detach()).requires_grad_(True)"
)
content = content.replace(
    "        return {'c_loss': c_loss.data[0], 'g_loss': g_loss.data[0]}",
    "        return {'c_loss': c_loss.item(), 'g_loss': g_loss.item()}"
)
with open(os.path.join(BASE, 'models.py'), 'w') as f:
    f.write(content)
print("✓ models.py fixed")

# ── 5. Patch train.py (iterations + PACOL injection) ──────────────────────
TRAIN_PY = '''import os.path
import copy
import numpy as np
import torch
from torch import optim
from torch import nn
import utils
import visual


def train(scholar, train_datasets, test_datasets, replay_mode,
          generator_lambda=10.,
          generator_c_updates_per_g_update=5,
          generator_iterations=8000,
          solver_iterations=5000,
          importance_of_new_task=.5,
          batch_size=32,
          test_size=1024,
          sample_size=36,
          lr=1e-03, weight_decay=1e-05,
          beta1=.5, beta2=.9,
          loss_log_interval=30,
          eval_log_interval=50,
          image_log_interval=100,
          sample_log_interval=300,
          sample_log=False,
          sample_dir=\'./samples\',
          checkpoint_dir=\'./checkpoints\',
          collate_fn=None,
          cuda=False,
          pacol_attacker=None,
          target_dataset=None,
          nontarget_task_ids=None,
          poison_ratio=0.0,
          seed=0):
    solver_criterion = nn.CrossEntropyLoss()
    solver_optimizer = optim.Adam(
        scholar.solver.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    generator_g_optimizer = optim.Adam(
        scholar.generator.generator.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    generator_c_optimizer = optim.Adam(
        scholar.generator.critic.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    scholar.solver.set_criterion(solver_criterion)
    scholar.solver.set_optimizer(solver_optimizer)
    scholar.generator.set_lambda(generator_lambda)
    scholar.generator.set_generator_optimizer(generator_g_optimizer)
    scholar.generator.set_critic_optimizer(generator_c_optimizer)
    scholar.generator.set_critic_updates_per_generator_update(
        generator_c_updates_per_g_update
    )
    scholar.train()

    previous_scholar = None
    previous_datasets = None

    for task, train_dataset in enumerate(train_datasets, 1):
        # PACOL injection: poison non-target tasks before training
        if pacol_attacker is not None and nontarget_task_ids and task in nontarget_task_ids:
            n_poison = max(1, int(len(train_dataset) * poison_ratio))
            rng = np.random.default_rng(seed + task)
            nt_idx = rng.choice(len(train_dataset), n_poison, replace=False)
            nt_x = torch.stack([train_dataset[int(i)][0] for i in nt_idx])
            nt_y = torch.tensor([train_dataset[int(i)][1] for i in nt_idx])
            print(f\'  [task {task}] crafting {n_poison} poison samples...\', flush=True)
            adv_x = pacol_attacker.craft_poison(target_dataset, nt_x, nt_y, seed=seed + task)
            train_dataset = _inject_poison(train_dataset, adv_x, nt_idx)
            print(f\'  [task {task}] poison injected, training...\', flush=True)

        generator_training_callbacks = [_generator_training_callback(
            loss_log_interval=loss_log_interval,
            image_log_interval=image_log_interval,
            sample_log_interval=sample_log_interval,
            sample_log=sample_log,
            sample_dir=sample_dir,
            sample_size=sample_size,
            current_task=task,
            total_tasks=len(train_datasets),
            total_iterations=generator_iterations,
            batch_size=batch_size,
            replay_mode=replay_mode,
            env=scholar.name,
        )]
        solver_training_callbacks = [_solver_training_callback(
            loss_log_interval=loss_log_interval,
            eval_log_interval=eval_log_interval,
            current_task=task,
            total_tasks=len(train_datasets),
            total_iterations=solver_iterations,
            batch_size=batch_size,
            test_size=test_size,
            test_datasets=test_datasets,
            replay_mode=replay_mode,
            cuda=cuda,
            collate_fn=collate_fn,
            env=scholar.name,
        )]

        scholar.train_with_replay(
            train_dataset,
            scholar=previous_scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            generator_iterations=generator_iterations,
            generator_training_callbacks=generator_training_callbacks,
            solver_iterations=solver_iterations,
            solver_training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

        previous_scholar = (
            copy.deepcopy(scholar) if replay_mode == \'generative-replay\' else None
        )
        previous_datasets = (
            train_datasets[:task] if replay_mode == \'exact-replay\' else None
        )

    print()
    utils.save_checkpoint(scholar, checkpoint_dir)
    print()
    print()


def _inject_poison(dataset, adv_x, indices):
    from torch.utils.data import Dataset as TorchDataset

    class PoisonedDataset(TorchDataset):
        def __init__(self, base, adv_x, indices):
            self.base = base
            self.adv = {int(idx): adv_x[i] for i, idx in enumerate(indices)}

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, y = self.base[i]
            return self.adv.get(i, x), y

    return PoisonedDataset(dataset, adv_x, indices)


def _generator_training_callback(
        loss_log_interval, image_log_interval, sample_log_interval,
        sample_log, sample_dir, current_task, total_tasks, total_iterations,
        batch_size, sample_size, replay_mode, env):

    def cb(generator, progress, batch_index, result):
        iteration = (current_task-1)*total_iterations + batch_index
        progress.set_description((
            \'<Training Generator> task: {task}/{tasks} | \' +
            \'progress: [{trained}/{total}] ({percentage:.0f}%) | \' +
            \'loss => g: {g_loss:.4} / w: {w_dist:.4}\'
        ).format(
            task=current_task, tasks=total_tasks,
            trained=batch_size * batch_index, total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            g_loss=result[\'g_loss\'], w_dist=-result[\'c_loss\'],
        ))
        if iteration % loss_log_interval == 0:
            visual.visualize_scalar(result[\'g_loss\'], \'generator g loss\', iteration, env=env)
            visual.visualize_scalar(-result[\'c_loss\'], \'generator w distance\', iteration, env=env)
        if iteration % image_log_interval == 0:
            visual.visualize_images(
                generator.sample(sample_size).data,
                \'generated samples ({replay_mode})\'.format(replay_mode=replay_mode), env=env,
            )
        if iteration % sample_log_interval == 0 and sample_log:
            utils.test_model(generator, sample_size,
                os.path.join(sample_dir, env + \'-sample-logs\', str(iteration)), verbose=False)

    return cb


def _solver_training_callback(
        loss_log_interval, eval_log_interval, current_task, total_tasks,
        total_iterations, batch_size, test_size, test_datasets, cuda,
        replay_mode, collate_fn, env):

    def cb(solver, progress, batch_index, result):
        iteration = (current_task-1)*total_iterations + batch_index
        progress.set_description((
            \'<Training Solver>    task: {task}/{tasks} | \' +
            \'progress: [{trained}/{total}] ({percentage:.0f}%) | \' +
            \'loss: {loss:.4} | prec: {prec:.4}\'
        ).format(
            task=current_task, tasks=total_tasks,
            trained=batch_size * batch_index, total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            loss=result[\'loss\'], prec=result[\'precision\'],
        ))
        if iteration % loss_log_interval == 0:
            visual.visualize_scalar(result[\'loss\'], \'solver loss\', iteration, env=env)
        if iteration % eval_log_interval == 0:
            names = [\'task {}\'.format(i+1) for i in range(len(test_datasets))]
            precs = [
                utils.validate(solver, test_datasets[i], test_size=test_size,
                    cuda=cuda, verbose=False, collate_fn=collate_fn)
                if i+1 <= current_task else 0
                for i in range(len(test_datasets))
            ]
            visual.visualize_scalars(precs, names,
                \'precision ({replay_mode})\'.format(replay_mode=replay_mode),
                iteration, env=env)

    return cb
'''
with open(os.path.join(BASE, 'train.py'), 'w') as f:
    f.write(TRAIN_PY)
print("✓ train.py updated (8000/5000 iters + PACOL injection)")

# ── 6. Patch data.py (add rotation-MNIST) ────────────────────────────────
with open(os.path.join(BASE, 'data.py'), 'r') as f:
    content = f.read()

rotation_code = """
ROTATION_ANGLES = [0, 45, 90, 135, 180]   # standard R-MNIST benchmark (PACOL paper)

"""
content = content.replace(
    "def _permutate_image_pixels",
    rotation_code + "def _permutate_image_pixels"
)

get_rotated_fn = """

def get_rotated_mnist_tasks(train=True, capacity=None):
    \"\"\"Return a list of 5 datasets, one per rotation angle (R-MNIST).\"\"\"
    base = datasets.MNIST(
        './datasets/mnist', train=train, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS),
    )
    task_datasets = []
    for angle in ROTATION_ANGLES:
        ds = copy.deepcopy(base)
        if angle != 0:
            rot = transforms.RandomRotation(degrees=(angle, angle))
            ds.transform = transforms.Compose([
                ds.transform,
                transforms.Lambda(lambda x, r=rot: r(x)),
            ])
        if capacity is not None and len(ds) < capacity:
            ds = ConcatDataset([copy.deepcopy(ds) for _ in range(math.ceil(capacity / len(ds)))])
        task_datasets.append(ds)
    return task_datasets

"""
content = content.replace(
    "\nDATASET_CONFIGS = {",
    get_rotated_fn + "\nDATASET_CONFIGS = {"
)
content = content.replace(
    "    'svhn': {'size': 32, 'channels': 3, 'classes': 10},\n\n}",
    "    'svhn': {'size': 32, 'channels': 3, 'classes': 10},\n"
    "    'rmnist': {'size': 32, 'channels': 1, 'classes': 10},\n}"
)
with open(os.path.join(BASE, 'data.py'), 'w') as f:
    f.write(content)
print("✓ data.py updated (rotation-MNIST added)")

# ── 7. Patch main.py (add rotated-mnist experiment + update iterations) ───
with open(os.path.join(BASE, 'main.py'), 'r') as f:
    content = f.read()
content = content.replace(
    "    choices=['permutated-mnist', 'svhn-mnist', 'mnist-svhn'],",
    "    choices=['permutated-mnist', 'svhn-mnist', 'mnist-svhn', 'rotated-mnist'],"
)
content = content.replace(
    "parser.add_argument('--generator-iterations', type=int, default=3000)",
    "parser.add_argument('--generator-iterations', type=int, default=8000)"
)
content = content.replace(
    "parser.add_argument('--solver-iterations', type=int, default=1000)",
    "parser.add_argument('--solver-iterations', type=int, default=5000)"
)
content = content.replace(
    "from data import get_dataset, DATASET_CONFIGS",
    "from data import get_dataset, get_rotated_mnist_tasks, DATASET_CONFIGS"
)
rotated_block = """    if experiment == 'rotated-mnist':
        train_datasets = get_rotated_mnist_tasks(train=True,  capacity=capacity)
        test_datasets  = get_rotated_mnist_tasks(train=False, capacity=capacity)
        dataset_config = DATASET_CONFIGS['rmnist']

    elif experiment == 'permutated-mnist':"""
content = content.replace(
    "    if experiment == 'permutated-mnist':",
    rotated_block
)
with open(os.path.join(BASE, 'main.py'), 'w') as f:
    f.write(content)
print("✓ main.py updated (rotated-mnist + 8000/5000 defaults)")

print("\nAll patches applied. Now copy attacks/ from PACOL_rebuild and push.")
print("  cp -r /content/PACOL_rebuild/attacks /content/DGR_sanity_check/")
