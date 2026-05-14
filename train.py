import os.path
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
          dynamic_importance=False,
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
          sample_dir='./samples',
          checkpoint_dir='./checkpoints',
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
        # DGR paper uses r = 1/τ; fall back to the fixed value when not requested.
        task_importance = (1.0 / task) if dynamic_importance else importance_of_new_task

        # PACOL injection: poison non-target tasks before training
        if pacol_attacker is not None and nontarget_task_ids and task in nontarget_task_ids:
            # Bug 1 fix: sync attacker model to current CL solver state (θ_{τ+n−1}).
            # The paper requires θ at the time of the attack, not the initial weights.
            pacol_attacker.model_orig = copy.deepcopy(scholar.solver).to(pacol_attacker.device)

            n_poison = max(1, int(len(train_dataset) * poison_ratio))
            rng = np.random.default_rng(seed + task)
            nt_idx = rng.choice(len(train_dataset), n_poison, replace=False)
            nt_x = torch.stack([train_dataset[int(i)][0] for i in nt_idx])
            nt_y = torch.tensor([train_dataset[int(i)][1] for i in nt_idx])
            print(f'  [task {task}] crafting {n_poison} poison samples...', flush=True)
            adv_x = pacol_attacker.craft_poison(
                target_dataset, nt_x, nt_y, seed=seed + task,
                x_min=0.0, x_max=1.0,  # Bug 2 fix: clip to valid pixel range [0,1]
            )
            train_dataset = _inject_poison(train_dataset, adv_x, nt_idx)
            print(f'  [task {task}] poison injected, training...', flush=True)

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
            importance_of_new_task=task_importance,
            batch_size=batch_size,
            generator_iterations=generator_iterations,
            generator_training_callbacks=generator_training_callbacks,
            solver_iterations=solver_iterations,
            solver_training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

        previous_scholar = (
            copy.deepcopy(scholar) if replay_mode == 'generative-replay' else None
        )
        previous_datasets = (
            train_datasets[:task] if replay_mode == 'exact-replay' else None
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
            '<Training Generator> task: {task}/{tasks} | ' +
            'progress: [{trained}/{total}] ({percentage:.0f}%) | ' +
            'loss => g: {g_loss:.4} / w: {w_dist:.4}'
        ).format(
            task=current_task, tasks=total_tasks,
            trained=batch_size * batch_index, total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            g_loss=result['g_loss'], w_dist=-result['c_loss'],
        ))
        if iteration % loss_log_interval == 0:
            visual.visualize_scalar(result['g_loss'], 'generator g loss', iteration, env=env)
            visual.visualize_scalar(-result['c_loss'], 'generator w distance', iteration, env=env)
        if iteration % image_log_interval == 0:
            visual.visualize_images(
                generator.sample(sample_size).data,
                'generated samples ({replay_mode})'.format(replay_mode=replay_mode), env=env,
            )
        if iteration % sample_log_interval == 0 and sample_log:
            utils.test_model(generator, sample_size,
                os.path.join(sample_dir, env + '-sample-logs', str(iteration)), verbose=False)

    return cb


def _solver_training_callback(
        loss_log_interval, eval_log_interval, current_task, total_tasks,
        total_iterations, batch_size, test_size, test_datasets, cuda,
        replay_mode, collate_fn, env):

    def cb(solver, progress, batch_index, result):
        iteration = (current_task-1)*total_iterations + batch_index
        progress.set_description((
            '<Training Solver>    task: {task}/{tasks} | ' +
            'progress: [{trained}/{total}] ({percentage:.0f}%) | ' +
            'loss: {loss:.4} | prec: {prec:.4}'
        ).format(
            task=current_task, tasks=total_tasks,
            trained=batch_size * batch_index, total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            loss=result['loss'], prec=result['precision'],
        ))
        if iteration % loss_log_interval == 0:
            visual.visualize_scalar(result['loss'], 'solver loss', iteration, env=env)
        if iteration % eval_log_interval == 0:
            names = ['task {}'.format(i+1) for i in range(len(test_datasets))]
            precs = [
                utils.validate(solver, test_datasets[i], test_size=test_size,
                    cuda=cuda, verbose=False, collate_fn=collate_fn)
                if i+1 <= current_task else 0
                for i in range(len(test_datasets))
            ]
            visual.visualize_scalars(precs, names,
                'precision ({replay_mode})'.format(replay_mode=replay_mode),
                iteration, env=env)

    return cb
