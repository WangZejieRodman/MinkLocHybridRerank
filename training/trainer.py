# Warsaw University of Technology
# Train MinkLoc model

import os
import numpy as np
import torch
import tqdm
import pathlib
import wandb

from misc.utils import TrainingParams, get_datetime
from models.losses.loss import make_losses
from models.model_factory import model_factory
from datasets.dataset_utils import make_dataloaders


def print_global_stats(phase, stats):
    from datetime import datetime

    s = f"{phase}  loss: {stats.get('loss', 0.0):.4f}   "

    # --- 新增：双流 Loss 与 DTW 统计信息 ---
    if 'loss_coarse' in stats and 'loss_fine' in stats:
        s += f"[Coarse: {stats['loss_coarse']:.4f}, Fine: {stats['loss_fine']:.4f}]   "
    if 'dtw_pos' in stats and 'dtw_neg' in stats:
        s += f"DTW(P/N): {stats['dtw_pos']:.1f}/{stats['dtw_neg']:.1f}   "

    # --- 新增：动态门控权重监控 ---
    if 'w_bev' in stats and 'w_cross' in stats:
        s += f"Gate(BEV/Cross): {stats['w_bev']:.2f}/{stats['w_cross']:.2f}   "
    # -----------------------------------

    if 'avg_embedding_norm' in stats:
        s += f"emb_norm: {stats['avg_embedding_norm']:.3f}  "
    if 'num_triplets' in stats:
        s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats.get('num_non_zero_triplets', 0):.1f}  " \
             f"Mean dist (pos/neg): {stats.get('mean_pos_pair_dist', 0):.3f}/{stats.get('mean_neg_pair_dist', 0):.3f}   "
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    if 'avg_voxels' in stats:
        s += f"体素: {stats['avg_voxels']:.0f}   "
    if 'gpu_memory_mb' in stats:
        s += f"显存: {stats['gpu_memory_mb']:.0f}MB   "

    print(s)

    log_file = "/home/wzj/pan1/MinkLoc3dv2_Chilean_原始点云/training/trainer.log"
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {s}\n")


def print_stats(phase, stats):
    print_global_stats(phase, stats['global'])


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}

    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    voxel_counts = []
    # 适配混合双流的字典 Key
    coords_key = 'coarse_coords' if 'coarse_coords' in batch else 'coords'

    if isinstance(batch[coords_key], torch.Tensor):
        batch_indices = batch[coords_key][:, 0]
        if len(batch_indices) > 0:
            for i in range(int(batch_indices.max()) + 1):
                voxels_in_sample = (batch_indices == i).sum().item()
                voxel_counts.append(voxels_in_sample)

    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)
        stats = model.stats.copy() if hasattr(model, 'stats') else {}

        loss, temp_stats = loss_fn(y, positives_mask, negatives_mask)
        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)

        if phase == 'train':
            loss.backward()
            optimizer.step()

    if voxel_counts:
        stats['avg_voxels'] = np.mean(voxel_counts)
        stats['max_voxels'] = np.max(voxel_counts)
        stats['min_voxels'] = np.min(voxel_counts)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        stats['gpu_memory_mb'] = torch.cuda.memory_allocated(device) / 1024 / 1024

    torch.cuda.empty_cache()
    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    all_voxel_counts = []

    # Stage 1 - 前向计算各个小批次特征 (关闭梯度)
    coarse_l = []
    fine_l = []

    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}

            # 适配混合双流的字典 Key
            coords_key = 'coarse_coords' if 'coarse_coords' in minibatch else 'coords'
            batch_indices = minibatch[coords_key][:, 0]
            if len(batch_indices) > 0:
                for i in range(int(batch_indices.max()) + 1):
                    voxels_in_sample = (batch_indices == i).sum().item()
                    all_voxel_counts.append(voxels_in_sample)

            y = model(minibatch)
            coarse_l.append(y['global'])
            if 'sequence' in y:
                fine_l.append(y['sequence'])

    torch.cuda.empty_cache()

    # Stage 2 - 将小批次特征拼接后，计算 Loss 对特征张量的梯度
    coarse_embeddings = torch.cat(coarse_l, dim=0)
    has_fine = len(fine_l) > 0
    if has_fine:
        fine_embeddings = torch.cat(fine_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            # Create leaf nodes out of the concatenated embeddings
            coarse_embeddings = coarse_embeddings.detach().requires_grad_(True)
            if has_fine:
                fine_embeddings = fine_embeddings.detach().requires_grad_(True)

        emb_dict = {'global': coarse_embeddings}
        if has_fine:
            emb_dict['sequence'] = fine_embeddings

        loss, stats = loss_fn(emb_dict, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)

        if phase == 'train':
            loss.backward()
            coarse_grad = coarse_embeddings.grad
            if has_fine:
                fine_grad = fine_embeddings.grad

    # 清理中间变量，防止显存泄漏
    coarse_l, fine_l, coarse_embeddings, emb_dict, y, loss = None, None, None, None, None, None

    # Stage 3 - 重新开启梯度进行前向计算，利用 Stage 2 缓存的梯度计算网络参数的梯度
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)

                coarse = y['global']
                minibatch_size = len(coarse)

                # 准备梯度回传张量列表
                tensors_to_backward = [coarse]
                grads_to_apply = [coarse_grad[i: i + minibatch_size]]

                if has_fine and 'sequence' in y:
                    fine = y['sequence']
                    tensors_to_backward.append(fine)
                    grads_to_apply.append(fine_grad[i: i + minibatch_size])

                # 通过 autograd.backward 同时将多条支路的梯度注入网络
                torch.autograd.backward(tensors=tensors_to_backward, grad_tensors=grads_to_apply)
                i += minibatch_size

            optimizer.step()

    if all_voxel_counts:
        stats['avg_voxels'] = np.mean(all_voxel_counts)
        stats['max_voxels'] = np.max(all_voxel_counts)
        stats['min_voxels'] = np.min(all_voxel_counts)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        stats['gpu_memory_mb'] = torch.cuda.memory_allocated(device) / 1024 / 1024

    torch.cuda.empty_cache()
    return stats


def do_train(params: TrainingParams, skip_final_eval=False):
    from datetime import datetime
    log_file = "/home/wzj/pan1/MinkLoc3dv2_Chilean_原始点云/training/trainer.log"
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_file, 'a') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始训练 (支持 Dual-Stream & Dynamic Gating)\n")
        f.write(f"{'=' * 80}\n")

    s = get_datetime()
    model = model_factory(params.model_params)
    model_name = 'model_' + params.model_params.model + '_' + s
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()

    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print('Model device: {}'.format(device))

    dataloaders = make_dataloaders(params)
    loss_fn = make_losses(params)

    if params.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif params.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {params.optimizer}")

    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr)
    else:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs + 1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = training_step
    else:
        train_step_fn = multistaged_training_step

    print("Wandb logging disabled - using local logging only")

    stats = {'train': [], 'eval': []}
    if 'val' in dataloaders:
        phases = ['train', 'val']
        stats['val'] = []
    else:
        phases = ['train']

    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        metrics = {'train': {}, 'val': {}}

        for phase in phases:
            running_stats = []
            count_batches = 0

            global_iter = iter(dataloaders['train']) if phase == 'train' else (
                None if dataloaders['val'] is None else iter(dataloaders['val']))

            while True:
                count_batches += 1
                batch_stats = {}
                if params.debug and count_batches > 2:
                    break

                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    batch_stats['global'] = temp_stats
                except StopIteration:
                    break

                running_stats.append(batch_stats)

            epoch_stats = {}
            for substep in running_stats[0]:
                epoch_stats[substep] = {}
                for key in running_stats[0][substep]:
                    temp = [e[substep][key] for e in running_stats if key in e[substep]]
                    if not temp: continue
                    if type(temp[0]) is dict:
                        epoch_stats[substep][key] = {k: np.mean([e[k] for e in temp if k in e]) for k in temp[0]}
                    elif type(temp[0]) is np.ndarray:
                        epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                    else:
                        epoch_stats[substep][key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(phase, epoch_stats)

        if scheduler is not None:
            scheduler.step()

        if params.save_freq > 0 and epoch % params.save_freq == 0:
            torch.save(model.state_dict(), model_pathname + "_" + str(epoch) + ".pth")

        if params.batch_expansion_th is not None:
            le_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' in le_train_stats['global']:
                rnz = le_train_stats['global']['num_non_zero_triplets'] / max(le_train_stats['global']['num_triplets'],
                                                                              1)
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

    print('')
    final_model_path = model_pathname + '_final.pth'
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)

    if not skip_final_eval:
        print('\n' + '=' * 60)
        print('Training completed! Skipping final eval.')
        print('=' * 60)

    return model, model_pathname


def create_weights_folder():
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path