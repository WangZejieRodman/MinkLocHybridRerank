# datasets/dataset_utils.py
import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree

from datasets.base_datasets import EvaluationTuple, TrainingDataset
from datasets.augmentation import TrainSetTransform
from datasets.cyd_loader.cyd_train import CYDTrainingDataset
from datasets.cyd_loader.cyd_train import TrainTransform as CYDTrainTransform
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams
from datasets.base_datasets import PointCloudLoader
from datasets.cyd_loader.cyd_raw import CYDPointCloudLoader


def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    return CYDPointCloudLoader()


def make_datasets(params: TrainingParams, validation: bool = True):
    datasets = {}
    train_set_transform = TrainSetTransform(params.set_aug_mode) if params.set_aug_mode > 0 else None
    train_transform = CYDTrainTransform(params.aug_mode) if params.aug_mode > 0 else None
    datasets['train'] = CYDTrainingDataset(params.dataset_folder, params.train_file,
                                           transform=train_transform, set_transform=train_set_transform)
    if validation:
        datasets['val'] = CYDTrainingDataset(params.dataset_folder, params.val_file)

    return datasets


def make_collate_fn(dataset: TrainingDataset, model_params, batch_split_size=None):
    is_hybrid = hasattr(model_params, 'quantizer_coarse') and hasattr(model_params, 'quantizer_fine')

    def collate_fn(data_list):
        clouds = [e[0][0] for e in data_list]
        centerlines = [e[0][1] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds, centerlines = dataset.set_transform((clouds, centerlines))
            clouds = clouds.split(lens)

        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in
                          labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        if is_hybrid:
            batch_coarse_coords, batch_coarse_feats = [], []
            batch_fine_coords, batch_fine_feats = [], []

            coarse_type = getattr(model_params, 'coarse_type', 'bev')

            for pc, cl in zip(clouds, centerlines):
                # 预备好两种格式的 centerline
                cl_tensor = cl if isinstance(cl, torch.Tensor) else torch.tensor(cl, dtype=torch.float32)
                cl_numpy = cl.cpu().numpy() if isinstance(cl, torch.Tensor) else cl

                # 动态分配粗流的数据类型
                cl_coarse = cl_tensor if coarse_type == 'bev' else cl_numpy

                # 提取粗流
                c_coords, c_features = model_params.quantizer_coarse(pc, cl_coarse)
                batch_coarse_coords.append(c_coords)
                batch_coarse_feats.append(c_features)

                # 提取精流 (始终为 Cross-Section，所以用 Numpy)
                f_coords, f_features = model_params.quantizer_fine(pc, cl_numpy)
                batch_fine_coords.append(f_coords)
                batch_fine_feats.append(f_features)

            if batch_split_size is None or batch_split_size == 0:
                batch = {
                    'coarse_coords': ME.utils.batched_coordinates(batch_coarse_coords),
                    'coarse_features': torch.cat(batch_coarse_feats, dim=0),
                    'fine_coords': ME.utils.batched_coordinates(batch_fine_coords),
                    'fine_features': torch.cat(batch_fine_feats, dim=0)
                }
            else:
                batch = []
                for i in range(0, len(batch_coarse_coords), batch_split_size):
                    minibatch = {
                        'coarse_coords': ME.utils.batched_coordinates(batch_coarse_coords[i:i + batch_split_size]),
                        'coarse_features': torch.cat(batch_coarse_feats[i:i + batch_split_size], dim=0),
                        'fine_coords': ME.utils.batched_coordinates(batch_fine_coords[i:i + batch_split_size]),
                        'fine_features': torch.cat(batch_fine_feats[i:i + batch_split_size], dim=0)
                    }
                    batch.append(minibatch)
        else:
            # 单一流模式 (兼容老代码)
            batch_coords = []
            batch_feats = []

            for pc, cl in zip(clouds, centerlines):
                coords, features = model_params.quantizer(pc, cl)
                batch_coords.append(coords)
                batch_feats.append(features)

            if batch_split_size is None or batch_split_size == 0:
                coords = ME.utils.batched_coordinates(batch_coords)
                feats = torch.cat(batch_feats, dim=0)
                batch = {'coords': coords, 'features': feats}
            else:
                batch = []
                for i in range(0, len(batch_coords), batch_split_size):
                    temp_coords = batch_coords[i:i + batch_split_size]
                    temp_feats = batch_feats[i:i + batch_split_size]
                    c = ME.utils.batched_coordinates(temp_coords)
                    f = torch.cat(temp_feats, dim=0)
                    minibatch = {'coords': c, 'features': f}
                    batch.append(minibatch)

        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: TrainingParams, validation=True):
    datasets = make_datasets(params, validation=validation)
    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    train_collate_fn = make_collate_fn(datasets['train'], params.model_params, params.batch_split_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], params.model_params, params.batch_split_size)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple], dist_threshold: float) -> \
        List[EvaluationTuple]:
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position
    kdtree = KDTree(map_pos)
    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1
    print(
        f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e