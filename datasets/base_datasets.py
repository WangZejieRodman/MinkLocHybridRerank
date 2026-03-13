# datasets/base_datasets.py
import os
import pickle
from typing import List, Dict
import torch
import numpy as np
from torch.utils.data import Dataset


class TrainingTuple:
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        assert position.shape == (2,)
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class EvaluationTuple:
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array):
        assert position.shape == (2,)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position


class TrainingDataset(Dataset):
    def __init__(self, dataset_path, query_filename, transform=None, set_transform=None):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        if os.path.isabs(query_filename):
            self.query_filepath = query_filename
        else:
            self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self)))

        self.pc_loader: PointCloudLoader = None

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
        query_pc, centerline = self.pc_loader(file_pathname)
        query_pc = torch.tensor(query_pc, dtype=torch.float)

        if self.transform is not None:
            # 此时 transform 支持传入 (点云, 中心线) 元组
            query_pc, centerline = self.transform((query_pc, centerline))

        return (query_pc, centerline), ndx

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        query_l = [e.to_tuple() for e in self.query_set]
        map_l = [e.to_tuple() for e in self.map_set]
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))
        self.query_set = [EvaluationTuple(e[0], e[1], e[2]) for e in query_l]
        self.map_set = [EvaluationTuple(e[0], e[1], e[2]) for e in map_l]

    def get_map_positions(self):
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


class PointCloudLoader:
    def __init__(self):
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"

        # 读取点云
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        # 读取对应的中心线
        centerline = self.read_centerline(file_pathname)

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane and self.ground_plane_level is not None:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc, centerline

    def read_pc(self, file_pathname: str) -> np.ndarray:
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")

    def read_centerline(self, file_pathname: str) -> np.ndarray:
        raise NotImplementedError("read_centerline must be overloaded in an inheriting class")