import math
import random
import numpy as np
import torch
from scipy.linalg import expm, norm
from torchvision import transforms as transforms


def _unpack(data):
    if isinstance(data, tuple):
        return data[0], data[1]
    return data, None


def _pack(coords, cl, original_is_tuple):
    return (coords, cl) if original_is_tuple else coords


class TrainSetTransform:
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            t = [RandomRotation(max_theta=180, axis=np.array([0, 0, 1])),
                 RandomFlip([0.25, 0.25, 0.])]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class RandomFlip:
    def __init__(self, p):
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, data):
        coords, cl = _unpack(data)
        is_tuple = isinstance(data, tuple)
        r = random.random()

        flip_idx = -1
        if r <= self.p_cum_sum[0]:
            flip_idx = 0
        elif r <= self.p_cum_sum[1]:
            flip_idx = 1
        elif r <= self.p_cum_sum[2]:
            flip_idx = 2

        if flip_idx != -1:
            coords[..., flip_idx] = -coords[..., flip_idx]
            if cl is not None:
                if isinstance(cl, list):
                    for c in cl: c[:, flip_idx] = -c[:, flip_idx]
                else:
                    cl[:, flip_idx] = -cl[:, flip_idx]
        return _pack(coords, cl, is_tuple)


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=None):
        self.axis = axis
        self.max_theta = max_theta
        self.max_theta2 = max_theta2

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, data):
        coords, cl = _unpack(data)
        is_tuple = isinstance(data, tuple)

        axis = self.axis if self.axis is not None else np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180.) * 2. * (np.random.rand(1) - 0.5))
        if self.max_theta2 is not None:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180.) * 2. * (np.random.rand(1) - 0.5))
            R = R @ R_n

        coords = coords @ torch.tensor(R, device=coords.device)

        if cl is not None:
            if isinstance(cl, list):
                for c in cl: c[:, :3] = c[:, :3] @ R
            else:
                cl[:, :3] = cl[:, :3] @ R
        return _pack(coords, cl, is_tuple)


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, data):
        coords, cl = _unpack(data)
        is_tuple = isinstance(data, tuple)
        trans = self.max_delta * np.random.randn(1, 3).astype(np.float32)

        coords = coords + torch.tensor(trans, device=coords.device)
        if cl is not None:
            if isinstance(cl, list):
                for c in cl: c[:, :3] = c[:, :3] + trans
            else:
                cl[:, :3] = cl[:, :3] + trans
        return _pack(coords, cl, is_tuple)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        self.sigma = sigma;
        self.clip = clip;
        self.p = p

    def __call__(self, data):
        e, cl = _unpack(data)
        is_tuple = isinstance(data, tuple)
        sample_shape = (e.shape[0],)
        if self.p < 1.:
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64)
        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])
        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)
        e[mask] = e[mask] + jitter
        return _pack(e, cl, is_tuple)


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) in [list, tuple]:
            self.r_min = float(r[0]);
            self.r_max = float(r[1])
        else:
            self.r_min = None;
            self.r_max = float(r)

    def __call__(self, data):
        e, cl = _unpack(data)
        is_tuple = isinstance(data, tuple)
        n = len(e)
        r = self.r_max if self.r_min is None else random.uniform(self.r_min, self.r_max)
        mask = np.random.choice(range(n), size=int(n * r), replace=False)
        e[mask] = torch.zeros_like(e[mask])
        return _pack(e, cl, is_tuple)


class RemoveRandomBlock:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p;
        self.scale = scale;
        self.ratio = ratio

    def get_params(self, coords):
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)
        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)
        return x, y, w, h

    def __call__(self, data):
        coords, cl = _unpack(data)
        is_tuple = isinstance(data, tuple)
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)
            mask = (x < coords[..., 0]) & (coords[..., 0] < x + w) & (y < coords[..., 1]) & (coords[..., 1] < y + h)
            coords[mask] = torch.zeros_like(coords[mask])
        return _pack(coords, cl, is_tuple)