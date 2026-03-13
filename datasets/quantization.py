import numpy as np
from abc import ABC, abstractmethod
import torch
import MinkowskiEngine as ME


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc, centerline=None):
        pass


class BEVQuantizer(Quantizer):
    def __init__(self,
                 coords_range=[-10., -10, -4, 10, 10, 8],
                 div_n=[256, 256, 32]):
        """
        BEV 量化器（专用于混合双流粗检索，引入偏航角对齐）
        Args:
            coords_range: 点云裁剪范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            div_n: 网格划分数 [nx, ny, nz]
        """
        super().__init__()
        self.coords_range = torch.tensor(coords_range, dtype=torch.float)
        self.div_n = torch.tensor(div_n, dtype=torch.int32)
        self.steps = (self.coords_range[3:] - self.coords_range[:3]) / self.div_n

        print(f"BEVQuantizer Initialized:")
        print(f"  Range: {self.coords_range.tolist()}")
        print(f"  Grid: {self.div_n.tolist()}")
        print(f"  Steps: {self.steps.tolist()}")

    def _yaw_alignment(self, pc, centerline):
        """
        核心：基于中心线的偏航角对齐 (Yaw Alignment)
        计算切线向量 T 在绝对坐标系 XY 平面上的偏航角 theta，将点云绕 Z 轴反向旋转 -theta
        """
        if centerline is None or centerline.shape[0] < 2:
            return pc, 0.0

        device = pc.device
        centerline = centerline.to(device)

        # 1. 寻找距离 LiDAR (原点) 最近的中心线点 (强制只取前三维 X,Y,Z 计算距离)
        origin = torch.tensor([0.0, 0.0, 0.0], device=device)
        dists = torch.norm(centerline[:, :3] - origin, dim=1)
        min_idx = torch.argmin(dists)

        # 2. 计算该点处的切线方向 T (强制只取前三维 X,Y,Z 做差)
        if min_idx == 0:
            tangent = centerline[1, :3] - centerline[0, :3]
        elif min_idx == len(centerline) - 1:
            tangent = centerline[-1, :3] - centerline[-2, :3]
        else:
            # 采用前后两点的向量差作为中心点的切线方向
            tangent = centerline[min_idx + 1, :3] - centerline[min_idx - 1, :3]

        # 3. 计算偏航角 theta (与 X 轴的夹角)
        theta = torch.atan2(tangent[1], tangent[0])

        # 4. 构建绕 Z 轴旋转 -theta 的旋转矩阵
        cos_t = torch.cos(-theta)
        sin_t = torch.sin(-theta)
        rot_mat = torch.tensor([
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)

        # 5. 将点云绕 Z 轴反向旋转 -theta，消除车头朝向变化带来的旋转差异
        pc_aligned = torch.matmul(pc, rot_mat.T)

        return pc_aligned, theta.item()

    def __call__(self, pc, centerline=None):
        """
        Args:
            pc: (N, 3) Tensor, Cartesian coordinates (X, Y, Z)
            centerline: (K, >=3) Tensor, 中心线点
        Returns:
            unique_xy: (M, 2) Tensor, 2D坐标
            features: (M, 32) Tensor, Z轴Occupancy特征
        """
        device = pc.device

        # --- 执行偏航角对齐 ---
        pc_aligned, yaw_angle = self._yaw_alignment(pc, centerline)

        coords_range = self.coords_range.to(device)
        steps = self.steps.to(device)
        div_n = self.div_n.to(device)

        # 过滤范围外的点 (使用对齐后的点云)
        mask = (pc_aligned[:, 0] >= coords_range[0]) & (pc_aligned[:, 0] < coords_range[3]) & \
               (pc_aligned[:, 1] >= coords_range[1]) & (pc_aligned[:, 1] < coords_range[4]) & \
               (pc_aligned[:, 2] >= coords_range[2]) & (pc_aligned[:, 2] < coords_range[5])
        pc_valid = pc_aligned[mask]

        if pc_valid.shape[0] == 0:
            return torch.zeros((0, 2), dtype=torch.int32, device=device), \
                torch.zeros((0, div_n[2]), dtype=torch.float32, device=device)

        # 计算网格索引
        indices = ((pc_valid - coords_range[:3]) / steps).long()
        indices = torch.clamp(indices, min=torch.zeros(3, dtype=torch.long, device=device),
                              max=(div_n - 1))

        xy_indices = indices[:, :2]
        z_indices = indices[:, 2]

        # XY平面去重
        unique_xy, inverse_indices = torch.unique(xy_indices, dim=0, return_inverse=True)

        # 构建特征
        num_unique = unique_xy.shape[0]
        num_channels = div_n[2]
        features = torch.zeros((num_unique, num_channels), dtype=torch.float32, device=device)
        features.index_put_((inverse_indices, z_indices), torch.tensor(1.0, device=device))

        return unique_xy.int(), features