import numpy as np
import torch
from abc import ABC, abstractmethod


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc, centerline=None):
        pass


class CrossSectionQuantizer(Quantizer):
    def __init__(self,
                 wz_range=[-10.0, -4.0, 10.0, 8.0],  # 截面范围 [W_min, Z_min, W_max, Z_max]
                 div_n=[256, 32],  # 截面网格划分 [W_div, Z_div]
                 s_range=[-12.0, 12.0],  # 中心线截取范围（前后各12米）
                 s_thickness=0.375):  # 切片厚度
        """
        横截面量化器：将点云沿中心线“拉直”并投影到 (W, Z, S) 坐标系
        """
        super().__init__()
        self.wz_range = np.array(wz_range, dtype=np.float32)
        self.div_n = np.array(div_n, dtype=np.int32)
        self.s_range = s_range
        self.s_thickness = s_thickness
        self.num_channels = int(np.ceil((s_range[1] - s_range[0]) / s_thickness))  # 64

        self.steps = (np.array([wz_range[2], wz_range[3]]) - np.array([wz_range[0], wz_range[1]])) / self.div_n

        print(f"CrossSectionQuantizer Initialized:")
        print(f"  Cross-Section Range (W, Z): {self.wz_range.tolist()}")
        print(f"  Grid (W, Z): {self.div_n.tolist()}")
        print(f"  S-Axis Range: {self.s_range}, Thickness: {self.s_thickness}, Channels: {self.num_channels}")

    def process_centerline(self, centerline):
        """处理中心线，提取主分支，计算弧长和切线"""
        # 1. 提取主分支 (寻找包含距离原点(0,0)最近点的那个分支)
        dists_to_origin = np.linalg.norm(centerline[:, :2], axis=1)  # 仅用XY算距离
        closest_idx = np.argmin(dists_to_origin)
        main_branch_id = centerline[closest_idx, 3]

        # 过滤并按序号排序
        cl_main = centerline[centerline[:, 3] == main_branch_id]
        cl_main = cl_main[np.argsort(cl_main[:, 4])][:, :3]  # 取 x, y, z

        # 2. 计算累计弧长
        diffs = np.diff(cl_main, axis=0)
        segment_lens = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.zeros(len(cl_main))
        arc_lengths[1:] = np.cumsum(segment_lens)

        # 3. 寻找锚点 (S=0) 并平移弧长坐标
        anchor_idx = np.argmin(np.linalg.norm(cl_main[:, :2], axis=1))
        arc_lengths = arc_lengths - arc_lengths[anchor_idx]

        # 4. 计算每个点的切线方向 (Tangent)
        tangents = np.zeros_like(cl_main)
        if len(cl_main) > 1:
            tangents[1:-1] = cl_main[2:] - cl_main[:-2]  # 中心差分
            tangents[0] = cl_main[1] - cl_main[0]  # 前向差分
            tangents[-1] = cl_main[-1] - cl_main[-2]  # 后向差分
        # 归一化切线向量
        tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)

        return cl_main, arc_lengths, tangents

    def project_points(self, pc_np, cl, arc_lengths, tangents):
        """将点云投影到 (W, Z, S)"""
        # 1. 为每个点寻找最近的中心线点 (注意：只使用 X, Y 找最近点，忽略 Z 的高度差)
        dist_matrix = np.linalg.norm(pc_np[:, np.newaxis, :2] - cl[np.newaxis, :, :2], axis=2)
        nearest_idx = np.argmin(dist_matrix, axis=1)

        C_j = cl[nearest_idx]
        S_j = arc_lengths[nearest_idx]
        T_j = tangents[nearest_idx]

        # 2. 计算点到最近中心线点的向量
        V = pc_np - C_j

        # 3. 计算沿着中心线的精确偏移量 (Delta S)
        delta_s = np.sum(V * T_j, axis=1)
        S_proj = S_j + delta_s

        # 4. 计算法线方向 (用于确定横向偏移 W，假设Z轴向上)
        Z_up = np.array([0.0, 0.0, 1.0])
        N_j = np.cross(Z_up, T_j)
        N_j = N_j / (np.linalg.norm(N_j, axis=1, keepdims=True) + 1e-8)

        # 5. 计算横向偏移 W 和 高度 Z
        W_proj = np.sum(V * N_j, axis=1)
        Z_proj = pc_np[:, 2]

        return S_proj, W_proj, Z_proj

    def __call__(self, pc, centerline):
        """
        前向处理函数
        pc: (N, 3) Tensor 或 ndarray
        centerline: (M, 5) ndarray
        """
        device = pc.device if isinstance(pc, torch.Tensor) else torch.device('cpu')
        pc_np = pc.cpu().numpy() if isinstance(pc, torch.Tensor) else pc

        # 1. 坐标转换
        cl, arc_lengths, tangents = self.process_centerline(centerline)
        S_proj, W_proj, Z_proj = self.project_points(pc_np, cl, arc_lengths, tangents)

        # 2. 截断 (保留 S 在 [-12, 12) 之间的点)
        mask_s = (S_proj >= self.s_range[0]) & (S_proj < self.s_range[1])
        # 截断 (保留 W, Z 在设定范围内的点)
        mask_wz = (W_proj >= self.wz_range[0]) & (W_proj < self.wz_range[2]) & \
                  (Z_proj >= self.wz_range[1]) & (Z_proj < self.wz_range[3])

        valid_mask = mask_s & mask_wz

        S_valid = S_proj[valid_mask]
        W_valid = W_proj[valid_mask]
        Z_valid = Z_proj[valid_mask]

        if len(S_valid) == 0:
            return torch.zeros((0, 2), dtype=torch.int32, device=device), \
                torch.zeros((0, self.num_channels), dtype=torch.float32, device=device)

        # 3. 计算离散化索引
        # 通道索引 (S轴)
        channel_indices = np.floor((S_valid - self.s_range[0]) / self.s_thickness).astype(np.int64)
        channel_indices = np.clip(channel_indices, 0, self.num_channels - 1)

        # 空间网格索引 (W, Z 轴)
        w_indices = np.floor((W_valid - self.wz_range[0]) / self.steps[0]).astype(np.int64)
        z_indices = np.floor((Z_valid - self.wz_range[1]) / self.steps[1]).astype(np.int64)

        w_indices = np.clip(w_indices, 0, self.div_n[0] - 1)
        z_indices = np.clip(z_indices, 0, self.div_n[1] - 1)

        wz_indices = np.stack([w_indices, z_indices], axis=1)

        # 4. 去重并构建特征 (同原来的BEV逻辑，只是XY变成了WZ，Z变成了S)
        wz_indices_tensor = torch.tensor(wz_indices, device=device, dtype=torch.long)
        unique_wz, inverse_indices = torch.unique(wz_indices_tensor, dim=0, return_inverse=True)

        num_unique = unique_wz.shape[0]
        features = torch.zeros((num_unique, self.num_channels), dtype=torch.float32, device=device)

        channel_indices_tensor = torch.tensor(channel_indices, device=device, dtype=torch.long)

        # 填充 Occupancy (对应通道设为1)
        features.index_put_((inverse_indices, channel_indices_tensor), torch.tensor(1.0, device=device))

        return unique_wz.int(), features