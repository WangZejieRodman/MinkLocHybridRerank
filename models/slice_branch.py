# models/slice_branch.py
import torch
import torch.nn as nn
import MinkowskiEngine as ME


class SliceSequenceBranch(nn.Module):
    def __init__(self, num_slices=64, feature_dim=32):
        super(SliceSequenceBranch, self).__init__()
        self.num_slices = num_slices
        self.feature_dim = feature_dim

        # 轻量级 2D 稀疏卷积网络 (作用于 W, Z 空间)
        # 这里 in_channels=1 是因为我们将 64 个切片展开成了 64 个独立的 Batch
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=1, out_channels=16, kernel_size=3, dimension=2),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(in_channels=16, out_channels=feature_dim, kernel_size=3, dimension=2),
            ME.MinkowskiBatchNorm(feature_dim),
            ME.MinkowskiReLU(inplace=True)
        )

        # 全局最大池化，用于把每个切片的独立 (W, Z) 空间压缩为一个向量
        self.pool = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x: ME.SparseTensor, batch_size: int):
        # 1. 解析输入的稀疏张量
        # coords: (M, 3) -> [batch_idx, W, Z]
        # features: (M, 64) 的二值占位网格
        coords = x.C
        features = x.F

        # 找到所有非空的 (点索引m_idx, 切片索引s_idx)
        non_zero_indices = features.nonzero()
        if len(non_zero_indices) == 0:
            # 极端情况：整个 batch 全空
            return torch.zeros((batch_size, self.num_slices, self.feature_dim), device=features.device)

        m_idx = non_zero_indices[:, 0]
        s_idx = non_zero_indices[:, 1]

        # 还原每个占用点的原始空间坐标
        orig_coords = coords[m_idx]
        batch_idx = orig_coords[:, 0]
        W = orig_coords[:, 1]
        Z = orig_coords[:, 2]

        # 2. 核心魔法：把 S 维度编码进 batch 维度，彻底隔离切片
        new_batch_idx = batch_idx * self.num_slices + s_idx
        new_coords = torch.stack([new_batch_idx, W, Z], dim=1).int()

        # 此时所有的输入通道变为 1 (即该位置存在点)
        new_features = torch.ones((len(m_idx), 1), dtype=torch.float32, device=features.device)

        # 构建新的虚拟 2D 稀疏张量，此时相当于把 batch 扩大了 64 倍
        st = ME.SparseTensor(features=new_features, coordinates=new_coords)

        # 3. 前向传播
        st = self.conv(st)
        st = self.pool(st)

        # 4. 散布回 dense 张量，这一步完美处理了“空切片”问题（空切片将保持为全0向量）
        seq_features = torch.zeros((batch_size * self.num_slices, self.feature_dim), device=features.device)

        # st.C[:, 0] 是池化后保留的 active_batch_idx
        active_indices = st.C[:, 0].long()
        seq_features[active_indices] = st.F

        # 5. 变回 (batch_size, 64, feature_dim) 序列矩阵
        seq_features = seq_features.view(batch_size, self.num_slices, self.feature_dim)

        return seq_features