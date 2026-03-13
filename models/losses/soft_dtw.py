# models/losses/soft_dtw.py
import torch
import torch.nn as nn


class BatchSoftDTW(nn.Module):
    def __init__(self, gamma=1.0):
        """
        可微的批处理 Soft-DTW (Dynamic Time Warping)
        :param gamma: 平滑系数，gamma 越接近 0 越趋近于 Hard DTW
        """
        super(BatchSoftDTW, self).__init__()
        self.gamma = gamma

    def forward(self, x, y):
        """
        :param x: (B, N, D) Anchor 序列
        :param y: (B, M, D) 待匹配序列 (Positive 或 Negative)
        :return: (B,) 的 DTW 距离张量
        """
        B, N, M = x.shape[0], x.shape[1], y.shape[1]

        # 极速计算 pairwise Euclidean 距离矩阵: (B, N, M)
        D = torch.cdist(x, y, p=2)

        # 使用 Python 嵌套列表替代 PyTorch Tensor，完美避开 inplace 修改报错
        R = [[None for _ in range(M + 1)] for _ in range(N + 1)]

        # 预先生成边界常量张量
        inf_tensor = torch.full((B,), float('inf'), device=x.device, dtype=x.dtype)
        zero_tensor = torch.zeros((B,), device=x.device, dtype=x.dtype)

        # 初始化边界
        for i in range(N + 1):
            for j in range(M + 1):
                if i == 0 and j == 0:
                    R[i][j] = zero_tensor
                elif i == 0 or j == 0:
                    R[i][j] = inf_tensor

        # Soft-DTW 动态规划主循环
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                r0 = R[i - 1][j - 1]
                r1 = R[i - 1][j]
                r2 = R[i][j - 1]

                # Soft-min 操作: -gamma * logsumexp( -[r0, r1, r2] / gamma )
                stacked = torch.stack([r0, r1, r2], dim=1)
                soft_min = -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=1)

                # 直接将新张量赋值给列表元素，不触发 inplace 错误
                R[i][j] = D[:, i - 1, j - 1] + soft_min

        # 返回到达终点 (N, M) 的累积距离
        return R[N][M]