# models/minkloc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.layers.pooling_wrapper import PoolingWrapper


class MinkLoc(torch.nn.Module):
    def __init__(self, backbone: nn.Module, pooling: PoolingWrapper,
                 normalize_embeddings: bool = False, slice_branch: nn.Module = None,
                 backbone_cross: nn.Module = None, pooling_cross: PoolingWrapper = None):
        super().__init__()
        # ==================================
        # 粗流第一视角 (BEV)
        # ==================================
        self.backbone = backbone
        self.pooling = pooling

        # ==================================
        # 粗流第二视角 (Cross-Section)
        # ==================================
        self.backbone_cross = backbone_cross
        self.pooling_cross = pooling_cross

        # ==================================
        # 精流分支 (Slice Sequence)
        # ==================================
        self.slice_branch = slice_branch

        self.normalize_embeddings = normalize_embeddings
        self.stats = {}

        # ==================================
        # 动态门控网络 (当存在双粗流时激活)
        # ==================================
        if self.backbone_cross is not None:
            feature_dim = self.pooling.output_dim
            self.gating = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 2),
                nn.Softmax(dim=-1)
            )

    def forward(self, batch):
        # 判断当前是混合双流模式(Hybrid)还是单一流模式
        is_hybrid = 'coarse_coords' in batch and 'fine_coords' in batch

        if is_hybrid:
            # 混合双流模式: 构建两个独立的 SparseTensor
            # coarse_x 对应 BEV 视角的点云特征
            coarse_x = ME.SparseTensor(batch['coarse_features'], coordinates=batch['coarse_coords'])
            # fine_x 对应 Cross-Section 视角的点云特征 (粗流Cross分支和精流分支完美复用)
            fine_x = ME.SparseTensor(batch['fine_features'], coordinates=batch['fine_coords'])

            # 用于精流切片提取的 batch_size 以 fine_coords 为准
            batch_size = int(batch['fine_coords'][:, 0].max().item() + 1) if len(batch['fine_coords']) > 0 else 1
        else:
            # 单一流模式 (兼容老代码)
            coarse_x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
            fine_x = coarse_x

            batch_size = int(batch['coords'][:, 0].max().item() + 1) if len(batch['coords']) > 0 else 1

        # =========================================================
        # 流1 (粗流 BEV): 全局特征提取
        # =========================================================
        bev_out = self.backbone(coarse_x)
        assert bev_out.shape[
                   1] == self.pooling.in_dim, f'Backbone output tensor has: {bev_out.shape[1]} channels. Expected: {self.pooling.in_dim}'

        bev_out = self.pooling(bev_out)
        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)

        assert bev_out.dim() == 2, f'Expected 2-dimensional tensor. Got {bev_out.dim()} dimensions.'

        if self.normalize_embeddings:
            bev_out = F.normalize(bev_out, dim=1)

        # =========================================================
        # 流2 (粗流 Cross-Section) + 动态门控特征级融合
        # =========================================================
        if self.backbone_cross is not None:
            cross_out = self.backbone_cross(fine_x)
            cross_out = self.pooling_cross(cross_out)

            if hasattr(self.pooling_cross, 'stats'):
                # 增加前缀避免与 BEV 的 stats 互相覆盖
                cross_stats = {f"cross_{k}": v for k, v in self.pooling_cross.stats.items()}
                self.stats.update(cross_stats)

            if self.normalize_embeddings:
                cross_out = F.normalize(cross_out, dim=1)

            # 特征拼接 (Batch, 512)
            fuse_features = torch.cat([bev_out, cross_out], dim=-1)

            # 经过门控网络计算各自的权重
            weights = self.gating(fuse_features)
            w_bev = weights[:, 0:1]
            w_cross = weights[:, 1:2]

            # 记录门控权重以供 Trainer 监控输出
            self.stats['w_bev'] = w_bev.mean().item()
            self.stats['w_cross'] = w_cross.mean().item()

            # 动态加权融合为最终的 256 维全局描述符
            coarse_out = w_bev * bev_out + w_cross * cross_out
        else:
            coarse_out = bev_out

        output = {'global': coarse_out}

        # =========================================================
        # 流3 (精流 Fine): 平行的切片序列提取 (64 * N-D)
        # =========================================================
        if self.slice_branch is not None:
            fine_seq = self.slice_branch(fine_x, batch_size)
            output['sequence'] = fine_seq

        return output

    def print_info(self):
        print('Model class: MinkLoc (Feature-level Fusion + Dynamic Gating)')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')

        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'BEV Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'BEV Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')

        if self.backbone_cross is not None:
            n_params = sum([param.nelement() for param in self.backbone_cross.parameters()])
            print(f'Cross Backbone: {type(self.backbone_cross).__name__} #parameters: {n_params}')
            n_params = sum([param.nelement() for param in self.pooling_cross.parameters()])
            print(f'Cross Pooling method: {self.pooling_cross.pool_method}   #parameters: {n_params}')
            n_params = sum([param.nelement() for param in self.gating.parameters()])
            print(f'Dynamic Gating #parameters: {n_params}')

        # 打印精流分支信息
        if self.slice_branch is not None:
            n_params = sum([param.nelement() for param in self.slice_branch.parameters()])
            print(f'Slice Branch: {type(self.slice_branch).__name__}   #parameters: {n_params}')
            print(f'# sequence slices: {self.slice_branch.num_slices}')
            print(f'# sequence feature dim: {self.slice_branch.feature_dim}')

        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')