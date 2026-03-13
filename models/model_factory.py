# models/model_factory.py
import torch.nn as nn

from models.minkloc import MinkLoc
from misc.utils import ModelParams
from models.layers.pooling_wrapper import PoolingWrapper
from models.minkbev import MinkBEVBackbone
from models.slice_branch import SliceSequenceBranch


def model_factory(model_params: ModelParams):
    """
    模型工厂（支持 BEV, Cross-Section 以及混合双流 Hybrid）
    """
    if model_params.model == 'MinkLocBEV':
        in_channels = getattr(model_params, 'in_channels', 32)
        print(f"Model Factory: Initializing MinkLocBEV...")
        print(f"  Input Channels (Z-layers): {in_channels}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        backbone = MinkBEVBackbone(in_channels=in_channels,
                                   out_channels=model_params.feature_size,
                                   dimension=2)

        pooling = PoolingWrapper(pool_method=model_params.pooling,
                                 in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)

        model = MinkLoc(backbone=backbone, pooling=pooling,
                        normalize_embeddings=model_params.normalize_embeddings)

    elif model_params.model == 'MinkLocCross':
        in_channels = getattr(model_params, 'in_channels', 64)
        print(f"Model Factory: Initializing MinkLocCross (Dual-Stream)...")
        print(f"  Input Channels (S-slices): {in_channels}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        backbone = MinkBEVBackbone(in_channels=in_channels,
                                   out_channels=model_params.feature_size,
                                   dimension=2)

        pooling = PoolingWrapper(pool_method=model_params.pooling,
                                 in_dim=model_params.feature_size,
                                 output_dim=model_params.output_dim)

        slice_feature_dim = getattr(model_params, 'slice_feature_dim', 32)
        slice_branch = SliceSequenceBranch(num_slices=in_channels, feature_dim=slice_feature_dim)

        model = MinkLoc(backbone=backbone, pooling=pooling,
                        normalize_embeddings=model_params.normalize_embeddings,
                        slice_branch=slice_branch)

    # =========================================================
    # 修改：混合双流 Hybrid 分支 (粗流双视角特征融合门控 + Cross精流)
    # =========================================================
    elif model_params.model == 'MinkLocHybrid':
        in_channels_bev = getattr(model_params, 'in_channels_bev', 32)
        in_channels_cross = getattr(model_params, 'in_channels_cross', 64)
        slice_feature_dim = getattr(model_params, 'slice_feature_dim', 32)

        print(f"Model Factory: Initializing MinkLocHybrid (Feature-level Fusion + Dynamic Gating)...")
        print(f"  [Coarse 1] BEV Input Channels: {in_channels_bev}")
        print(f"  [Coarse 2] Cross-Section Input Slices: {in_channels_cross}")
        print(f"  [Fine] Cross-Section Input Slices: {in_channels_cross}")
        print(f"  Feature Size (Backbone Out): {model_params.feature_size}")

        # 1. 粗流视角一：BEV Backbone & Pooling
        backbone_bev = MinkBEVBackbone(in_channels=in_channels_bev,
                                       out_channels=model_params.feature_size,
                                       dimension=2)
        pooling_bev = PoolingWrapper(pool_method=model_params.pooling,
                                     in_dim=model_params.feature_size,
                                     output_dim=model_params.output_dim)

        # 2. 粗流视角二：Cross-Section Backbone & Pooling (复用 MinkBEVBackbone 处理 2D 稀疏平面)
        backbone_cross = MinkBEVBackbone(in_channels=in_channels_cross,
                                         out_channels=model_params.feature_size,
                                         dimension=2)
        pooling_cross = PoolingWrapper(pool_method=model_params.pooling,
                                       in_dim=model_params.feature_size,
                                       output_dim=model_params.output_dim)

        # 3. 精流 Slice Sequence Branch
        slice_branch = SliceSequenceBranch(num_slices=in_channels_cross, feature_dim=slice_feature_dim)

        # 注入所有的分支，由 MinkLoc 内部控制融合逻辑
        model = MinkLoc(backbone=backbone_bev,
                        pooling=pooling_bev,
                        normalize_embeddings=model_params.normalize_embeddings,
                        slice_branch=slice_branch,
                        backbone_cross=backbone_cross,
                        pooling_cross=pooling_cross)

    elif model_params.model == 'MinkLoc':
        raise NotImplementedError(
            "MinkLoc (3D) has been removed from this codebase."
        )
    else:
        raise NotImplementedError(f'Model not implemented: {model_params.model}')

    return model