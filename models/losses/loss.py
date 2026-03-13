# Warsaw University of Technology

import torch
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance
from misc.utils import TrainingParams
from models.losses.loss_utils import *
from models.losses.truncated_smoothap import TruncatedSmoothAP

# 引入新增的 Soft-DTW
from models.losses.soft_dtw import BatchSoftDTW


def make_losses(params: TrainingParams):
    if params.loss == 'batchhardtripletmarginloss':
        loss_fn = BatchHardTripletLossWithMasks(params.margin)
    elif params.loss == 'batchhardcontrastiveloss':
        loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin)
    elif params.loss == 'truncatedsmoothap':
        loss_fn = TruncatedSmoothAP(tau1=params.tau1, similarity=params.similarity,
                                    positives_per_query=params.positives_per_query)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError

    # 包装为双流损失 (如果不支持双流，该包装器内部会自动降级兼容)
    return DualStreamLoss(coarse_loss_fn=loss_fn, gamma=1.0, alpha=1.0)


class DualStreamLoss:
    def __init__(self, coarse_loss_fn, gamma=1.0, alpha=1.0):
        """
        双流联合损失函数
        :param coarse_loss_fn: 粗流损失函数 (如 TruncatedSmoothAP)
        :param gamma: Soft-DTW 的平滑系数
        :param alpha: 精流损失(Fine Loss) 的权重系数
        """
        self.coarse_loss_fn = coarse_loss_fn
        self.soft_dtw = BatchSoftDTW(gamma=gamma)
        self.alpha = alpha

    def __call__(self, embeddings_dict, positives_mask, negatives_mask):
        # 兼容旧版 Trainer：如果传入的还只是单一张量，直接跑粗流逻辑
        if not isinstance(embeddings_dict, dict):
            return self.coarse_loss_fn(embeddings_dict, positives_mask, negatives_mask)

        # -------------------------------------------------------------------
        # 1. 计算粗流 (Coarse) 损失
        # -------------------------------------------------------------------
        coarse_emb = embeddings_dict['global']
        loss_coarse, stats = self.coarse_loss_fn(coarse_emb, positives_mask, negatives_mask)

        if 'sequence' not in embeddings_dict:
            return loss_coarse, stats

        # -------------------------------------------------------------------
        # 2. 计算精流 (Fine) Soft-DTW Triplet 损失
        # -------------------------------------------------------------------
        seq_emb = embeddings_dict['sequence']  # (B, 64, N)
        B = seq_emb.shape[0]

        # 利用粗流的特征计算欧氏距离矩阵，来做低成本的 Hard Mining
        with torch.no_grad():
            dist_mat = torch.cdist(coarse_emb, coarse_emb, p=2)

            # 找到最难的正样本 (距离最远)
            pos_dist_mat = dist_mat.clone()
            pos_dist_mat[~positives_mask] = -1.0  # 屏蔽非正样本
            hardest_pos_idx = pos_dist_mat.argmax(dim=1)

            # 找到最难的负样本 (距离最近)
            neg_dist_mat = dist_mat.clone()
            neg_dist_mat[~negatives_mask] = float('inf')  # 屏蔽非负样本
            hardest_neg_idx = neg_dist_mat.argmin(dim=1)

        anchor_seq = seq_emb
        pos_seq = seq_emb[hardest_pos_idx]
        neg_seq = seq_emb[hardest_neg_idx]

        # 逆序序列：针对 CYD 巷道双向行驶导致 S 轴切片完全倒置的先验设计
        pos_seq_rev = torch.flip(pos_seq, dims=[1])
        neg_seq_rev = torch.flip(neg_seq, dims=[1])

        # 对正向和逆向均计算 DTW 距离，取较小者，赋予模型天生的双向行驶鲁棒性
        dtw_pos = torch.min(self.soft_dtw(anchor_seq, pos_seq), self.soft_dtw(anchor_seq, pos_seq_rev))
        dtw_neg = torch.min(self.soft_dtw(anchor_seq, neg_seq), self.soft_dtw(anchor_seq, neg_seq_rev))

        # Triplet Margin Loss (Margin 可以作为超参，这里暂设为 1.0)
        margin = 1.0
        loss_fine_all = torch.clamp(dtw_pos - dtw_neg + margin, min=0.0)

        # 屏蔽那些没有其他正样本的 Query（即 pos 只能匹配到自身的帧）
        anchor_idx = torch.arange(B, device=coarse_emb.device)
        valid_mask = (hardest_pos_idx != anchor_idx)

        if valid_mask.sum() > 0:
            loss_fine = loss_fine_all[valid_mask].mean()
            stats['dtw_pos'] = dtw_pos[valid_mask].mean().item()
            stats['dtw_neg'] = dtw_neg[valid_mask].mean().item()
        else:
            loss_fine = torch.tensor(0.0, device=coarse_emb.device, requires_grad=True)

        # -------------------------------------------------------------------
        # 3. 聚合双流损失
        # -------------------------------------------------------------------
        total_loss = loss_coarse + self.alpha * loss_fine

        # 注入额外的监控指标
        stats['loss_coarse'] = loss_coarse.item()
        stats['loss_fine'] = loss_fine.item()
        stats['loss'] = total_loss.item()  # 覆盖原先的单流 loss，用于反向传播

        return total_loss, stats


# 以下保留原有类 (未做修改)
class HardTripletMinerWithMasks:
    def __init__(self, distance):
        self.distance = distance
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist[a_keep_idx]).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist[a_keep_idx]).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist[a_keep_idx]).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist[a_keep_idx]).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist[a_keep_idx]).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist[a_keep_idx]).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin: float):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }
        return loss, stats


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin: float, neg_margin: float):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2 * len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }
        return loss, stats