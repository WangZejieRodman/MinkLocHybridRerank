# eval/evaluate_cyd_rotation.py
import os
import sys
import torch
import numpy as np
import tqdm
import pickle
from sklearn.neighbors import KDTree
import MinkowskiEngine as ME

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datasets.cyd_loader.cyd_raw import CYDPointCloudLoader
from datasets.rotation_utils import rotate_point_cloud_z
from misc.utils import TrainingParams
from models.model_factory import model_factory


def compute_dtw_distance(seq1, seq2):
    N, M = seq1.shape[0], seq2.shape[0]
    if N == 0 or M == 0:
        return float('inf')

    dist_mat = np.linalg.norm(seq1[:, None, :] - seq2[None, :, :], axis=2)
    dtw_matrix = np.full((N + 1, M + 1), float('inf'))
    dtw_matrix[0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = dist_mat[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])

    return dtw_matrix[N, M]


def filter_empty_slices(seq):
    norms = np.linalg.norm(seq, axis=1)
    return seq[norms > 1e-5]


def compute_embedding(model, pc, cl, device, params):
    is_hybrid = hasattr(params.model_params, 'quantizer_coarse') and hasattr(params.model_params, 'quantizer_fine')

    if is_hybrid:
        cl_tensor = torch.tensor(cl, dtype=torch.float32) if not isinstance(cl, torch.Tensor) else cl
        cl_np = cl.cpu().numpy() if isinstance(cl, torch.Tensor) else cl

        coarse_type = getattr(params.model_params, 'coarse_type', 'bev')
        cl_coarse = cl_tensor if coarse_type == 'bev' else cl_np

        c_coords, c_features = params.model_params.quantizer_coarse(pc, cl_coarse)
        f_coords, f_features = params.model_params.quantizer_fine(pc, cl_np)

        with torch.no_grad():
            batch = {
                'coarse_coords': ME.utils.batched_coordinates([c_coords]).to(device),
                'coarse_features': c_features.to(device),
                'fine_coords': ME.utils.batched_coordinates([f_coords]).to(device),
                'fine_features': f_features.to(device)
            }
            y = model(batch)
    else:
        cl_target = torch.tensor(cl, dtype=torch.float32) if getattr(params.model_params, 'coordinates',
                                                                     '') == 'bev' else cl
        coords, feats = params.model_params.quantizer(pc, cl_target)
        with torch.no_grad():
            batch = {
                'coords': ME.utils.batched_coordinates([coords]).to(device),
                'features': feats.to(device)
            }
            y = model(batch)

    glob_emb = y['global'].detach().cpu().numpy()
    if 'sequence' in y:
        seq_emb = y['sequence'].detach().cpu().numpy()
    else:
        seq_emb = np.zeros((1, 64, 32), dtype=np.float32)

    return glob_emb, seq_emb


def get_latent_vectors_rot(model, point_cloud_set, device, params, rotation_angle):
    pc_loader = CYDPointCloudLoader()
    model.eval()

    global_embeddings = None
    sequence_embeddings = None

    for i, elem_ndx in enumerate(point_cloud_set):
        pc_file_path = os.path.join(params.dataset_folder, point_cloud_set[elem_ndx]["query"])
        pc, cl = pc_loader(pc_file_path)

        if rotation_angle != 0:
            pc = rotate_point_cloud_z(pc, rotation_angle)
            cl[:, :3] = rotate_point_cloud_z(cl[:, :3], rotation_angle)

        pc_tensor = torch.tensor(pc, dtype=torch.float32)
        glob_emb, seq_emb = compute_embedding(model, pc_tensor, cl, device, params)

        if global_embeddings is None:
            global_embeddings = np.zeros((len(point_cloud_set), glob_emb.shape[1]), dtype=glob_emb.dtype)
            sequence_embeddings = np.zeros((len(point_cloud_set), seq_emb.shape[1], seq_emb.shape[2]),
                                           dtype=seq_emb.dtype)

        global_embeddings[i] = glob_emb
        sequence_embeddings[i] = seq_emb

    return global_embeddings, sequence_embeddings


def evaluate_cyd_with_rotation(model, device, params, rotation_angles, fusion_weight=0.5):
    dataset_cyd_path = os.path.join(project_root, 'datasets', 'cyd')
    database_path = os.path.join(dataset_cyd_path, 'cyd_evaluation_database_100_108.pickle')
    query_path = os.path.join(dataset_cyd_path, 'cyd_evaluation_query_109_113.pickle')

    if not os.path.exists(database_path) or not os.path.exists(query_path):
        print("❌ 错误: 找不到测试字典文件。")
        return None

    with open(database_path, 'rb') as f:
        database_sets = pickle.load(f)
    with open(query_path, 'rb') as f:
        query_sets = pickle.load(f)

    db_global_list = []
    db_seq_list = []
    database_to_session_map = []

    print("\nComputing database embeddings (0° Hybrid Dual-Stream)...")
    for i, db_set in enumerate(tqdm.tqdm(database_sets)):
        if len(db_set) > 0:
            glob_emb, seq_emb = get_latent_vectors_rot(model, db_set, device, params, 0)
            db_global_list.append(glob_emb)
            db_seq_list.append(seq_emb)
            for local_idx in range(len(glob_emb)):
                database_to_session_map.append((i, local_idx))

    database_global_output = np.vstack(db_global_list)
    database_seq_output = np.concatenate(db_seq_list, axis=0)
    database_nbrs = KDTree(database_global_output)

    all_stats = {}

    for angle in rotation_angles:
        print(f"\n[{angle}°] Computing query embeddings (Dual-Stream)...")
        query_global_list = []
        query_seq_list = []
        for query_set in tqdm.tqdm(query_sets):
            if len(query_set) > 0:
                glob_emb, seq_emb = get_latent_vectors_rot(model, query_set, device, params, angle)
                query_global_list.append(glob_emb)
                query_seq_list.append(seq_emb)
            else:
                query_global_list.append(np.array([]).reshape(0, 256))
                query_seq_list.append(np.array([]).reshape(0, 64, 32))

        num_neighbors = 25
        recall_coarse = np.zeros(num_neighbors)
        recall_fine = np.zeros(num_neighbors)
        num_evaluated = 0

        for j, query_set in enumerate(query_sets):
            if len(query_set) == 0: continue

            q_globals = query_global_list[j]
            q_seqs = query_seq_list[j]

            for query_idx in range(len(q_globals)):
                query_details = query_set[query_idx]
                if 'positives' not in query_details: continue

                true_neighbors_global = []
                for db_session_idx in query_details['positives']:
                    for local_idx in query_details['positives'][db_session_idx]:
                        for global_idx, (sess_idx, loc_idx) in enumerate(database_to_session_map):
                            if sess_idx == db_session_idx and loc_idx == local_idx:
                                true_neighbors_global.append(global_idx)

                if len(true_neighbors_global) == 0: continue
                num_evaluated += 1

                # 1. 粗检索：获取索引及对应的粗流距离
                distances, indices = database_nbrs.query(np.array([q_globals[query_idx]]), k=num_neighbors)
                coarse_candidates = indices[0]
                coarse_distances = distances[0]

                for k in range(len(coarse_candidates)):
                    if coarse_candidates[k] in true_neighbors_global:
                        recall_coarse[k:] += 1
                        break

                q_seq_valid = filter_empty_slices(q_seqs[query_idx])

                # 2. 精检索：计算序列的 DTW 距离
                fine_scores = []
                for candidate_idx in coarse_candidates:
                    c_seq_valid = filter_empty_slices(database_seq_output[candidate_idx])
                    dtw_fwd = compute_dtw_distance(q_seq_valid, c_seq_valid)
                    c_seq_rev = c_seq_valid[::-1]
                    dtw_bwd = compute_dtw_distance(q_seq_valid, c_seq_rev)
                    min_dtw = min(dtw_fwd, dtw_bwd)
                    fine_scores.append(min_dtw)

                # ========================================================
                # 3. Late Fusion 联合评分
                # ========================================================
                coarse_dist_np = np.array(coarse_distances)
                fine_dist_np = np.array(fine_scores)

                # 归一化到 [0, 1] 防止量纲冲突
                c_range = coarse_dist_np.max() - coarse_dist_np.min()
                f_range = fine_dist_np.max() - fine_dist_np.min()

                c_norm = (coarse_dist_np - coarse_dist_np.min()) / (c_range if c_range > 1e-6 else 1.0)
                f_norm = (fine_dist_np - fine_dist_np.min()) / (f_range if f_range > 1e-6 else 1.0)

                # 按权重融合
                final_scores = fusion_weight * c_norm + (1.0 - fusion_weight) * f_norm

                # 按融合得分从小到大重排序
                re_ranked_indices = [x for _, x in sorted(zip(final_scores, coarse_candidates))]
                # ========================================================

                for k in range(len(re_ranked_indices)):
                    if re_ranked_indices[k] in true_neighbors_global:
                        recall_fine[k:] += 1
                        break

        ave_recall_coarse = (recall_coarse / float(num_evaluated)) * 100.0
        ave_recall_fine = (recall_fine / float(num_evaluated)) * 100.0

        all_stats[angle] = {
            'ave_recall_coarse': ave_recall_coarse,
            'ave_recall_fine': ave_recall_fine
        }
        print(f"  --> Coarse Recall@1 for {angle}°: {ave_recall_coarse[0]:.2f}%")
        print(f"  --> Fine (Fusion) Recall@1 for {angle}°: {ave_recall_fine[0]:.2f}%")

    return all_stats


if __name__ == "__main__":
    config_file = '../config/config_cyd_cross.txt'
    model_config_file = '../models/minkloc_hybrid.txt'

    weights_file = '/home/wzj/pan1/MinkLocBevCrossRerank-main/weights/model_MinkLocHybrid_20260312_2309_final.pth'

    params = TrainingParams(config_file, model_config_file, debug=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_factory(params.model_params)

    try:
        model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"✅ 成功加载混合双流模型权重: {weights_file}")
    except Exception as e:
        print(f"⚠️ 警告: 加载权重失败。({e})")

    model.to(device)

    rotation_angles = [0, 5, 10, 15, 30, 45, 90, 180]
    print("\n" + "=" * 60)
    print("开始旋转鲁棒性测试 (Hybrid Dual-Stream + Late Fusion)")
    print("=" * 60)

    # 这里的 fusion_weight 可调
    all_stats = evaluate_cyd_with_rotation(model, device, params, rotation_angles, fusion_weight=0.5)

    print("\n" + "=" * 60)
    print("🔥 旋转鲁棒性测试最终汇总 (Late Fusion) 🔥:")
    if all_stats:
        for angle in rotation_angles:
            print(
                f"[{angle:>3}°] Coarse Recall@1: {all_stats[angle]['ave_recall_coarse'][0]:.2f}% | Fine Recall@1: {all_stats[angle]['ave_recall_fine'][0]:.2f}%")
    print("=" * 60)