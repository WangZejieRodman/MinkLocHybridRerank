# datasets/cyd/generate_training_tuples_cyd.py
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import sys

# 添加项目根目录到环境变量，以便导入 datasets 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets.base_datasets import TrainingTuple

# ================= 数据集路径配置 =================
BASE_PATH = "/home/wzj/pan2/巷道/CYD/"
RUNS_FOLDER = "cyd_NoRot_NoScale/"
FILENAME = "pointcloud_locations_20m_10overlap.csv"
POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"

# ================= 划分策略 =================
TRAIN_SESSION_START = 100
TRAIN_SESSION_END = 108
TEST_SESSION_START = 109
TEST_SESSION_END = 113

POSITIVE_THRESHOLD = 7  # 7米内为正样本
NEGATIVE_THRESHOLD = 35  # 35米外为负样本


def check_in_test_set_by_session(session_id):
    return TEST_SESSION_START <= int(session_id) <= TEST_SESSION_END


def check_in_train_set_by_session(session_id):
    return TRAIN_SESSION_START <= int(session_id) <= TRAIN_SESSION_END


def construct_query_dict(df_centroids, filename, ind_nn_r=POSITIVE_THRESHOLD, ind_r_r=NEGATIVE_THRESHOLD):
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)

    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]

        scan_filename = os.path.split(query)[1]
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        positives = np.sort(positives[positives != anchor_ndx])
        non_negatives = np.sort(ind_r[anchor_ndx])

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=query,
            positives=positives,
            non_negatives=non_negatives,
            position=anchor_pos
        )

    file_path = os.path.join(os.path.dirname(__file__), filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 完成: {filename} (样本数: {len(queries)})")


if __name__ == '__main__':
    full_path = os.path.join(BASE_PATH, RUNS_FOLDER)
    print(f"开始处理 CYD 数据集: {full_path}")

    valid_folders = sorted([f for f in os.listdir(full_path) if f.isdigit()], key=int)

    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in valid_folders:
        csv_path = os.path.join(full_path, folder, FILENAME)
        if not os.path.exists(csv_path):
            continue

        df_locations = pd.read_csv(csv_path, sep=',')
        # 拼接相对路径: cyd_NoRot_NoScale/100/pointcloud_20m_10overlap/xxx.bin
        # 注意：不要加 centerline，之前我们写的 loader 会自动根据这个 .bin 路径去找对应的 centerline
        df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(
            str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        session_num = int(folder)
        if check_in_train_set_by_session(session_num):
            df_train = pd.concat([df_train, df_locations], ignore_index=True)
            print(f"  [Train] 加入 Session {folder}: {len(df_locations)} 帧")
        elif check_in_test_set_by_session(session_num):
            df_test = pd.concat([df_test, df_locations], ignore_index=True)
            print(f"  [Test] 加入 Session {folder}: {len(df_locations)} 帧")

    print("\n生成查询字典中...")
    if len(df_train) > 0: construct_query_dict(df_train, "training_queries_cyd.pickle")
    if len(df_test) > 0: construct_query_dict(df_test, "test_queries_cyd.pickle")