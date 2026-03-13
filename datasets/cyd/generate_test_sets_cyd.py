# datasets/cyd/generate_test_sets_cyd.py
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle

BASE_PATH = "/home/wzj/pan2/巷道/CYD/"
RUNS_FOLDER = "cyd_NoRot_NoScale/"
FILENAME = "pointcloud_locations_20m_10overlap.csv"
POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"

# 模拟：109-111为历史数据库，112-113为新查询
DATABASE_SESSION_START = 100
DATABASE_SESSION_END = 108
QUERY_SESSION_START = 109
QUERY_SESSION_END = 113
POSITIVE_THRESHOLD = 7


def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 保存: {filename}")


if __name__ == "__main__":
    database_sets = []
    database_coordinates_list = []
    query_sets = []

    print("构建 Database 集合...")
    for folder in range(DATABASE_SESSION_START, DATABASE_SESSION_END + 1):
        folder_str = str(folder)
        csv_path = os.path.join(BASE_PATH, RUNS_FOLDER, folder_str, FILENAME)
        if not os.path.exists(csv_path): continue

        df = pd.read_csv(csv_path, sep=',')
        database = {}
        valid_coords = []
        for idx, row in df.iterrows():
            rel_path = f"{RUNS_FOLDER}{folder_str}{POINTCLOUD_FOLS}{int(row['timestamp'])}.bin"
            database[len(database)] = {'query': rel_path, 'northing': row['northing'], 'easting': row['easting']}
            valid_coords.append([row['northing'], row['easting']])

        database_sets.append(database)
        database_coordinates_list.append(np.array(valid_coords))
        print(f"  加入 Database {folder_str}: {len(database)} 帧")

    print("构建 Query 集合...")
    for folder in range(QUERY_SESSION_START, QUERY_SESSION_END + 1):
        folder_str = str(folder)
        csv_path = os.path.join(BASE_PATH, RUNS_FOLDER, folder_str, FILENAME)
        if not os.path.exists(csv_path): continue

        df = pd.read_csv(csv_path, sep=',')
        queries = {}
        for idx, row in df.iterrows():
            rel_path = f"{RUNS_FOLDER}{folder_str}{POINTCLOUD_FOLS}{int(row['timestamp'])}.bin"
            queries[len(queries)] = {'query': rel_path, 'northing': row['northing'], 'easting': row['easting']}
        query_sets.append(queries)
        print(f"  加入 Query {folder_str}: {len(queries)} 帧")

    # 构建正样本对应关系
    database_trees = [KDTree(coords) if len(coords) > 0 else None for coords in database_coordinates_list]

    for i, (db_tree, db_set) in enumerate(zip(database_trees, database_sets)):
        if db_tree is None: continue
        for query_set in query_sets:
            for key in query_set.keys():
                if 'positives' not in query_set[key]: query_set[key]['positives'] = {}
                q_coord = np.array([[query_set[key]["northing"], query_set[key]["easting"]]])
                positive_indices = db_tree.query_radius(q_coord, r=POSITIVE_THRESHOLD)[0].tolist()
                query_set[key]['positives'][i] = positive_indices

    output_dir = os.path.dirname(__file__)
    db_file = os.path.join(output_dir,
                           f'cyd_evaluation_database_{DATABASE_SESSION_START}_{DATABASE_SESSION_END}.pickle')
    q_file = os.path.join(output_dir, f'cyd_evaluation_query_{QUERY_SESSION_START}_{QUERY_SESSION_END}.pickle')

    output_to_file(database_sets, db_file)
    output_to_file(query_sets, q_file)