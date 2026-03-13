# datasets/pointnetvlad/pnv_raw.py
import numpy as np
import os
from datetime import datetime
from datasets.base_datasets import PointCloudLoader


class CYDPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None
        self.point_count_stats = []
        self.log_file = "/home/wzj/pan1/MinkLoc3dv2_Chilean_原始点云/training/pnv_raw.log"

        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def read_pc(self, file_pathname: str) -> np.ndarray:
        pc = np.fromfile(file_pathname, dtype=np.float64)
        pc = np.float32(pc)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))

        MAX_POINTS = 20000
        if pc.shape[0] > MAX_POINTS:
            indices = np.random.choice(pc.shape[0], MAX_POINTS, replace=False)
            pc = pc[indices]

        num_points = pc.shape[0]
        self.point_count_stats.append(num_points)

        return pc

    def read_centerline(self, file_pathname: str) -> np.ndarray:
        """
        根据点云路径自动推断中心线路径并读取
        例如输入: /.../102/pointcloud_20m_10overlap/102003.bin
        寻找中心线: /.../102/centerline/102003_centerline.txt
        """
        # 解析路径目录
        dir_path = os.path.dirname(file_pathname)  # /.../102/pointcloud_20m_10overlap
        session_dir = os.path.dirname(dir_path)  # /.../102

        # 解析文件名
        filename = os.path.basename(file_pathname)  # 102003.bin
        basename = os.path.splitext(filename)[0]  # 102003

        # 拼凑中心线路径
        cl_path = os.path.join(session_dir, "centerline", f"{basename}_centerline.txt")

        if not os.path.exists(cl_path):
            raise FileNotFoundError(f"中心线文件未找到，请检查数据集完整性: {cl_path}")

        centerline = np.loadtxt(cl_path)
        return centerline