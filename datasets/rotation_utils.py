import numpy as np


def rotate_point_cloud_z(pc, angle_deg):
    """
    绕z轴旋转点云

    Args:
        pc: numpy数组 (N, 3) - 点云坐标 [x, y, z]
        angle_deg: float - 旋转角度（度）

    Returns:
        rotated_pc: numpy数组 (N, 3) - 旋转后的点云
    """
    # 转换为弧度
    angle_rad = np.deg2rad(angle_deg)

    # 构造绕z轴的旋转矩阵
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # 应用旋转: pc @ R^T (因为点云是行向量)
    rotated_pc = pc @ rotation_matrix.T

    return rotated_pc


def rotate_point_cloud_batch(pc_list, angle_deg):
    """
    批量旋转点云

    Args:
        pc_list: list of numpy数组 - 点云列表
        angle_deg: float - 旋转角度（度）

    Returns:
        rotated_list: list of numpy数组 - 旋转后的点云列表
    """
    return [rotate_point_cloud_z(pc, angle_deg) for pc in pc_list]