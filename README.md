# MinkLocBevCrossRerank

本项目是一个针对地下巷道等几何退化环境的 3D 点云位置识别与回环检测网络。基于 MinkowskiEngine 稀疏卷积，项目采用 **Hybrid Dual-Stream (混合双流)** 粗精检索架构，显著提升了在旋转、视角偏移场景下的召回率。

## 🌟 核心特性 (Architecture)

1. **混合双流特征提取 (Hybrid Dual-Stream)**:
   * **粗流 (Coarse Stream)**: 采用 BEV (Bird's Eye View) 或横截面 (Cross-Section) 多层切片作为输入，通过 2D 稀疏卷积主干网络提取全局描述符（256维）。
   * **精流 (Fine Stream)**: 将点云沿中心线“拉直”后提取连续的横截面特征序列（Sequence Embeddings），保留了丰富的局部空间结构。
2. **旋转不变性设计**:
   * 在 BEV 量化阶段 (`BEVQuantizer`)，利用轨迹中心线进行偏航角自动对齐 (Yaw Alignment)，消除车头朝向差异。
   * 精流序列匹配采用 Soft-DTW (动态时间规整)，自带前向/逆向序列比对逻辑，完美适配巷道双向行驶导致的点云扫描顺序倒置问题。
3. **Coarse-to-Fine 检索机制**:
   * **阶段一**: 通过全局描述符构建 KD-Tree，快速召回 Top-K (如 25) 候选帧。
   * **阶段二**: 对候选帧的横截面序列使用 DTW 距离进行 Re-ranking（重排序），输出最终的匹配结果。

## 🚀 核心运行逻辑解析

### 1. 训练逻辑: `training/train_cyd_hybrid.py`
该脚本是启动混合双流模型训练的入口：
* **配置加载**: 读取训练超参 `config_cyd_cross.txt` 和模型架构参数 `minkloc_hybrid.txt`。
* **数据流构建**: 利用 `make_dataloaders` 构建包含正负样本三元组 (Triplets) 的 DataLoader。数据预处理阶段会将点云及对应的中心线数据传入 `MinkLocHybrid` 模型。
* **联合损失计算 (`DualStreamLoss`)**: 
  * 粗流计算 `TruncatedSmoothAP` 损失，优化全局描述符空间。
  * 精流在粗流挖掘出的最难正负样本 (Hardest Triplets) 上，利用 `BatchSoftDTW` 计算序列特征的 Margin Loss，并将两者通过权重融合后反向传播。

### 2. 基础评估逻辑: `eval/evaluate_cyd.py`
该脚本用于在 CYD 验证/测试集上评估模型的 Coarse-to-Fine 召回率：
* **特征提取**: 遍历 Database 和 Query 集合，一次性提取所有点云的全局特征 (`glob_emb`) 和序列特征 (`seq_emb`)。
* **粗检索 (Coarse Retrieval)**: 对 Database 的全局特征建立 `KDTree`，查询每个 Query 的最近的 25 个候选帧，计算 `Recall@N`。
* **精检索重排序 (Fine Re-ranking)**: 取出这 25 个候选帧的序列特征，使用 `compute_dtw_distance`（支持序列正反向双向匹配）计算 DTW 距离，按距离从小到大重新排序，得到最终的 Fine `Recall@1/5/10`。

### 3. 旋转鲁棒性评估: `eval/evaluate_cyd_rotation.py`
用于验证模型在面对偏航角 (Yaw) 剧烈变化时的鲁棒性：
* **基准库构建**: 在 0° 视角下提取 Database 的特征库并构建 KDTree。
* **多角度查询**: 循环遍历预设的旋转角度 `[0, 5, 10, 15, 30, 45, 90, 180]`，通过 `rotate_point_cloud_z` 对 Query 点云及中心线施加严格的 Z 轴物理旋转。
* **鲁棒性验证**: 旋转后的点云输入模型，由于模型内部具备中心线自适应偏航角对齐 (`BEVQuantizer` 内核) 和 DTW 的平移容忍性，即使在 180° 掉头情况下，依然能输出稳定且高精度的召回率统计。

## 💻 快速开始

**1. 准备数据集**
确保 CYD 数据集遵循以下目录结构，包含点云文件 (`.bin`) 及对应的中心线数据 (`_centerline.txt`)：
```text
CYD/cyd_NoRot_NoScale/
  ├── 100/
  │   ├── pointcloud_20m_10overlap/100001.bin
  │   └── centerline/100001_centerline.txt
  └── pointcloud_locations_20m_10overlap.csv
