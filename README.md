# MinkLocHybridRerank

本项目是一个针对地下巷道等几何退化环境的 3D 点云位置识别与回环检测网络。基于 MinkowskiEngine 稀疏卷积，项目采用了先进的 **Hybrid Dual-Stream (混合双流)** 架构。通过引入 **特征级融合与动态门控 (Feature-level Fusion + Dynamic Gating)** 以及 **Late Fusion 联合评分** 机制，模型在旋转、视角偏移及双向行驶场景下的召回率和鲁棒性得到了显著提升。

## 🌟 核心特性 (Architecture)

1. **粗流：特征级融合与动态门控 (Coarse Stream with Dynamic Gating)**:
   * **双视角并行提取**：同时从 BEV (Bird's Eye View) 和横截面 (Cross-Section) 两个视角利用 2D 稀疏卷积主干网络提取特征。
   * **动态门控机制**：将 BEV 和横截面特征拼接后，通过一个基于 MLP 的动态门控网络（Gating Network）自适应地计算两个视角的权重 (`w_bev` 和 `w_cross`)。
   * **加权融合**：根据生成的权重将双视角特征进行动态加权融合，输出最终具有极强鲁棒性的 256 维全局描述符。

2. **精流：序列切片提取 (Fine Stream with Slice Sequence)**:
   * 将点云沿中心线拉直，利用轻量级 2D 稀疏卷积提取连续的横截面特征序列（Sequence Embeddings），保留丰富的局部空间结构。

3. **严格的旋转不变性设计**:
   * **偏航角对齐**：在 BEV 量化阶段 (`BEVQuantizer`)，利用轨迹中心线进行偏航角自动对齐 (Yaw Alignment)，消除车头朝向差异带来的影响。
   * **双向行驶自适应**：精流序列匹配采用 Soft-DTW (动态时间规整)，支持前向与逆向的双向序列比对，完美适配巷道掉头导致的点云扫描顺序倒置问题。

4. **Coarse-to-Fine 与 Late Fusion 联合评分**:
   * **阶段一 (Coarse)**: 通过粗流融合的全局描述符构建 KD-Tree，快速召回 Top-K (如 25) 候选帧。
   * **阶段二 (Fine)**: 对候选帧的横截面序列使用 DTW 计算距离。
   * **后期融合 (Late Fusion)**: 将粗流欧氏距离与精流 DTW 距离进行 Min-Max 归一化，通过可调权重进行联合评分 (`JointScore`)，按融合得分输出最终的重排序结果。

## 🚀 核心运行逻辑解析

### 1. 训练逻辑: `training/train_cyd_hybrid.py`
启动混合双流模型训练的核心入口：
* **数据流与架构**：解析 `minkloc_hybrid.txt` 配置，初始化包含双粗流（BEV+Cross）和单精流的 `MinkLocHybrid` 网络。
* **联合损失计算 (`DualStreamLoss`)**：
  * 粗流计算 `TruncatedSmoothAP` 损失，优化融合后的全局描述符空间。
  * 精流在粗流挖掘出的最难正负样本上，利用 `BatchSoftDTW` 计算序列特征的 Triplet Margin Loss，两者加权后进行反向传播。
* **门控权重监控**：训练过程会实时输出 `w_bev` 和 `w_cross` 的动态权重分配情况，便于观测模型对不同视角的依赖度。

### 2. 联合评分评估逻辑: `eval/evaluate_cyd_JointScore.py`
在 CYD 测试集上评估模型的 Coarse-to-Fine 召回率，并默认开启 Late Fusion：
* **特征提取**：一次性提取 Database 和 Query 集合的所有融合全局特征 (`glob_emb`) 和序列特征 (`seq_emb`)。
* **粗检索**：通过全局特征建立 `KDTree` 召回候选帧，获取粗流欧氏距离。
* **精检索与 Late Fusion**：计算候选序列的正/反向 DTW 距离，将归一化后的粗流距离与精流 DTW 距离进行按权重 (`fusion_weight`) 的联合评估，输出最终重排序的 `Recall@1/5/10`。

### 3. 旋转鲁棒性评估: `eval/evaluate_cyd_rotation_JointScore.py`
验证模型在严重偏航角 (Yaw) 变化时的稳定性：
* **基准库**：使用 0° 视角构建 Database 特征库。
* **物理旋转查询**：对 Query 点云施加 `[0, 5, 10, 15, 30, 45, 90, 180]` 度的严格 Z 轴旋转。
* **验证**：得益于中心线对齐和 DTW 平移容忍度，结合 Late Fusion，模型在 180° 完全掉头的情况下依然能保持高精度的召回率表现。

## 💻 快速开始

**1. 准备数据集**
确保 CYD 数据集遵循以下目录结构，包含点云文件 (`.bin`) 及对应的中心线数据 (`_centerline.txt`)：
```text
CYD/cyd_NoRot_NoScale/
  ├── 100/
  │   ├── pointcloud_20m_10overlap/100001.bin
  │   └── centerline/100001_centerline.txt
  └── pointcloud_locations_20m_10overlap.csv
```

**2. 生成训练/测试元数据**

```bash
python datasets/cyd/generate_training_tuples_cyd.py
python datasets/cyd/generate_test_sets_cyd.py
```

**3. 启动训练**

```bash
python training/train_cyd_hybrid.py
```

**4. 执行评估**

```bash
python eval/evaluate_cyd_JointScore.py
```
