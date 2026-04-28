# Hbb Classifier (WH→bb Analysis)

本项目包含针对 $H \to b\bar{b}$ 过程（特别是 $WH$ 通道）的分类器开发代码，对比了深度学习（DNN）和传统机器学习（TMVA BDT）两种方法。

---

## 📂 项目结构

```
hbb_classifier/
├── README.md                  # 本说明文档
├── DNN/                       # 深度学习分类器（PyTorch）
│   ├── baseline/              # 基准训练流程
│   │   ├── prepare_whbb_dnn_data.py       # 数据预处理：ROOT → NPZ
│   │   └── train_whbb_dnn_2fold_binnedZ.py # 2-fold CV 训练 + Binned Z 评估
│   └── tuning/                # 超参数调优
│       └── tune_whbb_dnn_lr_batchsize.py  # LR & Batch Size 网格搜索
└── tmva/                      # TMVA 分类器（ROOT）
    ├── final_tuning/          # 最终调优结果（BDT 参数网格扫描）
    ├── tmva_4var ~ tmva_12vars/ # 不同变量数量的训练配置
    └── tmva_2fold_2nJ/        # 2-fold 交叉验证配置
```

---

## 🚀 DNN 部分使用指南

### 1. 环境依赖

```bash
pip install torch numpy scipy scikit-learn uproot matplotlib
```

**推荐配置：**
- Python 3.8+
- PyTorch 1.10+（支持 CPU/CUDA/MPS）
- Uproot 4.x（用于读取 ROOT 文件）

### 2. 数据准备

将 ROOT Ntuple 转换为 NumPy `.npz` 格式以加速训练：

```bash
cd DNN/baseline
python prepare_whbb_dnn_data.py
```

**主要功能：**
- 从 ROOT 文件分块读取数据（支持大文件）
- 应用基础选择条件：`nTags == 2` 且 `nJ == 2`
- 构造二分类标签：
  - Signal: `sample == "qqWlvH125"` → 1
  - Background: 其他样本 → 0
- 保存特征矩阵、标签、事件权重、事件编号等
- 输出文件：`whbb_dnn_prepared.npz`

**输入变量（9个）：**
```python
feature_names = [
    "mBB",           # bb 系统质量
    "dRBB",          # 两个 b-jet 的 ΔR
    "pTV",           # 矢量玻色子横动量
    "pTB1", "pTB2",  # 第一、第二 b-jet 的横动量
    "bin_MV2c10B1", "bin_MV2c10B2",  # b-tagging discriminant
    "dPhiVBB",       # 矢量玻色子与 bb 系统的 Δφ
    "MET"            # 缺失横能量
]
```

### 3. 基准训练（Baseline Training）

使用默认超参数进行 2-fold 交叉验证训练：

```bash
python train_whbb_dnn_2fold_binnedZ.py
```

**模型架构：**
```
Input (10) → Linear(64) → ReLU → Dropout(0.1) 
         → Linear(64) → ReLU → Dropout(0.1)
         → Linear(32) → ReLU
         → Linear(1) → Sigmoid
```

**训练配置：**
- **优化器**: Adam (lr=1e-3)
- **损失函数**: Weighted Binary Cross Entropy
- **批次大小**: 8192
- **训练轮数**: 20 epochs
- **验证集比例**: 10%
- **数据预处理**: 
  - Clip: 连续变量裁剪至 0.1% - 99.9% 分位数
  - Standardization: Z-score 标准化

**评估指标：**
- **AUC** (Area Under ROC Curve)
- **Binned Asimov Significance** ($Z_A$):
  $$Z_A = \sqrt{2 \sum_{i=1}^{N_{bins}} \left[ (s_i + b_i) \ln\left(1 + \frac{s_i}{b_i}\right) - s_i \right]}$$
  其中 $s_i$ 和 $b_i$ 为第 $i$ 个 bin 中的加权信号和背景事件数（默认 20 bins）

**输出文件：**
```
whbb_dnn_training_output_binnedZ/
├── Even_train_Odd_test_model.pt          # Fold A 模型权重
├── Odd_train_Even_test_model.pt          # Fold B 模型权重
├── *_loss.png, *_val_auc.png             # 训练曲线
├── *_roc.png                             # ROC 曲线
├── *_score_dist.png                      # Score 分布
├── *_score_binned_shapes.png             # Binned 分布
├── *_preprocessing.json                  # 预处理参数
├── *_summary.json                        # 单 fold 结果
├── combined_roc.png                      # 合并 ROC
├── combined_score_binned_shapes.png      # 合并 Binned 分布
├── whbb_dnn_oof_scores.npz               # Out-of-fold scores
└── combined_summary.json                 # 最终汇总结果
```

**基准性能（参考 `base.log`）：**
- Combined AUC: **0.8473**
- Combined Binned Asimov Z (20 bins): **4.03**

### 4. 超参数调优（Hyperparameter Tuning）

对学习率（LR）和批次大小（Batch Size）进行网格搜索：

```bash
cd ../tuning
python tune_whbb_dnn_lr_batchsize.py
```

**默认搜索空间：**
- Learning Rates: `[3e-4, 1e-3, 3e-3]`
- Batch Sizes: `[8192, 16384, 32768]`
- Early Stopping Patience: 5 epochs（基于 Validation AUC）

**自定义搜索空间：**
```bash
python tune_whbb_dnn_lr_batchsize.py \
    --lrs "1e-4, 5e-4, 1e-3" \
    --batch-sizes "4096, 8192, 16384" \
    --epochs 30 \
    --early-stop-patience 10
```

**输出文件：**
```
lr_batchsize_tuning_output/
├── lr_3e-04_bs_8192.json          # 每个 trial 的详细结果
├── lr_3e-04_bs_16384.json
├── ...
├── all_results.json               # 所有结果（按 Binned Z 排序）
└── best_result.json               # 最佳配置
```

**当前最佳结果（参考 `best_result.json`）：**
| 超参数 | 值 |
|--------|-----|
| Learning Rate | 0.0003 |
| Batch Size | 16384 |
| Combined AUC | 0.8457 |
| **Combined Binned Asimov Z** | **4.47** |

> **注意**：虽然该配置的 AUC 略低于基准，但 Binned Asimov Z 更高，说明在物理分析中更具发现潜力。

---

## 📊 TMVA 部分说明

`tmva/` 目录包含使用 ROOT TMVA 工具包进行的 Boosted Decision Tree (BDT) 分析。

### 目录结构

```
tmva/
├── tmva_4var/     # 4 个输入变量的配置
├── tmva_5vars/    # 5 个输入变量的配置
├── ...
├── tmva_12vars/   # 12 个输入变量的配置
├── tmva_2fold_2nJ/ # 2-fold 交叉验证（nJ=2）
└── final_tuning/  # 最终 BDT 参数调优
    └── grid_scan_output/
        ├── NTrees400_MaxDepth3_Beta0p20/
        ├── NTrees400_MaxDepth3_Beta0p30/
        ├── ...
        └── NTrees1200_MaxDepth5_Beta0p50/
```

### 典型工作流程

#### 1. 训练单个分类器

```bash
cd tmva/tmva_4var/MET_plus
root -l 'MET_plus.C'
```

这将执行：
- 读取 ROOT 数据集（Even/Odd folds）
- 训练 BDT 分类器
- 生成权重文件（`dataset_Even/weights/`, `dataset_Odd/weights/`）
- 输出训练诊断图

#### 2. 计算显著性

```bash
root -l 'CalculateKSignificance.C'
```

或使用通用脚本：
```bash
cd tmva/tmva_4var
root -l 'calculate.C'
```

### BDT 参数网格扫描

`final_tuning/grid_scan_output/` 包含了针对不同 BDT 超参数的系统性扫描：

**扫描维度：**
- **NTrees**: [400, 800, 1200]
- **MaxDepth**: [3, 4, 5]
- **BoostType (Beta)**: [0.20, 0.30, 0.50]

**共 27 种配置组合**，每种配置独立训练并评估。

### 关键变量演进

| 配置 | 变量数量 | 典型变量 |
|------|---------|---------|
| `tmva_4var` | 4 | MET, MEff, dEtaBB, pTB2 |
| `tmva_5vars` | 5 | + bin_MV2c10B1, bin_MV2c10B2 |
| `tmva_6vars` | 6 | + pTB1 |
| `tmva_7vars` | 7 | + sumPtJets |
| `tmva_8vars` | 8 | + dPhiVBB |
| `tmva_9vars` | 9 | + mBB |
| `tmva_10vars` | 10 | + dRBB |
| `tmva_11vars` | 11 | + pTV |
| `tmva_12vars` | 12 | 全部变量 |

---

## 🔬 方法对比

| 特性 | DNN (PyTorch) | TMVA (BDT) |
|------|---------------|------------|
| **框架** | PyTorch | ROOT TMVA |
| **模型类型** | 全连接神经网络 | Gradient Boosted Decision Trees |
| **训练速度** | 较快（GPU 加速） | 较慢（CPU） |
| **可解释性** | 较低（黑盒） | 较高（特征重要性） |
| **超参数调优** | LR, Batch Size, Architecture | NTrees, MaxDepth, BoostType |
| **最佳 Binned Z** | ~4.47 | 需查看 `final_tuning` 结果 |
| **优势** | 灵活、易扩展、支持大规模数据 | 成熟稳定、物理分析标准工具 |

---

## 📝 关键概念解释

### Binned Asimov Significance

与传统 AUC 不同，Binned Asimov Z 直接衡量物理分析中的**发现潜力**：

- **AUC**: 衡量整体区分能力（0.5 = 随机猜测，1.0 = 完美分离）
- **Binned Z**: 考虑事件权重和 binning 策略，更接近实际物理显著性

公式推导基于 Cowan et al. (2011) 的 Asimov 数据集方法，广泛应用于高能物理实验。

### 2-Fold Cross Validation

基于事件编号奇偶性划分：
- **Fold A**: Even events 训练 → Odd events 测试
- **Fold B**: Odd events 训练 → Even events 测试

**优势：**
- 避免数据泄露
- 充分利用所有数据
- 提供无偏的性能估计

### 数据预处理

1. **Clipping**: 去除极端离群值（0.1% - 99.9% 分位数）
2. **Standardization**: Z-score 归一化（mean=0, std=1）
3. **Weighted Loss**: 考虑事件权重，避免类别不平衡问题

---

## ⚠️ 注意事项

1. **随机种子**: 所有实验设置 `seed=12345` 以保证可复现性
2. **设备支持**: DNN 代码自动检测并使用 `mps` (Mac Silicon), `cuda` (NVIDIA GPU) 或 `cpu`
3. **数据路径**: 修改 `prepare_whbb_dnn_data.py` 中的 `filename` 参数指向您的 ROOT 文件
4. **内存需求**: 
   - 数据准备: ~10 GB RAM（23M 事件）
   - DNN 训练: ~8 GB RAM（取决于 batch size）
5. **TMVA 依赖**: 需要安装 ROOT 6.x 并配置 TMVA

---

## 📚 相关资源

以下是在本项目中使用的工具和算法的相关文档：

### 统计方法
- **Asimov Significance**: 基于 Cowan et al. (2011) 提出的 Asimov 数据集方法，广泛应用于高能物理实验中的显著性计算
  - 公式来源：Cowan, G., Cranmer, K., Gross, E., & Vitells, O. "Asymptotic formulae for likelihood-based tests of new physics", Eur. Phys. J. C 71 (2011) 1554

### 工具框架
- **ROOT TMVA**: ROOT 的多变量分析工具包
  - 官方文档：https://root.cern.ch/root/htmldoc/guides/tmva/TMVAUsersGuide.html
  
- **PyTorch**: 深度学习框架
  - 官方文档：https://pytorch.org/docs/

- **Uproot**: Python 读取 ROOT 文件的库
  - 官方文档：https://uproot.readthedocs.io/

> **注意**：以上参考文献是项目中使用的算法和工具的通用文档，具体实现细节请参考各工具的官方文档.



## 👥 贡献者

- 项目开发：HaoranYou
- 最后更新：2026-04-28

---

## 📄 许可证

本项目仅供学术研究使用。
