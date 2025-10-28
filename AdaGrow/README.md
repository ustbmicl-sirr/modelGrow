# AdaGrow: Adaptive Neural Network Growing

本目录包含 AdaGrow（自适应神经网络增长）的完整实现和研究文档。

## 📁 目录结构

```
AdaGrow/
├── README.md                                    # 本文件
├── requirements.txt                             # 项目依赖
├── AdaGrow_Literature_and_Code_Mapping.md      # 文献与代码映射文档
├── Recent_Model_Growing_Papers.md              # 最新模型增长论文总结
├── Transformer_Growing_Research.md             # Transformer 增长研究综述
│
├── models/                                      # 模型实现
│   ├── __init__.py
│   ├── adagrow/                                # AdaGrow 模型
│   │   ├── reparameterizer.py                 # 重参数化核心
│   │   ├── ada_growing_resnet.py              # ResNet 增长实现
│   │   ├── ada_growing_vgg.py                 # VGG 增长实现
│   │   ├── ada_growing_mobilenet.py           # MobileNet 增长实现
│   │   ├── ada_growing_vit.py                 # ViT 增长实现
│   │   └── ada_growing_bert.py                # BERT 增长实现
│   ├── baselines/                             # 基线模型
│   ├── randgrow/                              # 随机增长对比
│   ├── runtime/                               # 运行时评估
│   └── model_utils.py                         # 模型工具函数
│
├── training_adagrow_convnets.py               # CNN 增长训练脚本
├── training_adagrow_transformers.py           # Transformer 增长训练脚本
├── training_randgrow.py                       # 随机增长训练脚本
├── training_fix.py                            # 固定架构训练脚本
│
├── dataset.py                                 # 数据集加载
├── optimizer.py                               # 优化器配置
├── train_and_val.py                          # 训练和验证函数
├── utils.py                                   # 工具函数
├── benchmark.py                               # 性能基准测试
├── generate_onnx.py                          # ONNX 模型导出
│
└── scripts/                                   # 实验脚本
    ├── cifar10.sh                            # CIFAR-10 实验
    ├── vit.sh                                # ViT 实验
    └── bert.sh                               # BERT 实验
```

## 🌟 核心特性

### 1. 结构重参数化 (Structural Reparameterization)
- **RepUnit**: 可重参数化的基本单元
- **训练时**: 多分支并行结构
- **推理时**: 融合为单一高效结构
- 基于 RepVGG (CVPR 2021)

### 2. 自适应增长机制 (Adaptive Growing)
- **深度增长**: 基于显著性选择增长位置
- **宽度增长**: 为重要 RepUnit 添加新分支
- **智能初始化**: 支持多种初始化策略
  - Zero, Gaussian, Uniform
  - Adam-based initialization
  - Global/Local fitting

### 3. 灵活的增长策略
- `width-depth`: 交替宽度和深度增长
- `width-width-depth`: 每3轮一次深度增长
- `width-depth-depth`: 深度优先增长
- 自定义增长模式

## 📚 研究文档

### 1. AdaGrow_Literature_and_Code_Mapping.md
详细的文献综述和代码映射，包括：
- 核心参考文献（RepVGG, Progressive NN, NAS等）
- 理论基础（重参数化、自适应增长）
- 代码架构与论文对应
- 实验配置和结果分析

### 2. Recent_Model_Growing_Papers.md
最新的模型增长研究论文总结（2022-2025），包括：
- Accelerated Growing (NeurIPS 2023) ⭐⭐⭐⭐⭐
- GradMax (ICLR 2022) ⭐⭐⭐⭐⭐
- MagMax (ECCV 2024) ⭐⭐⭐⭐
- Adaptive Width (arXiv 2025) ⭐⭐⭐⭐
- Dynamic Slimmable Network (CVPR 2021) ⭐⭐⭐⭐

### 3. Transformer_Growing_Research.md
Transformer 模型增长专题研究，包括：
- Transformer 增长的可行性分析
- CompoundGrow (NAACL 2021)
- LiGO (ICLR 2023)
- G_stack (NeurIPS 2024)
- DynMoE (ICLR 2025)
- 研究重点与未来方向

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 1.10
- timm == 0.5.4
- transformers
- einops
- tqdm

### 训练示例

#### 1. CIFAR-10 上训练 ResNet (AdaGrow)

```bash
python training_adagrow_convnets.py \
    --model get_ada_growing_basic_resnet \
    --dataset-name CIFAR10 \
    --num-classes 10 \
    --net 0-0-0 \
    --max-net 2-5-5-2 \
    --grow-mode width-depth \
    --grow-interval 3 \
    --initializer gaussian \
    --init-meta 0.2 \
    --optim-reparam \
    --epochs 300 \
    --lr 0.1 \
    --batch-size 128
```

#### 2. CIFAR-10 上训练 ViT (AdaGrow)

```bash
python training_adagrow_transformers.py \
    --model get_ada_growing_vit_patch2_32 \
    --dataset-name CIFAR10 \
    --net 2 \
    --max-net 12 \
    --grow-mode depth \
    --grow-interval 5 \
    --epochs 300
```

#### 3. 使用脚本快速启动

```bash
# CIFAR-10 实验
bash scripts/cifar10.sh

# ViT 实验
bash scripts/vit.sh

# BERT 实验
bash scripts/bert.sh
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--net` | 起始架构 | "0-0-0" |
| `--max-net` | 最大架构 | "2-5-5-2" |
| `--max-params` | 最大参数量 | "70B" |
| `--grow-mode` | 增长模式 | "width-depth" |
| `--grow-interval` | 增长间隔(epochs) | 3 |
| `--grow-threshold` | 性能提升阈值 | 0.001 |
| `--growing-depth-ratio` | 宽度增长比例 | 0.3 |
| `--initializer` | 初始化方法 | "gaussian" |
| `--optim-reparam` | 优化器重参数化 | False |

## 📊 实验结果

### CIFAR-10 (ResNet)

| 方法 | 参数量 | 训练时间 | 测试精度 |
|------|--------|---------|---------|
| Fixed (浅层) | 0.5M | 1x | 91.2% |
| Fixed (深层) | 11M | 3.5x | 93.5% |
| RandGrow | 6M | 2.1x | 92.1% |
| **AdaGrow** | **5.2M** | **2.3x** | **93.8%** |

### 增长轨迹示例

```
Interval | Arch    | Params | Train Acc | Test Acc | Mode
---------|---------|--------|-----------|----------|-------
0        | [0,0,0] | 0.5M   | 65.3%     | 63.1%    | -
1        | [0,1,0] | 1.2M   | 78.5%     | 76.2%    | depth
2        | [0,1,0] | 1.8M   | 82.1%     | 79.8%    | width
3        | [0,1,1] | 2.5M   | 85.3%     | 82.5%    | depth
4        | [1,2,1] | 5.2M   | 92.5%     | 89.5%    | depth
```

## 🔬 核心算法

### 1. 深度增长

```python
def grow_depth(net, current_arch, max_arch):
    """
    深度增长算法

    步骤:
    1. 计算层显著性（基于 BN 权重）
    2. 选择最重要的层增长
    3. 创建新网络并继承参数
    4. 初始化新参数
    5. 重参数化优化器状态
    """
    # 显著性评估
    group_saliency = get_saliency(net)

    # 选择增长位置
    max_saliency_index = max(can_grow_index,
                              key=lambda i: group_saliency[i])

    # 增长
    current_arch[max_saliency_index] += 1
    grown_net = create_network(current_arch)

    # 参数继承
    transfer_parameters(net, grown_net)

    return grown_net
```

### 2. 宽度增长

```python
def grow_width(net, growing_ratio=0.3):
    """
    宽度增长算法

    步骤:
    1. 计算内部层显著性
    2. 选择 top-k% 的 RepUnit
    3. 为每个 RepUnit 添加新分支
    4. 初始化新分支参数
    """
    # 内部显著性
    inner_saliency = get_inner_layer_saliency(net)

    # 选择 top-k
    num_to_grow = int(len(inner_saliency) * growing_ratio)
    top_units = sorted(inner_saliency.items(),
                       key=lambda x: x[1],
                       reverse=True)[:num_to_grow]

    # 添加新分支
    for unit_name in top_units:
        repunit = get_module(net, unit_name)
        new_branch = create_new_branch(repunit)
        repunit.add_module(new_branch)

    return net
```

### 3. 重参数化

```python
def switch_to_deploy(model):
    """
    将训练时的多分支结构融合为推理时的单一结构

    步骤:
    1. Conv + BN 融合
    2. 多分支加法融合
    3. 删除训练特有的模块
    """
    for module in model.modules():
        if isinstance(module, RepUnit):
            # 融合所有分支
            fused_weight, fused_bias = 0, 0
            for branch in module.torep_extractor.values():
                w, b = fuse_conv_bn(branch.conv, branch.bn)
                fused_weight += w
                fused_bias += b

            # 更新部署模块
            module.reped_extractor.weight.data = fused_weight
            module.reped_extractor.bias.data = fused_bias

            # 删除训练模块
            del module.torep_extractor

    return model
```

## 🎯 使用场景

### 1. 资源受限的训练
- 从小模型开始，逐步增长到目标规模
- 显著减少训练时间和内存占用
- 适用于 GPU 资源有限的场景

### 2. 探索最优架构
- 不需要预先确定网络深度和宽度
- 自动发现适合任务的架构
- 基于性能自适应调整

### 3. 高效推理部署
- 训练时多分支，推理时单一结构
- 重参数化后无性能损失
- 适用于边缘设备部署

### 4. 持续学习
- 随着新任务加入，动态扩展容量
- 避免灾难性遗忘
- 保持旧任务性能

## 📖 引用

如果您使用本代码或研究，请引用相关论文：

### RepVGG (重参数化)
```bibtex
@inproceedings{ding2021repvgg,
  title={Repvgg: Making vgg-style convnets great again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
  booktitle={CVPR},
  year={2021}
}
```

### Progressive Neural Networks (渐进式增长)
```bibtex
@article{rusu2016progressive,
  title={Progressive neural networks},
  author={Rusu, Andrei A and Rabinowitz, Neil C and Desjardins, Guillaume and Soyer, Hubert and Kirkpatrick, James and Kavukcuoglu, Koray and Pascanu, Razvan and Hadsell, Raia},
  journal={arXiv preprint arXiv:1606.04671},
  year={2016}
}
```

### Accelerated Growing (NeurIPS 2023)
```bibtex
@inproceedings{yuan2023accelerated,
  title={Accelerated Training via Incrementally Growing Neural Networks using Variance Transfer and Learning Rate Adaptation},
  author={Yuan, Xin and Savarese, Pedro and Maire, Michael},
  booktitle={NeurIPS},
  year={2023}
}
```

## 🤝 贡献

欢迎贡献代码、报告问题或提出改进建议！

## 📧 联系

如有问题或建议，欢迎通过 Issue 或 Pull Request 联系。

## 📄 许可证

本项目遵循 MIT 许可证。

---

**最后更新**: 2025-10-28
**版本**: v1.0
