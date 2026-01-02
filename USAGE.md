# SMPL-X Contact Prediction - Usage Guide

## 项目结构

```
Contact/
├── configs/
│   └── default.yaml          # 配置文件
├── data/
│   ├── __init__.py
│   └── dataset.py            # 数据集加载
├── models/
│   ├── __init__.py
│   ├── backbone.py           # ResNet18特征提取器（冻结）
│   ├── geometry.py           # 几何处理模块
│   └── contact_net.py        # 主模型
├── utils/
│   ├── __init__.py
│   ├── geometry_utils.py     # 几何工具函数
│   └── visualization.py      # 可视化工具
├── train.py                  # 训练脚本
├── test_system.py            # 系统测试脚本
├── requirements.txt          # 依赖列表
├── USAGE.md                  # 本文件
└── README.md                 # 项目说明文档
```

## 环境配置

### 1. 创建Python环境

```bash
# 使用conda（推荐）
conda create -n contact python=3.9
conda activate contact

# 或使用venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
# 首先安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch torchvision

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 准备SMPL-X模型文件

1. 访问 [SMPL-X官网](https://smpl-x.is.tue.mpg.de/) 注册并下载模型文件
2. 将下载的文件放在 `smplx_models/` 目录下
3. 确保目录包含以下文件：
   - `SMPLX_NEUTRAL.pkl`
   - `SMPLX_MALE.pkl`
   - `SMPLX_FEMALE.pkl`

```bash
mkdir -p smplx_models
# 将下载的文件复制到这里
```

### 4. 准备数据集

数据集目录结构应为：

```
data_contact/
├── category1/
│   ├── id1/
│   │   ├── image.jpg
│   │   ├── smplx_parameters.json
│   │   ├── contact.json
│   │   ├── box_annotation.json
│   │   ├── calibration.json
│   │   ├── extrinsic.json
│   │   └── normals_smplx.npy (可选)
│   ├── id2/
│   └── ...
├── category2/
└── ...
```

## 配置说明

编辑 `configs/default.yaml` 来调整超参数：

### 关键配置项

```yaml
data:
  root_dir: "data_contact"           # 数据集根目录
  smplx_model_path: "smplx_models"   # SMPL-X模型文件路径
  img_size: [512, 512]               # 图像尺寸

training:
  batch_size: 8                      # 批次大小（根据GPU内存调整）
  num_epochs: 100                    # 训练轮数
  learning_rate: 1e-3                # 学习率
  pos_weight: 5.0                    # 正样本权重（针对类别不平衡）
  save_frequency: 5                  # 每N个epoch保存一次
  
device: "cuda"                       # 使用 "cuda" 或 "cpu"
```

## 使用流程

### 1. 系统测试

在开始训练前，运行测试脚本验证环境配置：

```bash
python test_system.py
```

这会检查：
- ✓ 数据集是否正确加载
- ✓ SMPL-X模型是否可用
- ✓ 模型是否正确初始化
- ✓ 前向传播是否正常工作

### 2. 开始训练

```bash
# 使用默认配置
python train.py

# 使用自定义配置
python train.py --config configs/custom.yaml

# 从checkpoint恢复训练
python train.py --resume checkpoints/checkpoint_epoch_10.pth
```

### 3. 监控训练

训练过程中会：
- 每10个batch打印一次loss
- 每个epoch结束后在验证集上评估
- 保存checkpoint到 `checkpoints/` 目录
- 保存可视化结果到 `visualizations/` 目录
- 保存训练曲线到 `checkpoints/training_curves.png`

### 4. 查看结果

训练完成后：

```
checkpoints/
├── best_model.pth                    # 最佳模型
├── checkpoint_epoch_5.pth            # 定期checkpoint
├── checkpoint_epoch_10.pth
└── training_curves.png               # 训练曲线

visualizations/
├── epoch_0/
│   ├── category1_id1_projection.png  # 投影可视化
│   └── category1_id1_heatmap.png     # 接触热力图
└── epoch_5/
    └── ...
```

## 常见问题

### 1. CUDA Out of Memory

降低batch_size：

```yaml
training:
  batch_size: 4  # 或更小
```

### 2. 数据集加载失败

检查：
- 数据集路径是否正确
- 所有必需文件是否存在
- JSON文件格式是否正确

### 3. SMPL-X模型加载失败

确保：
- 模型文件路径正确
- 文件名为 `SMPLX_NEUTRAL.pkl` （不是.npz）
- smplx库已正确安装

### 4. 训练不收敛

调整超参数：
- 增加/减少 `pos_weight`（如果recall低则增加，precision低则减少）
- 调整学习率
- 增加训练epoch数

### 5. 投影点不在图像内

检查：
- 相机内参K是否正确
- 顶点坐标系是否为相机坐标系
- 图像resize后内参是否同步缩放

## 训练技巧

### 1. 调试模式

先用小数据集测试：

```python
# 在dataset.py中临时限制样本数
if len(all_samples) > 10:
    all_samples = all_samples[:10]
```

### 2. 过拟合测试

用单个样本训练100个epoch，loss应该降到接近0：

```python
# 在train.py中
train_dataset = Subset(train_dataset, [0])
```

### 3. 可视化调试

启用可视化并检查投影点位置：

```yaml
visualization:
  enabled: true
  num_samples: 4
```

### 4. 学习率调整

如果loss震荡，降低学习率：

```yaml
training:
  learning_rate: 5e-4  # 或更小
```

## 性能优化

### 1. 数据加载加速

```yaml
data:
  num_workers: 4  # 增加worker数量
```

### 2. 混合精度训练

在train.py中添加：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(...)
    loss = criterion(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 预计算法向量

如果训练时计算法向量太慢，可以预处理：

```bash
python precompute_normals.py  # 需要自行实现
```

## 评估指标

训练时输出的指标：
- **Loss**: BCE loss值
- **Precision**: 预测为接触的点中，真正接触的比例
- **Recall**: 真正接触的点中，被正确预测的比例
- **F1**: Precision和Recall的调和平均
- **Accuracy**: 整体准确率

## 模型使用

加载训练好的模型进行推理：

```python
import torch
import yaml
from models.contact_net import ContactNet

# 加载配置
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载模型
device = torch.device('cuda')
model = ContactNet(config).to(device)

# 加载权重
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    logits = model(images, vertices, normals, pose_params, K, bbox)
    probs = torch.sigmoid(logits)  # 转换为概率
    
    # 二值化
    contact_pred = (probs > 0.5).float()
```

## 引用

如果您在研究中使用了本代码，请引用相关论文。

## 联系方式

如有问题，请提交Issue或联系开发者。
