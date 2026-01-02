# SMPL-X Contact Prediction 详细训练框架设计文档 (v2.2)

## **1. 任务背景与目标 (Overview)**

* **核心任务**：给定单张 RGB 图像、对应的 SMPL-X 网格信息、物体 Bounding Box（以及物体 Mask 派生的距离场），预测人体网格上每个顶点的**接触概率**。  
* **关键挑战**：数据量少（~2500样本）、存在遮挡、人物可能未完整出现在图像中（截断）。  
* **工程目标**：搭建一个基于 PyTorch 的模块化训练框架，确保代码可读性、可扩展性和实验的可复现性。

## **2. 详细方法论 (Detailed Methodology)**

本方案采用 **Pixel-aligned Implicit-style Framework**，通过融合视觉特征、几何特征和先验知识来预测接触。

### **2.1 输入层 (Inputs)**

模型接收一个 Batch 的数据，包含以下核心张量：

1. **图像 (Images)**: $B \times 3 \times 512 \times 512$。经过归一化处理的 RGB 图像。  
2. **SMPL-X 顶点 (Vertices)**: $B \times N_{verts} \times 3$。世界坐标系或相机坐标系下的顶点坐标。  
3. **SMPL-X 姿态参数 (Pose Params)**: $B \times 63$。Axis-angle 格式的姿态参数。  
4. **相机内参 (K)**: $B \times 3 \times 3$。用于将 3D 顶点投影到 2D 图像平面。  
5. **物体包围盒 (Object BBox)**: $B \times 4$。格式为 $[x_{min}, y_{min}, x_{max}, y_{max}]$。  
6. **物体 Mask 距离场 (Mask Distance Field)**: $B \times 1 \times 512 \times 512$。由 `object_mask.png` 经过**膨胀**与**距离变换**得到，归一化到 $[0, 1]$（0 表示接近/在物体 mask 上，1 表示远离物体）。
7. **顶点编号 (Vertex IDs, NEW)**: $B \times N_{verts}$（int64）。每个顶点在 SMPL-X 模板网格上的固定索引（通常为 $0 \ldots 10474$）。该输入可以由 Dataset 显式提供，也可以在模型内部根据 $N_{verts}$ 自动生成（本工程实现为自动生成）。

### **2.2 核心处理流程 (Processing Pipeline)**

处理流程分为三个并行分支，最后进行融合。其中物体信息不再仅依赖粗糙 BBox，而是引入基于 Mask 的距离场作为强几何约束。

#### **分支 A: 视觉特征提取 (Visual Feature Extraction)**

* **Backbone**: ResNet18 (预训练)。**完全冻结 (Frozen)**，不参与梯度更新。  
* **特征层**: 提取 Layer 2 (Stride 8) 和 Layer 3 (Stride 16) 的特征图。  
* **投影 (Projection)**: 利用相机内参 $K$ 将 3D 顶点投影到 2D 像素坐标 $(u, v)$。  
* **采样 (Sampling)**:  
  * 使用 grid_sample 在特征图上进行双线性插值。  
  * **关键处理 (Corner Case)**: 设置 padding_mode='zeros'。当投影点超出图像边界时，强制采样值为 0。这避免了边缘伪影，并为模型提供了“此处无视觉信息”的明确信号。  
* **输出**: 每个顶点的视觉特征向量 $F_{vis}$ (约 384 维)。

#### **分支 B: 几何与状态特征 (Geometry & Context)**

* **几何特征**:  
  * **归一化坐标**: $F_{xyz}$ (3维)。将顶点坐标归一化到 $[-1, 1]$ 范围，提供相对空间位置。  
  * **法向**: $F_{normal}$ (3维)。顶点法向量，指示表面朝向。  
* **数据有效性标志 (Corner Case Handling)**:  
  * **Is Inside Image Flag**: $F_{in\_img}$ (1维)。  
  * **定义**: 计算投影点 $(u,v)$ 是否在图像范围内 ($0 \le u < W, 0 \le v < H$) 且深度 $z > 0$。  
  * **区分点**: 这是一个**技术性标志**。如果为 0，明确告知 MLP 忽略全 0 的视觉特征（Padding 造成的），转而完全依赖几何和姿态先验。不要将其与 BBox 特征合并。  
* **物体上下文 (Semantic Context)**:  
  * **Inside BBox Flag**: $F_{in\_box}$ (1维)。投影点是否在物体 BBox 内。这是**逻辑性标志**，表示接触的可能性极高。  
  * **BBox Center Distance (Legacy)**: $F_{dist\_bbox}$ (1维)。投影点距离物体 BBox 中心的归一化距离（粗糙上下文，保留用于兼容/消融）。  
  * **Mask Distance Field (NEW)**: $F_{dist\_mask}$ (1维)。在“膨胀距离场”上按投影坐标采样得到的距离特征，用于提供更精细的物体形状约束，缓解 Over-prediction。

#### **分支 B-1: 膨胀距离场构建 (Dilated Distance Field, NEW)**

该分支在 **Dataset 预处理阶段**完成（图像空间），解决 `object_mask.png` 因遮挡造成的断裂：

1. **读取 Mask**: `object_mask.png`（灰度/二值），Resize 到训练分辨率（512×512，nearest）。  
2. **形态学膨胀 (Dilation)**: 使用椭圆核对 mask 做膨胀以填补遮挡空洞。  
3. **距离变换 (Distance Transform)**: 计算每个像素到“膨胀后物体区域”的最近距离。  
4. **归一化**: 将距离除以图像对角线长度，得到 $[0, 1]$ 的距离场。  

> **Corner Case**: grid_sample 对画外点默认 padding=0，会误导为“很近”。因此在模型里会用 `is_inside_img` 将画外点的 $F_{dist\_mask}$ 强制设为 1（很远）。

#### **分支 C: 全局姿态先验 (Global Pose Prior)**

* **Pose Embedding**: 将 63 维的 Pose 参数输入一个小型的 MLP (Linear -> ReLU)，映射为 32 维的 Embedding $F_{pose}$。  
* **广播**: 将这 32 维特征复制并拼接到每个顶点上。

#### **分支 D: 网格拓扑先验 (Mesh Topology Prior, NEW)**

* **Vertex ID Embedding**: 将每个顶点的整数编号 $id \in [0, N_{verts})$ 输入一个可学习的 `nn.Embedding`，得到 $F_{vid}$。  
* **直觉**: 顶点编号对应 SMPL-X 模板上的固定语义位置（例如足底、膝盖附近等）。该 embedding 为模型提供“这是身体的哪个点”的强先验，有助于抑制不合理的过分割预测。

### **2.3 特征融合与输出 (Fusion & Output)**

* 特征拼接:  

  $$F_{final} = [F_{vis} \oplus F_{xyz} \oplus F_{normal} \oplus F_{in\_img} \oplus F_{in\_box} \oplus F_{dist\_bbox} \oplus F_{dist\_mask} \oplus F_{pose} \oplus F_{vid}]$$  

  * 总维度约为：$384 + 3 + 3 + 1 + 1 + 1 + 1 + 32 + D_{vid}$ 维。  
  * 在默认配置下：$D_{vid}=16$，因此总维度为 **442**。  

* **分类器 (Head)**: Shared MLP (Point-wise)。  
  
  * 结构：`Linear(442, 256) -> BN -> ReLU -> Dropout -> Linear(256, 128) -> BN -> ReLU -> Dropout -> Linear(128, 1)`（默认配置）。  
  
* **最终输出**: Sigmoid 激活后的标量，表示接触概率 $P \in [0, 1]$。

## **3. 代码仓库结构设计 (Code Structure)**

```
smplx_contact_prediction/  
├── configs/                    # 配置管理  
│   └── default.yaml            # 所有的超参数  
├── data/                       # 数据管道  
│   ├── __init__.py  
│   └── dataset.py              # SmplContactDataset  
├── models/                     # 模型组件  
│   ├── __init__.py  
│   ├── backbone.py             # ResNet 提取器 (Frozen)  
│   ├── geometry.py             # ★核心：投影、采样、几何计算  
│   └── contact_net.py          # 模型组装  
├── utils/                      # 工具箱  
│   ├── geometry_utils.py       # 坐标变换工具  
│   └── visualization.py        # 调试可视化  
├── train.py                    # 训练脚本  
└── requirements.txt
```

## **4. 模块详细开发说明 (Implementation Details)**

### **4.1 models/geometry.py (最核心模块)**

这是实现 Pixel-aligned 和 Corner Case 处理的关键。

* **类定义**: class GeometryProcessor(nn.Module):  
* **Forward 参数**: vertices (B, N, 3), normals (B, N, 3), K (B, 3, 3), img_size (H, W), object_bbox (B, 4)  
* **核心逻辑**:  
  1. **3D -> 2D 投影**:  
     * $P_{cam} = K \times P_{world}$  
     * $u = P_{cam}[x] / P_{cam}[z], v = P_{cam}[y] / P_{cam}[z]$  
  2. **图像范围掩码 (Is Inside Image)**:  
     * mask_x = (u >= 0) & (u < W)  
     * mask_y = (v >= 0) & (v < H)  
     * mask_z = (P_{cam}[z] > 0)  
     * is_inside_img = (mask_x & mask_y & mask_z).float().unsqueeze(-1)
  3. **归一化采样坐标**:  
     * 将 $(u, v)$ 从 $[0, W]$ 映射到 $[-1, 1]$ 用于 grid_sample。  
  4. **物体上下文 (BBox)**:  
     * 根据 object_bbox 计算 is_inside_box 和 dist_to_center（bbox center distance）。  
  5. **返回值**: grid_coords (用于采样), geom_feats（包含 is_inside_img, in_box, dist_to_center, normals 等）。

### **4.2 models/backbone.py (Frozen)**

* **类定义**: class FeatureExtractor(nn.Module):  
* **初始化逻辑 (__init__)**:  
  * 加载 resnet18(pretrained=True)。  
  * 去掉 avgpool 和 fc。  
  * **冻结权重**: 遍历所有参数，设置 param.requires_grad = False。  
  * **BN 层处理**: 建议将 BN 层设置为 .eval() 模式，固定统计量（Running Mean/Var），防止小 Batch Size 导致 BN 统计抖动。  
* **Forward**: 返回列表 [layer2_out, layer3_out]。

### **4.3 models/contact_net.py**

* **类定义**: class ContactNet(nn.Module):  
* **逻辑**:  
  1. 调用 backbone 得到特征图 (无梯度)。  
  2. 调用 geometry_processor 得到 grid_coords 和 geom_feats。  
  3. **Visual Sampling**:  
     * F.grid_sample(feat_map, grid_coords, padding_mode='zeros') **(注意：zeros padding)**。  
  4. **Mask Distance Sampling (NEW)**:  
     * 对 `mask_dist_field` 做 grid_sample 得到每顶点 $F_{dist\_mask}$。  
     * 用 `is_inside_img` 将画外点的 $F_{dist\_mask}$ 设为 1（远离物体），避免 padding=0 造成的伪近距离。  
  5. **Pose Embedding**: self.pose_mlp(pose_params)。  
  6. **Vertex ID Embedding (NEW)**: `nn.Embedding(num_verts, D_vid)` 生成 $F_{vid}$ 并拼接。  
  7. torch.cat 所有特征（默认总维度 442）。  
  8. 传入 MLP 得到输出。

### **4.4 data/dataset.py (含数据集格式说明)**

* **注意**: 确保加载的 Vertices 和 Labels 是一一对应的。  
* **数据增强**: 即使只有 2500 样本，建议在 Dataset 中加入简单的图像增强 (ColorJitter) 和几何增强 (微小的 Vertices 偏移)，以提高鲁棒性。

#### **4.4.1 数据集文件映射 (Dataset Specification)**

根据您提供的文件列表，我们在 __getitem__ 中仅需加载以下**核心文件**。请忽略 *.obj (除非用于调试)、depth.npy 等非必要文件。注意：`object_mask.png` 现在是**必需输入**（缺失将直接报错）。

| 原始文件名            | 对应模型输入          | 用途与处理说明                                               |
| :-------------------- | :-------------------- | :----------------------------------------------------------- |
| image.jpg             | **Images**            | 读取并 Resize 到 512x512，归一化。                           |
| object_mask.png       | **Mask Distance Field** | 读取并 Resize（nearest），对 mask 做膨胀 + 距离变换，输出 `mask_dist_field`（1×H×W，归一化到[0,1]）。 |
| smplx_parameters.json | **Pose Params**       | 读取 body_pose (63 dims) 用于 Pose Embedding。 读取 transl, global_orient, betas 等配合 smplx Layer 生成 **Vertices**。 |
| contact.json          | **GT Labels**         | 读取 Contact Labels (0/1)。需与 Vertices 顺序严格对齐。      |
| box_annotation.json   | **Object BBox**       | 读取物体 2D 包围盒 $[x_1, y_1, x_2, y_2]$。                  |
| calibration.json      | **Intrinsics (K)**    | 读取相机内参矩阵 $3 \times 3$。                              |
| extrinsic.json        | **Extrinsics (R, T)** | (必要时) 用于将 Vertices 从 World Space 转换到 Camera Space。建议在 Dataset 中完成此转换，模型只接收 Cam Space 坐标。 |
| normals_smplx.npy     | **Normals**           | (可选优化) 直接读取预计算好的法向，避免实时计算。作为几何特征输入。 |

**建议**: 虽然 h_mesh.obj 包含顶点，但推荐使用 smplx_parameters.json 动态生成或加载，以确保 Pose 参数与顶点形状的数学对应关系绝对一致。

### **4.5 configs/default.yaml**

```
model:
  visual_feat_dim: 384
  num_verts: 10475
  geometry_feat_dim: 3
  normal_feat_dim: 3
  flag_feat_dim: 4      # in_img + in_box + dist_to_center + mask_dist
  pose_embed_dim: 32
  vertex_id_embed_dim: 16
  total_feat_dim: 442
```

  

## **5. 核心注意事项与易错点 (Precautions & Best Practices)**

在编写代码和训练过程中，请务必关注以下几点：

### **5.1 数据与对齐 (Crucial)**

* **坐标系一致性**: 最常见的 Bug 来源。确保 Dataset 中输出的 vertices 和 object_bbox、cam_intrinsics 都在**同一个坐标系**（通常建议统一转换到 **Camera Space**）。如果一个在 World 一个在 Camera，投影会完全错误。  
* **图像尺寸**: grid_sample 归一化坐标是 $[-1, 1]$。如果你在 Dataset 里 Resize 了图像（比如到 512x512），请确保内参 $K$ 也进行了相应的缩放（Scale $f_x, f_y, c_x, c_y$）。
### **5.2 SMPL-X 模型构建与环境依赖 (Critical)**

为了保证输入模型的顶点索引 ($0 \dots 10474$) 与真值标签 (contact.json) 绝对对齐，**严禁直接读取** h_mesh.obj 或其他导出的 Mesh 文件。必须使用官方的 parametric layer 动态生成网格。

* **依赖库要求**:  
  * pip install smplx (官方库，用于从参数生成 Mesh)  
  * pip install opencv-python (必需：用于 Mask 膨胀与距离变换 distanceTransform)  
  * pip install trimesh (可选，用于法向计算等辅助操作)  
* **模型文件依赖 (Model Assets)**:  
  * 必须前往 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/) 注册并下载模型文件 (SMPLX_NEUTRAL.pkl, SMPLX_MALE.pkl, 等)。  
  * **配置**: 在 configs/default.yaml 中添加 smplx_model_path，指向这些模型文件的存放目录。  
* **对齐原理**:  
  * 只有通过 smplx 库加载 smplx_parameters.json 中的 betas, body_pose, global_orient, transl，才能保证生成的 Mesh 拓扑结构是标准的。  
  * 如果读取 obj 文件，某些导出软件（如 Blender, MeshLab）可能会隐式地重排顶点顺序（Vertex Reordering），导致 Contact Label $y_i$ 对应到了错误的顶点 $v_j$，使训练完全失效。
  
### **5.3 算法特性与限制**

* **Visual Features 的脆弱性**: 在人物身体边缘，由于投影误差，很容易采样到背景像素。这就是为什么**Label Dilation（标签膨胀）**在预处理阶段极其重要——它容忍了这种边缘误差。  
* **Zero Padding 的意义**: 务必检查 grid_sample 的 padding_mode='zeros'。如果使用默认的 border，当人有一半身体在画外时，画外的点会重复采样边缘像素，导致严重的误判。
* **Mask 距离场的画外处理 (NEW)**: 距离场语义是“0=近，1=远”。因此必须避免画外点被 padding=0 伪装成“近距离”。本实现用 `is_inside_img` 将画外点的 $F_{dist\_mask}$ 强制设为 1。

### **5.4 训练稳定性**

* **冻结策略 (Frozen Backbone)**:  
  * 由于 Backbone 已在模型内部冻结，**优化器 (Optimizer) 初始化时** 务必注意：  
  * 不要直接用 model.parameters()（虽然 requires_grad=False 的参数通常会被优化器自动忽略，但显式过滤更安全）。  
  * 推荐写法：optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)。  
* **Pos_weight 的调节**: 初始设定建议为 5.0。  
  * 如果发现 Recall 很高但 Precision 极低（满屏都是红点），适当降低权重（如 3.0）。  
  * 如果发现 Recall 很低（几乎没有预测出接触），适当增加权重（如 8.0）。

### **5.5 调试技巧**

* **第一步永远是可视化**: 不要直接看 Loss。在 train.py 里写一个 helper function，把投影点 $(u, v)$ 画在 Input Image 上。  
  * 如果点跑到了人外面，说明 $K$ 或坐标系错了。  
  * 如果点是对的，但模型不收敛，再检查 Loss。  
* **过拟合测试**: 在写完代码后，用**一张图片（Batch Size=1）**重复训练 100 次。  
  * Loss 应该降到 0。  
  * 预测的热力图应该和 GT 完全一致。  
  * 如果做不到这点，说明代码逻辑有 Bug（通常是 geometry.py 里的维度拼接或采样问题）。

