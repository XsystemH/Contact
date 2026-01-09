# Web-Based Multi-User Data Convert Tool

## 概述

这是一个基于 Web 浏览器的多用户协同数据处理系统，用于批量转换和标注数据集。相比传统的 PIL/matplotlib 界面，Web 版本提供了：

- 🌐 **浏览器访问**：在任何设备的浏览器中预览和操作
- 👥 **多用户协同**：支持局域网内多人同时处理不同数据
- 🚀 **更高性能**：使用 Plotly 实现流畅的 3D 交互式可视化
- 📊 **实时统计**：所有用户的进度实时同步
- 💾 **自动保存**：进度自动保存，支持断点续传

## 安装依赖

在 Contact 仓库内建议用额外依赖文件：

```bash
pip install -r requirements_web_convert.txt
```

## 使用方法

### 1. 启动 Web 服务器

在 Contact 仓库中启动：

```bash
cd Contact/web_convert
python web_interface.py \
   --source_dir /path/to/source \
   --target_dir /path/to/target \
   --object_dir /path/to/data_object
```

默认配置：
- Host: `0.0.0.0` (允许局域网访问)
- Port: `5000`
- SMPL-X 模型路径：默认使用 `Contact/smplx_models`

### 2. 自定义配置

```bash
python web_interface.py \
   --host 0.0.0.0 \
   --port 8080 \
   --source_dir /path/to/source \
   --target_dir /path/to/target \
   --object_dir /path/to/data_object \
   --category bottle \
   --random_order
```

### 3. 访问界面

启动后，服务器会显示访问地址，例如：
```
Server running at: http://0.0.0.0:5000
```

在浏览器中访问：
- 本机：`http://localhost:5000`
- 局域网内其他设备：`http://服务器IP:5000`

### 4. 多用户协同

1. 将服务器地址分享给团队成员
2. 每个成员在浏览器中打开该地址
3. 点击 "Start Working" 开始工作
4. 系统自动分配不同的任务给不同用户
5. 每个用户独立处理分配到的任务

## 界面功能

### 主页面 (/)
- 显示总体统计信息
- 查看当前配置
- 显示活跃用户数量

### 任务查看器 (/viewer)

#### 左侧控制面板
- **任务信息**：显示当前任务的类别、路径和 ID
- **Distance Ratio 调节**：调整接触检测的距离阈值（0.01-3.0）
- **Update Preview**：使用新的参数重新生成可视化
- **Accept & Process**：接受当前任务并进行处理
- **Skip This Task**：跳过当前任务
- **Get Next Task**：获取下一个待处理任务

#### 右侧可视化区域
- **Human Mesh**：人体网格和接触区域
  - 蓝色：非接触点
  - 橙色：仅通过内部检测识别的接触
  - 黄色：仅通过距离检测识别的接触
  - 红色：两种方法都检测到的接触
  
- **Object Mesh**：物体网格和接触区域
  - 绿色：非接触点
  - 颜色编码同人体网格

- **Combined View**：人体和物体的组合视图
  
- **Reference Image**：原始参考图像

#### 统计信息
- Human Total：人体接触点总数
- Object Total：物体接触点总数
- Total Contact：总接触点数
- H Interior：人体内部检测的接触点
- H Proximity：人体距离检测的接触点
- H Both：两种方法都检测到的接触点

## 工作流程

1. **启动服务器**：在服务器机器上运行 `python web_interface.py --source_dir ... --target_dir ...`

2. **连接客户端**：团队成员在浏览器中访问服务器地址

3. **自动任务分配**：
   - 点击 "Start Working" 后自动获取第一个任务
   - 每个用户处理不同的任务，避免重复工作

4. **调整参数**：
   - 调节 Distance Ratio 参数
   - 点击 "Update Preview" 查看效果
   - 在 3D 视图中交互式旋转、缩放查看

5. **做出决策**：
   - **Accept**：处理当前数据集（复制文件、生成 contact.json 等）
   - **Skip**：跳过当前数据集（记录在 give_up.json 中）

6. **自动保存**：
   - 每个决策后自动保存进度
   - 支持断点续传，重启后继续未完成的任务

## 数据处理流程

当用户点击 "Accept & Process" 时，系统会：

1. 复制源数据到目标目录
2. 重命名 `obj_mask.png` 为 `object_mask.png`
3. 复制 `obj_pcd_h_align.obj` 文件
4. 生成 `calibration.json` / `extrinsic.json`
5. 根据用户设定的 distance_ratio 生成 `contact.json`

## 进度管理

### 自动保存
- 进度保存在 `{target_dir}/progress.json`
- 包含每个任务的处理状态、参数和错误信息

### 跳过记录
- 跳过的任务记录在 `{target_dir}/give_up.json`
- 包含跳过原因和尝试的参数

### 断点续传
- 重启服务器后自动加载之前的进度
- 只处理未完成的任务

## 技术特性

- **Flask + Socket.IO**：实时双向通信
- **Plotly.js**：高性能交互式 3D 可视化
- **线程安全**：多用户并发处理的任务分配
- **会话管理**：为每个用户维护独立的会话状态
- **实时同步**：所有用户的统计信息实时更新

## 与传统模式对比

| 特性 | 传统模式 (matplotlib) | Web 模式 (Plotly) |
|------|---------------------|------------------|
| 界面 | 桌面 GUI | 浏览器 |
| 性能 | 较慢，卡顿 | 流畅 |
| 交互 | 有限 | 完全交互式 3D |
| 多用户 | ❌ | ✅ |
| 远程访问 | ❌ | ✅ (局域网) |
| 移动设备 | ❌ | ✅ |
| 进度同步 | ❌ | ✅ 实时 |

## 故障排除

### 端口被占用
```bash
# 使用其他端口
python data_convert.py --web-mode --port 8080
```

### 无法从其他设备访问
1. 确保防火墙允许该端口
2. 检查服务器 IP 地址：`hostname -I` 或 `ip addr`
3. 确保使用 `--host 0.0.0.0` 而不是 `localhost`

### 3D 可视化加载缓慢
- 这是正常现象，首次加载需要计算网格数据
- 后续更新会更快

## 传统模式

如果仍需使用传统的桌面 GUI 模式：

```bash
# 交互式模式（matplotlib）
python data_convert.py

# 自动模式（无交互）
python data_convert.py --auto-scale
```

## 参考

- Flask 文档: https://flask.palletsprojects.com/
- Socket.IO 文档: https://socket.io/docs/
- Plotly.js 文档: https://plotly.com/javascript/

## 许可证

与主项目保持一致
