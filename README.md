# 课堂教学行为分析系统

基于 `Tkinter + OpenCV + YOLOv8 + FaceNet` 的本地桌面端课堂分析工具。

当前版本做了几处收敛：
- 启动时只加载姿态模型。
- 桌面物品模型、人脸模型、行为模型按需加载。
- 图片检测入口已移除，避免污染课堂会话统计。
- 录制改为显式开关，默认关闭。

## 目录说明

- `app.py`：主程序入口
- `classroom_constants.py`：行为标签映射
- `classroom_face_store.py`：人脸库读写与管理
- `classroom_metrics.py`：课堂与学生指标计算
- `classroom_recording.py`：实时录制状态与视频写入
- `classroom_rendering.py`：中文绘字
- `classroom_report_analysis.py`：报告文本分析结论生成
- `classroom_reporting.py`：报告生成
- `models/best.pt`：行为检测模型
- `models/yolov8n-pose.pt`：姿态模型
- `models/yolov8n.pt`：桌面物品模型
- `face_database.pkl`：人脸库
- `attention_logs/`：报告输出目录
- `realtime_videos/`：录制输出目录

## 运行环境

- Windows
  原因：代码使用了 `winsound` 做声音提醒。
- Python 图形环境
  `tkinter` 一般随 Windows 版 Python 提供；如果缺失，需要重新安装带 Tcl/Tk 的 Python。
- 与 `torch`、`torchvision`、`facenet-pytorch` 兼容的 Python 环境
  如果直接安装失败，先按 PyTorch 官方方式安装适配你机器的 `torch` / `torchvision`，再执行依赖安装。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

## 使用说明

- 默认打开摄像头，启动阶段只加载姿态模型。
- 勾选“桌面物品检测”后，首次使用时才加载 `models/yolov8n.pt`。
- 勾选“人脸识别”后，首次使用时才加载 `MTCNN + InceptionResnetV1`。
- 摄像头模式下，如果 `models/best.pt` 存在，行为模型默认启用，并在首次处理帧时按需加载。
- 本地视频模式只做抬头/低头分析，不启用桌面物品检测、人脸识别和行为模型。
- “开始录制”保存的是带检测叠加的画面，不是原始视频。

## 输出文件

- 报告文本：`attention_logs/report_YYYYMMDD_HHMMSS/classroom_report.txt`
- 图表目录：`attention_logs/report_YYYYMMDD_HHMMSS/`
- 录制视频：`realtime_videos/`
- 设置文件：`settings/config.txt`

## 模型文件要求

运行前请确认这些文件存在：

- `models/yolov8n-pose.pt`
- `models/yolov8n.pt`
- `models/best.pt`

如果行为模型 `models/best.pt` 不存在，系统会回退到规则判定，不会阻止程序启动。

## 已知边界

- 人物稳定 ID 主要依赖框重叠，遮挡或交叉走动时可能串号。
- 单人结果更适合参考，不建议作为唯一评价依据。
- 光照、角度、遮挡会直接影响识别效果。
