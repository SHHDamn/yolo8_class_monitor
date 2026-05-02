import atexit
import os
import pickle
import threading
import time
from collections import defaultdict
from threading import Lock

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import torch
import winsound
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, simpledialog
from ultralytics import YOLO

from classroom_constants import BEHAVIOR_CLASS_ZH, normalize_behavior_class_name
from classroom_rendering import draw_chinese_text, draw_chinese_texts
from classroom_reporting import generate_new_classroom_report

matplotlib.use("Agg")


class ClassroomMonitor:
    def __init__(self,
                 model_path="models/yolov8n-pose.pt",
                 video_source=0,
                 head_down_threshold=18,
                 time_threshold=2.0,
                 head_turn_threshold=35,
                 confidence_threshold=0.45,
                 total_students=30,
                 # behavior_model_path="models/behavior_student_v2_3cls.pt"):
                 behavior_model_path="models/best.pt"):

        # 加载姿态估计模型
        self.model = YOLO(model_path)
        self.object_model_path = "models/yolov8n.pt"
        self.object_model = None
        self.behavior_model_path = behavior_model_path
        self._behavior_zh_by_norm_key = dict(BEHAVIOR_CLASS_ZH)
        self.behavior_model = None
        self.behavior_model_enabled = os.path.exists(behavior_model_path)
        if self.behavior_model_enabled:
            print(f"行为检测模型待按需加载: {behavior_model_path}")
        else:
            print(f"未找到行为检测模型，继续使用规则判定: {behavior_model_path}")
        
        self.attention_logs = defaultdict(list)
        self.cap = cv2.VideoCapture(video_source)
        self.video_source = video_source
        self.head_down_threshold = head_down_threshold
        self.time_threshold = time_threshold
        self.head_turn_threshold = head_turn_threshold
        self.confidence_threshold = confidence_threshold
        self.total_students = total_students
        
        self.student_states = defaultdict(lambda: {
            "head_down": False,
            "head_down_start_time": 0,
            "warning_issued": False,
            "last_position": None,
            "position_history": [],
            "student_number": None,
            "identity": "未知",
            "last_behavior_zh": "",
            "last_behavior_conf": None,
        })
        
        self.attention_color = (0, 255, 0)  # 绿色：专注
        self.warning_color = (0, 0, 255)    # 红色：警告
        self.object_color = (0, 165, 255)   # 橙色：物品
        self.identity_color = (0, 255, 255)  # 黄色：身份信息
        
        self.debug = True
        self.beep_enabled = True
        self.keypoint_threshold = max(0.2, self.confidence_threshold)
        
        self.latest_class_metrics = {
            "students_analyzed": 0,
            "focus_rate": 0.0,
            "head_up_rate": 0.0,
            "dominant_habit": "insufficient_data"
        }
        
        # 骨架连接定义
        self.skeleton_edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
        ]

        # 滤波器参数
        self.filter_size = 3
        self.frame_count = 0
        self.id_mapping = {}
        self.next_id = 0
        self.track_threshold = 0.6

        # 当前检测到的人数
        self.current_count = 0

        # 录制相关变量（保存叠加检测结果后的画面）
        self.realtime_save_enabled = False
        self.realtime_video_writer = None
        self.realtime_video_path = None
        self.realtime_video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.realtime_video_fps = 0
        
        # 帧率控制相关属性
        self.actual_fps = 0
        self.recording_fps = 30
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self._frame_processing_times = []
        self.target_fps = 30
        self._sync_target_fps_with_source()
        
        # 桌面物品检测相关
        self.detected_objects = []
        self.object_detection_enabled = False
        # COCO 类别索引: cell phone, book, laptop, keyboard, mouse, cup
        self.desk_object_classes = [67, 73, 63, 66, 64, 41]
        self.local_video_object_stride = 6
        self._cached_detected_objects = []
        self.local_video_behavior_stride = 6
        self.behavior_model_conf = 0.35
        self.behavior_model_imgsz = 512
        self._cached_behavior_detections = []

        # 会话级行为统计（用于退出 / 手动报告中的柱状图、饼图）
        self.session_behavior_samples = []
        self._finalize_report_lock = Lock()
        self._finalize_report_done = False

        # ========== 人脸识别相关（深度学习版） ==========
        self.face_recognition_enabled = False  # 人脸识别开关
        self.known_face_encodings = []  # 已知人脸编码（512维深度特征向量）
        self.known_face_names = []  # 已知人脸姓名
        self.face_database_path = "face_database.pkl"  # 人脸数据库路径
        self.face_tolerance = 0.65  # 人脸识别容差（余弦相似度阈值，越高越严格）
        self.local_video_face_stride = 12
        self._cached_face_results = ([], [], [])
        self.local_video_pose_imgsz = 640
        self.local_video_pose_conf = 0.2
        self._cached_pose_result = None

        # 本地视频“仅抬头/低头”模式：由 GUI 在打开本地视频时开启
        # 开启后会跳过行为模型/桌面物品/人脸，只保留姿态关键点计算抬头/低头
        self.simple_pose_only = False
        
        # 选择运算设备（有GPU用GPU，没有就用CPU）
        self.face_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mtcnn = None
        self.facenet = None
        
        # 加载人脸数据库
        self.load_face_database()

    def ensure_object_model_loaded(self):
        if self.object_model is not None:
            return True, None
        if not os.path.exists(self.object_model_path):
            return False, f"未找到桌面物品模型: {self.object_model_path}"

        try:
            self.object_model = YOLO(self.object_model_path)
            print(f"桌面物品模型已加载: {self.object_model_path}")
            return True, None
        except Exception as e:
            return False, f"桌面物品模型加载失败: {e}"

    def ensure_behavior_model_loaded(self):
        if self.behavior_model is not None:
            return True, None
        if not self.behavior_model_enabled:
            return False, "行为检测模型未启用"
        if not os.path.exists(self.behavior_model_path):
            self.behavior_model_enabled = False
            return False, f"未找到行为检测模型: {self.behavior_model_path}"

        try:
            self.behavior_model = YOLO(self.behavior_model_path)
            self._merge_model_behavior_names()
            print(f"行为检测模型已加载: {self.behavior_model_path}")
            return True, None
        except Exception as e:
            self.behavior_model_enabled = False
            return False, f"行为检测模型加载失败: {e}"

    def ensure_face_models_loaded(self):
        if self.mtcnn is not None and self.facenet is not None:
            return True, None

        try:
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.face_device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=True,
            )
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.face_device)
            print(f"人脸识别深度学习模型已加载，运算设备: {self.face_device}")
            return True, None
        except Exception as e:
            self.mtcnn = None
            self.facenet = None
            return False, f"人脸识别模型加载失败: {e}"

    def stop_realtime_recording(self):
        final_path = self.realtime_video_path
        if self.realtime_video_writer is not None:
            self.realtime_video_writer.release()
            self.realtime_video_writer = None
        self.realtime_video_path = None
        self.realtime_video_fps = 0
        return final_path

    def set_recording_enabled(self, enabled):
        self.realtime_save_enabled = bool(enabled)
        if not self.realtime_save_enabled:
            return self.stop_realtime_recording()
        return self.realtime_video_path

    # ========== 人脸识别相关方法 ==========
    
    def load_face_database(self):
        if os.path.exists(self.face_database_path):
            try:
                with open(self.face_database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"已加载人脸数据库：{len(self.known_face_names)} 人")
            except Exception as e:
                print(f"加载人脸数据库失败：{e}")
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            print("人脸数据库不存在，将创建新数据库")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_face_database(self):
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.face_database_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"人脸数据库已保存：{len(self.known_face_names)} 人")
            return True
        except Exception as e:
            print(f"保存人脸数据库失败：{e}")
            return False

    def _merge_model_behavior_names(self):
        if not self.behavior_model_enabled or self.behavior_model is None:
            return
        names = getattr(self.behavior_model, "names", None)
        if not isinstance(names, dict):
            return
        for _, raw in names.items():
            key = normalize_behavior_class_name(raw)
            if not key:
                continue
            if key not in self._behavior_zh_by_norm_key:
                self._behavior_zh_by_norm_key[key] = self._fallback_zh_for_unknown_behavior(raw)

    def translate_behavior_class(self, source_name):
        if source_name is None:
            return "未知"
        if isinstance(source_name, (int, float)):
            source_name = str(int(source_name))
        raw = str(source_name).strip()
        if any("\u4e00" <= c <= "\u9fff" for c in raw):
            return raw
        key = normalize_behavior_class_name(raw)
        if key in self._behavior_zh_by_norm_key:
            return self._behavior_zh_by_norm_key[key]
        return self._fallback_zh_for_unknown_behavior(raw)

    def _fallback_zh_for_unknown_behavior(self, raw):
        key = normalize_behavior_class_name(raw)
        if key in self._behavior_zh_by_norm_key:
            return self._behavior_zh_by_norm_key[key]
        spaced = key.replace("_", " ").strip()
        return spaced if spaced else str(raw)

    def record_session_behavior_sample(self, label_zh, confidence, is_focused):
        if not label_zh:
            return
        conf = float(confidence) if confidence is not None else None
        self.session_behavior_samples.append(
            {"zh": label_zh, "conf": conf, "focused": bool(is_focused)}
        )

    def close_finalize_reports(self):
        with self._finalize_report_lock:
            if self._finalize_report_done:
                return
            try:
                # 稳妥落盘：退出时释放实时检测视频写入器
                try:
                    self.stop_realtime_recording()
                except Exception as e:
                    print(f"退出时释放实时视频写入器失败: {e}")

                # 新版报告：退出时也按新版逻辑生成（不再生成旧版 attention_report.txt）
                generate_new_classroom_report(self, save_dir="attention_logs")
            except Exception as e:
                print(f"生成退出检测报告时出错: {e}")
            self._finalize_report_done = True

    def _behavior_label_is_focused(self, label_zh):
        unfocused = {
            "玩手机", "睡觉", "低头", "转头", "托腮走神", "非专注", "教师讲解", "教师辅导",
            "无书本", "靠桌子",
        }
        focused = {
            "阅读", "写作", "举手", "起立作答", "开卷阅读", "合书", "电子书", "课堂作业",
            "专注听讲",
        }
        if label_zh in unfocused:
            return False
        if label_zh in focused:
            return True
        return None

    def plot_session_behavior_dashboard(self, save_dir="attention_logs"):
        """根据会话样本生成行为柱状图、专注度饼图与文本摘要。"""
        os.makedirs(save_dir, exist_ok=True)
        if not self.session_behavior_samples:
            return

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        agg = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "conf_n": 0})
        focus_bins = [0, 0]
        for row in self.session_behavior_samples:
            zh = row["zh"]
            agg[zh]["count"] += 1
            if row["conf"] is not None:
                agg[zh]["conf_sum"] += row["conf"]
                agg[zh]["conf_n"] += 1
            cat = self._behavior_label_is_focused(zh)
            if cat is True:
                focus_bins[0] += 1
            elif cat is False:
                focus_bins[1] += 1
            else:
                focus_bins[1 if not row["focused"] else 0] += 1

        names_sorted = sorted(agg.keys(), key=lambda k: agg[k]["count"], reverse=True)
        counts = [agg[k]["count"] for k in names_sorted]
        total_n = sum(counts) or 1
        mean_probs = []
        for k in names_sorted:
            if agg[k]["conf_n"] > 0:
                mean_probs.append(100.0 * agg[k]["conf_sum"] / agg[k]["conf_n"])
            else:
                mean_probs.append(0.0)

        report_lines = [
            "行为识别检测报告（会话汇总）",
            "====================",
            "",
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"样本总数（人·帧累计）: {total_n}",
            "",
            "识别结果（行为类别）:",
        ]
        report_lines.append("、".join(names_sorted))
        report_lines.append("")
        report_lines.append("预测概率（各类别平均置信度）:")
        for k, p in zip(names_sorted, mean_probs):
            report_lines.append(f"  {k}: {p:.2f}%")
        report_lines.append("")

        txt_path = os.path.join(save_dir, "detection_session_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(names_sorted))
        bars = ax1.bar(x, counts, color="#1976D2")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names_sorted, rotation=25, ha="right")
        ax1.set_ylabel("检测次数")
        ax1.set_title("行为统计柱状图")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        for rect, c in zip(bars, counts):
            pct = 100.0 * c / total_n
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{c} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fsum = focus_bins[0] + focus_bins[1]
        if fsum <= 0:
            ax2.text(0.5, 0.5, "无专注度样本", ha="center", va="center")
            ax2.axis("off")
        else:
            sizes = [focus_bins[0], focus_bins[1]]
            labels_pie = [
                f"专注 ({100.0 * sizes[0] / fsum:.1f}%)",
                f"非专注 ({100.0 * sizes[1] / fsum:.1f}%)",
            ]
            ax2.pie(
                sizes,
                labels=labels_pie,
                colors=["#1976D2", "#FF9800"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"width": 0.35},
            )
            ax2.set_title("专注度分布（饼图）")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "detection_session_overview.png"), dpi=200)
        plt.close()

    def get_face_database_entries(self):
        return [{"index": idx, "name": name} for idx, name in enumerate(self.known_face_names)]

    def delete_face_from_database(self, index):
        if index < 0 or index >= len(self.known_face_names):
            return False, "无效的人脸索引"

        try:
            deleted_name = self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
            if not self.save_face_database():
                return False, "删除后保存人脸数据库失败"
            print(f"已删除人脸：{deleted_name}")
            return True, deleted_name
        except Exception as e:
            return False, f"删除失败：{e}"

    def rename_face_in_database(self, index, new_name):
        normalized_name = new_name.strip()
        if not normalized_name:
            return False, "姓名不能为空"
        if index < 0 or index >= len(self.known_face_names):
            return False, "无效的人脸索引"

        try:
            old_name = self.known_face_names[index]
            self.known_face_names[index] = normalized_name
            if not self.save_face_database():
                self.known_face_names[index] = old_name
                return False, "重命名后保存人脸数据库失败"
            print(f"已重命名人脸：{old_name} -> {normalized_name}")
            return True, old_name
        except Exception as e:
            return False, f"重命名失败：{e}"
    
    def add_face_to_database(self, face_image, name):

        try:
            ok, message = self.ensure_face_models_loaded()
            if not ok:
                print(message)
                return False
            # 将 OpenCV 的 BGR 格式转换为 RGB 格式（深度学习模型要求）
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # 使用 MTCNN 神经网络检测人脸并裁剪对齐
            # MTCNN 会自动完成人脸检测、关键点定位、仿射对齐三个步骤
            face_tensors, probs = self.mtcnn(pil_image, return_prob=True)
            
            if face_tensors is None or len(face_tensors) == 0:
                print("MTCNN 未检测到人脸")
                return False
            
            # 取置信度最高的人脸
            best_idx = probs.argmax() if len(probs) > 1 else 0
            face_tensor = face_tensors[best_idx].unsqueeze(0).to(self.face_device)
            
            # 使用 InceptionResnetV1 (FaceNet) 提取 512 维深度特征向量
            # 该网络经过 VGGFace2 (包含 331万张人脸图像) 数据集预训练
            with torch.no_grad():
                face_encoding = self.facenet(face_tensor).cpu().numpy().flatten()
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.save_face_database()
            print(f"已添加人脸：{name}（特征维度: {len(face_encoding)}维）")
            return True
        except Exception as e:
            print(f"添加人脸失败：{e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recognize_faces(self, frame):
        if not self.face_recognition_enabled:
            return [], [], []
        
        try:
            ok, message = self.ensure_face_models_loaded()
            if not ok:
                print(message)
                return [], [], []
            # 缩放图像以加速 MTCNN 检测（教室场景人脸较小，0.5倍足够）
            scale_factor = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            
            # 使用 MTCNN 深度神经网络检测所有人脸的边界框和置信度
            boxes, probs = self.mtcnn.detect(pil_frame)
            
            face_locations = []
            face_names = []
            face_confidences = []
            
            if boxes is None or len(boxes) == 0:
                return [], [], []
            
            face_encodings = []
            if self.face_recognition_enabled and len(self.known_face_encodings) > 0:
                face_tensors = self.mtcnn(pil_frame)
                if face_tensors is not None:
                    with torch.no_grad():
                        face_encodings = self.facenet(face_tensors.to(self.face_device)).cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                
                name = "未知"
                confidence = 0.0
                
                if self.face_recognition_enabled and i < len(face_encodings) and len(self.known_face_encodings) > 0:
                    encoding = face_encodings[i]
                    # 计算当前人脸特征与所有已知人脸的余弦相似度
                    # 余弦相似度衡量两个向量在高维空间中的方向一致性
                    similarities = []
                    for known_encoding in self.known_face_encodings:
                        similarity = np.dot(encoding, known_encoding) / (
                            np.linalg.norm(encoding) * np.linalg.norm(known_encoding) + 1e-8
                        )
                        similarities.append(similarity)
                    
                    best_match_index = np.argmax(similarities)
                    best_similarity = similarities[best_match_index]
                    confidence = max(0, float(best_similarity))
                    
                    # 当余弦相似度超过阈值时，判定为同一人
                    if best_similarity >= self.face_tolerance:
                        name = self.known_face_names[best_match_index]
                
                # 还原人脸位置到原始分辨率（因为之前缩小了图像）
                inv_scale = 1.0 / scale_factor
                top = int(y1 * inv_scale)
                right = int(x2 * inv_scale)
                bottom = int(y2 * inv_scale)
                left = int(x1 * inv_scale)
                
                face_locations.append((top, right, bottom, left))
                face_names.append(name)
                face_confidences.append(confidence)
            return face_locations, face_names, face_confidences
            
        except Exception as e:
            print(f"人脸识别错误：{e}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def draw_face_info(self, frame, face_locations, face_names, face_confidences):
        """在图像上绘制人脸信息"""
        for (top, right, bottom, left), name, confidence in zip(
            face_locations, face_names, face_confidences
        ):
            # 绘制人脸框
            color = (0, 255, 0) if name != "未知" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            label_lines = []
            if self.face_recognition_enabled:
                label_lines.append(f"{name} ({confidence:.1%})" if name != "未知" else "身份: 未知")

            if not label_lines:
                label_lines.append("人脸已检测")

            start_y = max(10, top - 24 * len(label_lines))
            for offset, label in enumerate(label_lines):
                frame = draw_chinese_text(
                    frame,
                    label,
                    (left, start_y + offset * 22),
                    font_size=20,
                    color=color,
                )
        
        return frame

    def get_stable_id(self, box, keypoints):
        """基于IOU和关键点位置给人物分配稳定ID"""
        self.track_threshold = 0.7
        # 注意：frame_count 已在 process_frame() 中递增，这里不再重复递增，避免隔帧逻辑被“加速”
        
        if self.frame_count == 1:
            for i in range(len(box)):
                self.id_mapping[i] = self.next_id
                self.next_id += 1
            return self.id_mapping

        current_boxes = {}
        for i, b in enumerate(box):
            best_iou = 0
            best_id = None
            curr_box = b.xyxy[0]

            for prev_id, state in self.student_states.items():
                if state["last_position"] is not None:
                    prev_box = state["last_position"]
                    iou = self.calculate_iou(curr_box, prev_box)
                    if iou > best_iou and iou > self.track_threshold:
                        best_iou = iou
                        best_id = prev_id

            if best_id is not None:
                current_boxes[i] = best_id
            else:
                current_boxes[i] = self.next_id
                self.next_id += 1

        self.id_mapping = current_boxes
        return self.id_mapping

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IOU（交并比）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / float(box1_area + box2_area - intersection)

    def calculate_body_action(self, keypoints):
        nose = keypoints[0]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        action_state = {'hand_raised': False, 'resting_head': False, 'sleeping': False}
        
        if nose[2] < self.confidence_threshold or left_shoulder[2] < self.confidence_threshold or right_shoulder[2] < self.confidence_threshold:
            return action_state
            
        # 1. 举手判定 (Hand Raising)
        l_wrist_high = left_wrist[2] > self.confidence_threshold and left_wrist[1] < nose[1]
        r_wrist_high = right_wrist[2] > self.confidence_threshold and right_wrist[1] < nose[1]
        if l_wrist_high or r_wrist_high:
            action_state['hand_raised'] = True

        # 2. 托腮判定 (Resting Head)
        def calc_dist(pt1, pt2):
            import math
            return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            
        face_radius = abs(left_shoulder[0] - right_shoulder[0]) * 0.4
        
        if left_wrist[2] > self.confidence_threshold and left_elbow[2] > self.confidence_threshold:
            if left_elbow[1] > left_wrist[1]:
                dist_to_face = min(calc_dist(left_wrist, nose), calc_dist(left_wrist, left_ear))
                if dist_to_face < face_radius:
                    action_state['resting_head'] = True
                    
        if right_wrist[2] > self.confidence_threshold and right_elbow[2] > self.confidence_threshold:
            if right_elbow[1] > right_wrist[1]:
                dist_to_face = min(calc_dist(right_wrist, nose), calc_dist(right_wrist, right_ear))
                if dist_to_face < face_radius:
                    action_state['resting_head'] = True

        # 3. 趴桌睡觉判定 (Sleeping on Desk)
        # 判定标准：鼻子下沉到接近或低于肩膀水平线，同时双眼置信度极低（脸朝下被遮挡）
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        
        if shoulder_width > 0:
            # 正常坐姿鼻子远在肩膀上方(Y值更小)，趴下时鼻子Y接近甚至大于肩膀Y
            nose_drop_ratio = (nose[1] - shoulder_mid_y) / shoulder_width
            eyes_invisible = (left_eye[2] < 0.3 and right_eye[2] < 0.3)
            nose_near_desk = nose_drop_ratio > -0.05
            # 睡觉需要更强的低头证据，避免把玩手机/看书误判为趴桌睡觉
            if eyes_invisible and nose_near_desk:
                action_state['sleeping'] = True

        return action_state

    def analyze_desk_context(self, person_box, keypoints, detected_items, allow_reading=True):
        if not detected_items:
            return None
            
        l_elbow = keypoints[7]
        r_elbow = keypoints[8]
        l_wrist = keypoints[9]
        r_wrist = keypoints[10]
        
        px1, py1, px2, py2 = person_box
        person_height = py2 - py1
        interaction_y_start = py1 + person_height * 0.4
        handheld_phone_y_end = py1 + person_height * 0.82
        
        def in_box(pt, box_coords, expand=40):
            if pt[2] < self.confidence_threshold: return False
            bx1, by1, bx2, by2 = box_coords
            return (bx1 - expand <= pt[0] <= bx2 + expand) and (by1 - expand <= pt[1] <= by2 + expand)
            
        for item in detected_items:
            if item['class_id'] not in [67, 73, 63]: 
                continue
                
            bx1, by1, bx2, by2 = item["bbox"]
            item_cx = (bx1 + bx2) / 2
            item_cy = (by1 + by2) / 2
            
            is_near_hands = (
                in_box(l_wrist, item["bbox"], expand=45)
                or in_box(r_wrist, item["bbox"], expand=45)
                or in_box(l_elbow, item["bbox"], expand=55)
                or in_box(r_elbow, item["bbox"], expand=55)
            )
            is_in_front = (px1 - 50 < item_cx < px2 + 50) and (item_cy > interaction_y_start)
            is_handheld_phone = (
                item['class_id'] == 67
                and px1 - 20 < item_cx < px2 + 20
                and py1 + person_height * 0.15 < item_cy < handheld_phone_y_end
                and is_near_hands
            )
            
            if item['class_id'] == 67 and (is_near_hands or is_in_front or is_handheld_phone):
                return "玩手机"
            if allow_reading and (is_near_hands or is_in_front):
                if item['class_id'] == 73 or item['class_id'] == 63:
                    return "阅读"
        return None

    def calculate_head_angle(self, keypoints):
        """计算头部角度，判断低头和转头状态"""
        nose = keypoints[0]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        if (nose[2] < self.confidence_threshold or 
            left_shoulder[2] < self.confidence_threshold or 
            right_shoulder[2] < self.confidence_threshold):
            return None

        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2

        dx = nose[0] - shoulder_mid_x
        dy = nose[1] - shoulder_mid_y
        
        main_angle = np.degrees(np.arctan2(dy, dx))

        if left_eye[2] > self.confidence_threshold and right_eye[2] > self.confidence_threshold:
            eye_mid_x = (left_eye[0] + right_eye[0]) / 2
            eye_mid_y = (left_eye[1] + right_eye[1]) / 2
        elif left_eye[2] > self.confidence_threshold:
            eye_mid_x = left_eye[0]
            eye_mid_y = left_eye[1]
        elif right_eye[2] > self.confidence_threshold:
            eye_mid_x = right_eye[0]
            eye_mid_y = right_eye[1]
        else:
            eye_mid_x = nose[0]
            eye_mid_y = nose[1] - 5

        eye_nose_dx = nose[0] - eye_mid_x
        eye_nose_dy = nose[1] - eye_mid_y
        eye_nose_angle = np.abs(np.degrees(np.arctan2(eye_nose_dy, eye_nose_dx)))

        ears_visible = left_ear[2] > self.confidence_threshold and right_ear[2] > self.confidence_threshold
        ear_eye_relation = False
        single_ear_visible = False
        single_ear_relation = False
        
        if ears_visible and left_eye[2] > self.confidence_threshold and right_eye[2] > self.confidence_threshold:
            ear_mid_y = (left_ear[1] + right_ear[1]) / 2
            eye_mid_y = (left_eye[1] + right_eye[1]) / 2
            ear_eye_relation = ear_mid_y < eye_mid_y
        elif left_ear[2] > self.confidence_threshold and left_eye[2] > self.confidence_threshold:
            single_ear_visible = True
            single_ear_relation = left_ear[1] < left_eye[1]
        elif right_ear[2] > self.confidence_threshold and right_eye[2] > self.confidence_threshold:
            single_ear_visible = True
            single_ear_relation = right_ear[1] < right_eye[1]

        ears_ratio = 0
        if left_ear[2] > self.confidence_threshold and right_ear[2] > self.confidence_threshold:
            left_distance = abs(nose[0] - left_ear[0])
            right_distance = abs(nose[0] - right_ear[0])
            if max(left_distance, right_distance) > 0:
                ears_ratio = min(left_distance, right_distance) / max(left_distance, right_distance)

        angle_threshold = self.head_down_threshold
        angle_based_head_down = abs(main_angle) < angle_threshold or abs(main_angle) > (180 - angle_threshold)
        
        vertical_ratio = abs(dy) / (abs(dx) + 1e-5)
        vertical_posture = vertical_ratio < 0.25
        
        nose_below_shoulder = nose[1] > shoulder_mid_y
        traditional_head_down = (main_angle > 90 and main_angle < 270) or nose_below_shoulder

        is_head_down = (traditional_head_down or angle_based_head_down or 
                       ear_eye_relation or single_ear_relation or vertical_posture)

        horizontal_angle = np.degrees(np.arctan2(dx, abs(dy)))
        profile_detected = ears_ratio < 0.5
        is_head_turned = abs(horizontal_angle) > self.head_turn_threshold or profile_detected

        return {
            "angle": main_angle,
            "is_head_down": is_head_down,
            "horizontal_angle": horizontal_angle,
            "is_head_turned": is_head_turned,
            "ears_ratio": ears_ratio,
            "debug_info": {
                "traditional": traditional_head_down,
                "angle_based": angle_based_head_down,
                "ear_eye_relation": ear_eye_relation,
                "single_ear_relation": single_ear_relation,
                "vertical_posture": vertical_posture,
                "eye_nose_angle": eye_nose_angle,
                "profile_detected": profile_detected
            }
        }

    def smooth_detection(self, person_id, is_distracted, window_size=3):
        state = self.student_states[person_id]
        state["position_history"].append(is_distracted)

        if len(state["position_history"]) > window_size:
            state["position_history"] = state["position_history"][-window_size:]

        distracted_frames = sum(state["position_history"])
        return distracted_frames >= window_size / 2

    def ensure_student_state_fields(self, state):
        state.setdefault("total_frames", 0)
        state.setdefault("focused_frames", 0)
        state.setdefault("head_up_frames", 0)
        state.setdefault("head_down_frames", 0)
        state.setdefault("head_turn_frames", 0)
        state.setdefault("distracted_frames", 0)
        state.setdefault("habit_label", "insufficient_data")
        state.setdefault("identity", "未知")

    def classify_learning_habit(self, metrics):
        if metrics["total_frames"] < 10:
            return "数据不足"
        if metrics["focus_rate"] >= 80 and metrics["head_up_rate"] >= 70 and metrics["head_turn_rate"] < 20:
            return "稳定听讲型"
        if metrics["head_down_rate"] >= 40 and metrics["focus_rate"] >= 55:
            return "前倾低头型"
        if metrics["head_turn_rate"] >= 35 and metrics["focus_rate"] >= 55:
            return "侧向关注型"
        if metrics["focus_rate"] < 55:
            return "易分心型"
        return "混合型"

    def calculate_student_metrics(self, person_id):
        state = self.student_states[person_id]
        self.ensure_student_state_fields(state)
        total_frames = state["total_frames"]
        
        if total_frames <= 0:
            return {
                "total_frames": 0,
                "focus_rate": 0.0,
                "head_up_rate": 0.0,
                "head_down_rate": 0.0,
                "head_turn_rate": 0.0,
                "distracted_rate": 0.0,
                "habit_label": "数据不足"
            }

        metrics = {
            "total_frames": total_frames,
            "focus_rate": state["focused_frames"] / total_frames * 100,
            "head_up_rate": state["head_up_frames"] / total_frames * 100,
            "head_down_rate": state["head_down_frames"] / total_frames * 100,
            "head_turn_rate": state["head_turn_frames"] / total_frames * 100,
            "distracted_rate": state["distracted_frames"] / total_frames * 100,
        }
        metrics["habit_label"] = self.classify_learning_habit(metrics)
        return metrics

    def update_student_statistics(self, person_id, student_number, is_head_down, is_head_turned, is_focused):
        state = self.student_states[person_id]
        self.ensure_student_state_fields(state)
        state["student_number"] = student_number
        state["total_frames"] += 1
        state["focused_frames"] += int(is_focused)
        state["head_up_frames"] += int(not is_head_down)
        state["head_down_frames"] += int(is_head_down)
        state["head_turn_frames"] += int(is_head_turned)
        state["distracted_frames"] += int(not is_focused)

        metrics = self.calculate_student_metrics(person_id)
        state["habit_label"] = metrics["habit_label"]
        return metrics

    def calculate_classroom_metrics(self):
        student_metrics = []
        habit_counts = defaultdict(int)

        for person_id, state in self.student_states.items():
            self.ensure_student_state_fields(state)
            if state["student_number"] is None or state["total_frames"] <= 0:
                continue
            metrics = self.calculate_student_metrics(person_id)
            student_metrics.append(metrics)
            habit_counts[metrics["habit_label"]] += 1

        if not student_metrics:
            self.latest_class_metrics = {
                "students_analyzed": 0,
                "focus_rate": 0.0,
                "head_up_rate": 0.0,
                "dominant_habit": "insufficient_data"
            }
            return self.latest_class_metrics

        focus_rate = sum(item["focus_rate"] for item in student_metrics) / len(student_metrics)
        head_up_rate = sum(item["head_up_rate"] for item in student_metrics) / len(student_metrics)
        dominant_habit = max(habit_counts.items(), key=lambda item: item[1])[0] if habit_counts else "mixed"

        self.latest_class_metrics = {
            "students_analyzed": len(student_metrics),
            "focus_rate": focus_rate,
            "head_up_rate": head_up_rate,
            "dominant_habit": dominant_habit
        }
        return self.latest_class_metrics

    def get_parameter_snapshot(self):
        return {
            "head_down_threshold": self.head_down_threshold,
            "time_threshold": self.time_threshold,
            "head_turn_threshold": self.head_turn_threshold,
            "confidence_threshold": self.confidence_threshold,
            "filter_size": self.filter_size,
            "total_students": self.total_students,
            "debug": self.debug,
            "beep_enabled": self.beep_enabled,
            "object_detection_enabled": self.object_detection_enabled,
            "face_recognition_enabled": self.face_recognition_enabled,
        }

    def get_student_alert_count(self, person_id):
        logs = self.attention_logs.get(person_id, [])
        if not logs:
            return 0

        alert_count = 0
        previous_alert = False
        for log in logs:
            current_alert = bool(log.get("alert", False))
            if current_alert and not previous_alert:
                alert_count += 1
            previous_alert = current_alert
        return alert_count

    def get_warning_events(self):
        warning_events = []
        for person_id, logs in self.attention_logs.items():
            previous_alert = False
            for log in logs:
                current_alert = bool(log.get("alert", False))
                if current_alert and not previous_alert:
                    warning_events.append({
                        "timestamp": log.get("timestamp", 0),
                        "person_id": person_id,
                        "student_number": log.get("student_number"),
                        "identity": log.get("identity", "未知"),
                    })
                previous_alert = current_alert

        warning_events.sort(key=lambda item: item["timestamp"])
        return warning_events

    def detect_desk_objects(self, frame):
        if not self.object_detection_enabled:
            self.detected_objects = []
            return []
        ok, message = self.ensure_object_model_loaded()
        if not ok:
            print(message)
            self.detected_objects = []
            return []
        
        results = self.object_model.predict(
            frame, 
            classes=self.desk_object_classes,
            conf=0.3,
            verbose=False,
            imgsz=416,
            half=self.face_device.type == "cuda"
        )
        
        detected_items = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                class_names = {
                    67: "手机",
                    73: "书籍", 
                    63: "笔记本电脑",
                    66: "键盘",
                    64: "鼠标",
                    41: "杯子"
                }
                class_name = class_names.get(class_id, "物品")
                
                detected_items.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })
        
        self.detected_objects = detected_items
        return detected_items

    def detect_behavior_targets(self, frame):
        if not self.behavior_model_enabled or self.behavior_model is None:
            ok, message = self.ensure_behavior_model_loaded()
            if not ok:
                if message != "行为检测模型未启用":
                    print(message)
                return []

        results = self.behavior_model.predict(
            frame,
            conf=self.behavior_model_conf,
            verbose=False,
            imgsz=self.behavior_model_imgsz,
            half=self.face_device.type == "cuda"
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            names = getattr(result, "names", getattr(self.behavior_model, "names", {}))
            boxes = result.boxes.cpu().numpy() if result.boxes is not None else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                source_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                target_label = self.translate_behavior_class(source_name)
                detections.append({
                    "label": target_label,
                    "source_name": source_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })
        return detections

    def match_behavior_to_person(self, person_box, behavior_detections):
        px1, py1, px2, py2 = person_box
        best_match = None
        best_score = 0.0

        for item in behavior_detections:
            bx1, by1, bx2, by2 = item["bbox"]
            center_x = (bx1 + bx2) / 2
            center_y = (by1 + by2) / 2
            center_in_person = px1 <= center_x <= px2 and py1 <= center_y <= py2
            overlap = self.calculate_iou(person_box, item["bbox"])
            if overlap < 0.1 and not center_in_person:
                continue

            score = overlap + item["confidence"] * 0.2 + (0.05 if center_in_person else 0.0)
            if score > best_score:
                best_score = score
                best_match = item

        return best_match

    def draw_detected_objects(self, frame, detected_items):
        for item in detected_items:
            x1, y1, x2, y2 = item["bbox"]
            label = f"{item['class_name']} {item['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.object_color, 2)
            frame = draw_chinese_text(frame, label, (x1, y1 - 25), font_size=20, color=self.object_color)
        return frame

    def draw_pose_skeleton(self, frame, keypoints, color):
        if keypoints is None:
            return frame

        threshold = max(self.keypoint_threshold, self.confidence_threshold)
        points = []
        for point in keypoints:
            x, y, conf = point
            if conf >= threshold:
                px, py = int(x), int(y)
                points.append((px, py))
                cv2.circle(frame, (px, py), 4, color, -1)
            else:
                points.append(None)

        for start_idx, end_idx in self.skeleton_edges:
            start_point = points[start_idx]
            end_point = points[end_idx]
            if start_point is not None and end_point is not None:
                cv2.line(frame, start_point, end_point, color, 2)

        return frame

    def process_frame(self, frame, current_time):
        self.frame_count += 1
        analysis_frame = frame.copy()
        display_frame = frame.copy()
        is_local_video = isinstance(self.video_source, str)
        simple_pose_only = bool(getattr(self, "simple_pose_only", False))

        if (not simple_pose_only) and self.object_detection_enabled:
            refresh_objects = (
                not is_local_video
                or self.frame_count % self.local_video_object_stride == 0
                or not self._cached_detected_objects
            )
            if refresh_objects:
                detected_objects = self.detect_desk_objects(analysis_frame)
                self._cached_detected_objects = detected_objects
            else:
                detected_objects = self._cached_detected_objects
            self.detected_objects = detected_objects
        else:
            self.detected_objects = []
            self._cached_detected_objects = []
            detected_objects = []

        if (not simple_pose_only) and self.behavior_model_enabled:
            refresh_behaviors = (
                not is_local_video
                or self.frame_count % self.local_video_behavior_stride == 0
                or not self._cached_behavior_detections
            )
            if refresh_behaviors:
                behavior_detections = self.detect_behavior_targets(analysis_frame)
                self._cached_behavior_detections = behavior_detections
            else:
                behavior_detections = self._cached_behavior_detections
        else:
            self._cached_behavior_detections = []
            behavior_detections = []

        if (not simple_pose_only) and self.face_recognition_enabled:
            refresh_faces = (
                not is_local_video
                or self.frame_count % self.local_video_face_stride == 0
                or not self._cached_face_results[0]
            )
            if refresh_faces:
                face_locations, face_names, face_confidences = self.recognize_faces(analysis_frame)
                self._cached_face_results = (face_locations, face_names, face_confidences)
            else:
                face_locations, face_names, face_confidences = self._cached_face_results
        else:
            self._cached_face_results = ([], [], [])
            face_locations, face_names, face_confidences = [], [], []

        pose_conf = self.confidence_threshold
        pose_imgsz = 320
        if is_local_video:
            pose_conf = max(0.2, min(self.confidence_threshold, self.local_video_pose_conf))
            pose_imgsz = self.local_video_pose_imgsz

        # 性能优化：姿态推理是最大耗时，允许本地视频/低配机器隔帧推理并复用上一次结果
        pose_stride = 1
        if is_local_video:
            pose_stride = int(getattr(self, "local_video_pose_stride", 2)) or 2
        else:
            pose_stride = int(getattr(self, "camera_pose_stride", 1)) or 1

        use_cached_pose = (
            pose_stride > 1
            and (self.frame_count % pose_stride != 0)
            and hasattr(self, "_cached_pose_result")
            and self._cached_pose_result is not None
        )

        if use_cached_pose:
            boxes, keypoints = self._cached_pose_result
            self.current_count = len(boxes) if boxes is not None else 0
        else:
            results = self.model.predict(
                analysis_frame,
                classes=[0],
                conf=pose_conf,
                verbose=False,
                imgsz=pose_imgsz,
                half=self.face_device.type == "cuda"
            )

        # 更新当前检测到的人数
        if (not use_cached_pose) and len(results) > 0:
            result = results[0]
            boxes = result.boxes.cpu().numpy()
            self.current_count = len(boxes)
            keypoints = result.keypoints.cpu().numpy() if hasattr(result, 'keypoints') else None
            self._cached_pose_result = (boxes, keypoints)
        elif use_cached_pose:
            # boxes/keypoints 已从缓存取出
            pass
        else:
            boxes = []
            keypoints = None
            self.current_count = 0

        if keypoints is not None and len(boxes) > 0:
                # 获取稳定的人物ID
                id_mapping = self.get_stable_id(boxes, keypoints)

                # 创建学生位置列表，用于排序
                student_positions = []
                for i, box in enumerate(boxes):
                    box_x1, box_y1, box_x2, box_y2 = map(int, box.xyxy[0])
                    center_x = (box_x1 + box_x2) // 2
                    center_y = (box_y1 + box_y2) // 2
                    student_positions.append((i, center_x, center_y, box_x1, box_y1, box_x2, box_y2))

                # 按从左到右、从下到上排序
                student_positions.sort(key=lambda p: (p[2], -p[1]), reverse=True)

                # 为每个学生分配编号（从1开始）
                for position_index, (i, center_x, center_y, box_x1, box_y1, box_x2, box_y2) in enumerate(student_positions):
                    student_number = position_index + 1
                    person_id = id_mapping[i]

                    # 更新学生位置记录
                    self.student_states[person_id]["last_position"] = boxes[i].xyxy[0]
                    self.student_states[person_id]["student_number"] = student_number

                    # ========== 尝试识别人脸身份 ==========
                    student_identity = "未知"
                    if len(face_locations) > 0:
                        # 计算人体框与人脸框的重叠，匹配身份
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # 检查人脸是否在人体框内
                            face_center_x = (left + right) // 2
                            face_center_y = (top + bottom) // 2
                            
                            if box_x1 < face_center_x < box_x2 and box_y1 < face_center_y < box_y2:
                                if name != "未知":
                                    student_identity = name
                                break
                    
                    self.student_states[person_id]["identity"] = student_identity

                    # 获取关键点
                    kpt = keypoints[i]
                    head_angle_info = self.calculate_head_angle(kpt.data[0])
                    body_action_info = self.calculate_body_action(kpt.data[0])
                    matched_behavior = self.match_behavior_to_person(
                        (box_x1, box_y1, box_x2, box_y2),
                        behavior_detections,
                    )
                    model_behavior = matched_behavior["label"] if matched_behavior else None
                    behavior_conf = matched_behavior["confidence"] if matched_behavior else None

                    if head_angle_info is not None:
                        is_head_down = head_angle_info["is_head_down"]
                        is_head_turned = False if simple_pose_only else head_angle_info["is_head_turned"]
                        if model_behavior == "睡觉":
                            body_action_info["sleeping"] = True
                        elif model_behavior in {"玩手机", "阅读"}:
                            body_action_info["sleeping"] = False

                        desk_behavior = None
                        if not simple_pose_only:
                            if model_behavior == "玩手机":
                                desk_behavior = "玩手机"
                            elif model_behavior == "阅读" and is_head_down:
                                desk_behavior = "阅读"
                            object_behavior = self.analyze_desk_context(
                                (box_x1, box_y1, box_x2, box_y2),
                                kpt.data[0],
                                detected_objects,
                                allow_reading=is_head_down,
                            )
                            if object_behavior == "玩手机":
                                desk_behavior = "玩手机"
                            elif desk_behavior is None:
                                desk_behavior = object_behavior

                        # Integrate body actions into distraction flag
                        if simple_pose_only:
                            is_distracted = bool(is_head_down)
                        elif model_behavior == "睡觉":
                            is_distracted = True
                        elif desk_behavior == "玩手机":
                            is_distracted = True
                        elif desk_behavior == "阅读":
                            is_distracted = False
                        elif model_behavior == "举手":
                            is_distracted = False
                        elif model_behavior in {"写作", "开卷阅读", "合书", "电子书", "课堂作业", "起立作答"}:
                            is_distracted = False
                        elif model_behavior in {"低头", "转头"}:
                            is_distracted = True
                        elif body_action_info.get('sleeping'):
                            is_distracted = True
                        elif body_action_info.get('hand_raised'):
                            is_distracted = False
                        elif body_action_info.get('resting_head'):
                            is_distracted = True
                        else:
                            is_distracted = is_head_down or is_head_turned

                        if (not simple_pose_only) and desk_behavior in {"玩手机", "阅读"}:
                            body_action_info['sleeping'] = False

                        # 平滑检测结果，减少抖动
                        is_distracted = self.smooth_detection(person_id, is_distracted, self.filter_size)
                        is_focused = not is_distracted

                        student_state = self.student_states[person_id]
                        self.ensure_student_state_fields(student_state)

                        if is_distracted and not student_state["head_down"]:
                            student_state["head_down"] = True
                            student_state["head_down_start_time"] = current_time
                        elif not is_distracted and student_state["head_down"]:
                            student_state["head_down"] = False
                            student_state["warning_issued"] = False

                        if student_state["head_down"]:
                            head_down_duration = current_time - student_state["head_down_start_time"]
                            if head_down_duration >= self.time_threshold and not student_state["warning_issued"]:
                                student_state["warning_issued"] = True
                                if self.beep_enabled:
                                    winsound.Beep(1000, 500)

                        metrics = self.update_student_statistics(
                            person_id,
                            student_number,
                            is_head_down,
                            is_head_turned,
                            is_focused
                        )

                        self.attention_logs[person_id].append({
                            "timestamp": time.time(),
                            "focused": is_focused,
                            "head_up": not is_head_down,
                            "head_down": is_head_down,
                            "head_turned": is_head_turned,
                            "alert": student_state["warning_issued"],
                            "focus_rate": metrics["focus_rate"],
                            "head_up_rate": metrics["head_up_rate"],
                            "habit": metrics["habit_label"],
                            "student_number": student_number,
                            "identity": student_identity,
                        })

                        color = self.warning_color if student_state["warning_issued"] else self.attention_color
                        display_frame = self.draw_pose_skeleton(display_frame, kpt.data[0], color)
                        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), color, 2)

                        # 显示学生编号、身份和状态
                        number_text = f"#{student_number}"
                        if student_identity != "未知":
                            number_text = f"#{student_number} {student_identity}"

                        # 画面标签
                        if simple_pose_only:
                            display_behavior = "低头" if is_head_down else "抬头"
                            color = (0, 140, 255) if is_head_down else self.attention_color
                        else:
                            # 显示具体行为名称（及模型置信度），不再使用泛化的「警告」
                            if model_behavior == "睡觉" or body_action_info.get("sleeping"):
                                display_behavior = "睡觉"
                                color = (128, 0, 255)
                            elif desk_behavior == "玩手机" or model_behavior == "玩手机":
                                display_behavior = "玩手机"
                                color = (0, 0, 255)
                            elif desk_behavior == "阅读" or model_behavior == "阅读":
                                display_behavior = "阅读"
                                color = (0, 255, 0)
                            elif model_behavior == "举手" or (body_action_info and body_action_info.get("hand_raised")):
                                display_behavior = "举手"
                                color = (0, 255, 255)
                            elif model_behavior == "低头" or (body_action_info and body_action_info.get("resting_head")):
                                display_behavior = "托腮走神" if body_action_info.get("resting_head") else "低头"
                                color = (0, 165, 255) if body_action_info.get("resting_head") else (0, 140, 255)
                            elif model_behavior == "转头":
                                display_behavior = "转头"
                                color = (255, 128, 0)
                            elif model_behavior:
                                display_behavior = model_behavior
                                color = (200, 220, 255)
                            elif is_head_turned:
                                display_behavior = "转头"
                                color = (255, 128, 0)
                            elif is_head_down:
                                display_behavior = "低头"
                                color = (0, 140, 255)
                            elif is_focused:
                                display_behavior = "专注听讲"
                                color = self.attention_color
                            else:
                                display_behavior = "非专注"
                                color = self.warning_color

                        if student_state["warning_issued"]:
                            color = self.warning_color

                        label_suffix = "" if simple_pose_only else (f" {behavior_conf:.0%}" if behavior_conf is not None else "")
                        overlay_text = f"{number_text} {display_behavior}{label_suffix}"
                        student_state["last_behavior_zh"] = display_behavior
                        student_state["last_behavior_conf"] = behavior_conf

                        sample_conf = behavior_conf if behavior_conf is not None else (0.82 if self.behavior_model_enabled else None)
                        self.record_session_behavior_sample(display_behavior, sample_conf, is_focused)

                        display_frame = draw_chinese_text(display_frame, overlay_text, (box_x1, box_y1 - 30),
                                                          font_size=30, color=color)

                        if self.debug and head_angle_info is not None and not is_local_video:
                            angle_text = f"头部角度: {head_angle_info['angle']:.1f}°, 转头角度: {head_angle_info['horizontal_angle']:.1f}°"
                            display_frame = draw_chinese_text(display_frame, angle_text, (box_x1, box_y2 + 20), font_size=20,
                                                              color=(255, 255, 0))

                            debug_info = head_angle_info["debug_info"]
                            debug_text = f"T:{debug_info['traditional']} A:{debug_info['angle_based']} E:{debug_info['ear_eye_relation']} S:{debug_info['single_ear_relation']} P:{debug_info['profile_detected']:.2f}"
                            display_frame = draw_chinese_text(display_frame, debug_text, (box_x1, box_y2 + 40), font_size=16,
                                                              color=(255, 255, 0))
        else:
            self.current_count = 0

        if (not simple_pose_only) and detected_objects:
            display_frame = self.draw_detected_objects(display_frame, detected_objects)

        if (not simple_pose_only) and len(face_locations) > 0:
            display_frame = self.draw_face_info(display_frame, face_locations, face_names, face_confidences)

        # 在画面右上角添加出勤率信息
        attendance_rate = self.current_count / self.total_students * 100 if self.total_students > 0 else 0
        attendance_text = f"出勤率: {self.current_count}/{self.total_students} ({attendance_rate:.1f}%)"
        class_metrics = self.calculate_classroom_metrics()
        summary_text = f"专注度 {class_metrics['focus_rate']:.1f}% | 抬头率 {class_metrics['head_up_rate']:.1f}%"
        habit_text = f"主导习惯: {class_metrics['dominant_habit']}"
        # 性能优化：合并多段中文绘制，减少 PIL 往返次数
        display_frame = draw_chinese_texts(
            display_frame,
            [
                (attendance_text, (display_frame.shape[1] - 300, 70), 24, (255, 255, 255)),
                (summary_text, (max(20, display_frame.shape[1] - 420), 100), 22, (255, 255, 255)),
                (habit_text, (max(20, display_frame.shape[1] - 420), 130), 20, (255, 255, 255)),
            ],
        )
        
        # 显示检测到的物品数量
        if (not simple_pose_only) and detected_objects:
            object_text = f"检测到桌面物品: {len(detected_objects)}个"
            display_frame = draw_chinese_text(display_frame, object_text, (20, max(20, display_frame.shape[0] - 50)), font_size=20,
                                              color=self.object_color)
        
        # 显示人脸识别状态
        if (not simple_pose_only) and self.face_recognition_enabled:
            recognized_count = sum(1 for state in self.student_states.values() 
                                 if state.get("identity", "未知") != "未知")
            face_text = f"已识别学生: {recognized_count}人"
            display_frame = draw_chinese_text(display_frame, face_text, (20, max(20, display_frame.shape[0] - 80)), font_size=20,
                                              color=self.identity_color)

        # 如需扩展，可在这里添加录像指示器

        return display_frame

    def process_local_behavior_preview(self, frame, local_behavior_preview_conf=0.6):
        """本地视频暂停帧预览：临时跑行为模型，只绘制画面，不写入统计。"""
        analysis_frame = frame.copy()
        display_frame = frame.copy()
        behavior_detections = []
        behavior_model_ready = False
        preview_conf = float(local_behavior_preview_conf)

        previous_behavior_enabled = self.behavior_model_enabled
        previous_behavior_conf = self.behavior_model_conf
        try:
            if self.behavior_model is not None or os.path.exists(self.behavior_model_path):
                self.behavior_model_enabled = True
                self.behavior_model_conf = preview_conf
                behavior_detections = self.detect_behavior_targets(analysis_frame)
                behavior_model_ready = self.behavior_model is not None
        finally:
            self.behavior_model_conf = previous_behavior_conf
            self.behavior_model_enabled = previous_behavior_enabled

        results = self.model.predict(
            analysis_frame,
            classes=[0],
            conf=max(0.2, min(self.confidence_threshold, self.local_video_pose_conf)),
            verbose=False,
            imgsz=self.local_video_pose_imgsz,
            half=self.face_device.type == "cuda"
        )

        if not results:
            return display_frame, behavior_model_ready

        result = results[0]
        boxes = result.boxes.cpu().numpy()
        keypoints = result.keypoints.cpu().numpy() if hasattr(result, 'keypoints') else None
        if keypoints is None or len(boxes) <= 0:
            return display_frame, behavior_model_ready

        student_positions = []
        for i, box in enumerate(boxes):
            box_x1, box_y1, box_x2, box_y2 = map(int, box.xyxy[0])
            center_x = (box_x1 + box_x2) // 2
            center_y = (box_y1 + box_y2) // 2
            student_positions.append((i, center_x, center_y, box_x1, box_y1, box_x2, box_y2))
        student_positions.sort(key=lambda p: (p[2], -p[1]), reverse=True)

        for position_index, (i, _center_x, _center_y, box_x1, box_y1, box_x2, box_y2) in enumerate(student_positions):
            kpt = keypoints[i]
            head_angle_info = self.calculate_head_angle(kpt.data[0])
            if head_angle_info is None:
                continue

            matched_behavior = self.match_behavior_to_person(
                (box_x1, box_y1, box_x2, box_y2),
                behavior_detections,
            ) if behavior_model_ready else None
            behavior_conf = matched_behavior["confidence"] if matched_behavior else None
            model_behavior = (
                matched_behavior["label"]
                if behavior_conf is not None and behavior_conf >= preview_conf
                else None
            )

            is_head_down = head_angle_info["is_head_down"]

            display_behavior = "低头" if is_head_down else "抬头"
            color = (0, 140, 255) if is_head_down else self.attention_color
            if model_behavior == "睡觉":
                display_behavior = "睡觉"
                color = (128, 0, 255)
            elif model_behavior == "玩手机":
                display_behavior = "玩手机"
                color = (0, 0, 255)
            elif model_behavior == "阅读":
                display_behavior = "阅读"
                color = (0, 255, 0)
            elif model_behavior == "举手":
                display_behavior = "举手"
                color = (0, 255, 255)
            elif model_behavior == "低头":
                display_behavior = "低头"
                color = (0, 140, 255)
            elif model_behavior == "转头":
                display_behavior = "转头"
                color = (255, 128, 0)
            elif model_behavior:
                display_behavior = model_behavior
                color = (200, 220, 255)

            display_frame = self.draw_pose_skeleton(display_frame, kpt.data[0], color)
            cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), color, 2)

            label_suffix = f" {behavior_conf:.0%}" if model_behavior and behavior_conf is not None else ""
            overlay_text = f"#{position_index + 1} {display_behavior}{label_suffix}"
            display_frame = draw_chinese_text(
                display_frame,
                overlay_text,
                (box_x1, box_y1 - 30),
                font_size=30,
                color=color,
            )

        return display_frame, behavior_model_ready

    def plot_attention_logs(self, save_dir="attention_logs"):
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        for person_id, logs in self.attention_logs.items():
            if len(logs) < 2:
                continue

            timestamps = [log["timestamp"] - logs[0]["timestamp"] for log in logs]
            attention_values = [1 if log["focused"] else 0 for log in logs]

            focus_percent = (sum(attention_values) / len(attention_values)) * 100

            student_number = ""
            identity = ""
            if person_id in self.student_states:
                state = self.student_states[person_id]
                if "student_number" in state:
                    student_number = f"#{state['student_number']}"
                if "identity" in state and state["identity"] != "未知":
                    identity = f" ({state['identity']})"

            label = f"学生 {student_number}{identity} (ID {person_id}, 专注度: {focus_percent:.1f}%)"
            plt.plot(timestamps, attention_values, label=label)

        plt.xlabel("时间 (秒)")
        plt.ylabel("专注状态 (1=专注, 0=分心)")
        plt.title("课堂专注度时间曲线")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, "attention_plot.png"), dpi=300)
        plt.close()

        self.plot_behavior_summary(save_dir)
        self.plot_warning_timeline(save_dir)
        self.plot_session_behavior_dashboard(save_dir)

        self.generate_summary_report(save_dir)

    def plot_behavior_summary(self, save_dir):
        """可视化行为统计概览"""
        student_metrics = []
        for person_id, state in self.student_states.items():
            self.ensure_student_state_fields(state)
            if state["student_number"] is None or state["total_frames"] < 2:
                continue
            student_metrics.append((state["student_number"], self.calculate_student_metrics(person_id), state.get("identity", "未知")))

        if not student_metrics:
            return

        student_metrics.sort(key=lambda item: item[0])
        labels = []
        for number, _, identity in student_metrics:
            if identity != "未知":
                labels.append(f"#{number}\n{identity}")
            else:
                labels.append(f"#{number}")
        
        focus_rates = [metrics["focus_rate"] for _, metrics, _ in student_metrics]
        head_up_rates = [metrics["head_up_rate"] for _, metrics, _ in student_metrics]
        habits = [metrics["habit_label"] for _, metrics, _ in student_metrics]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(12, 8))
        plt.bar(x - width / 2, focus_rates, width, label="专注度")
        plt.bar(x + width / 2, head_up_rates, width, label="抬头率")
        plt.ylim(0, 100)
        plt.xticks(x, labels)
        plt.ylabel("比例 (%)")
        plt.title("课堂行为统计概览")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        for idx, habit in enumerate(habits):
            top = max(focus_rates[idx], head_up_rates[idx])
            plt.text(x[idx], min(98, top + 2), habit, ha="center", va="bottom", fontsize=8, rotation=20)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "behavior_summary.png"), dpi=300)
        plt.close()

    def plot_warning_timeline(self, save_dir):
        """绘制告警时间线。"""
        warning_events = self.get_warning_events()
        plt.figure(figsize=(12, 6))

        if not warning_events:
            plt.text(0.5, 0.5, "当前无告警数据", ha="center", va="center", fontsize=14)
            plt.axis("off")
        else:
            first_timestamp = warning_events[0]["timestamp"]
            time_buckets = defaultdict(int)
            for event in warning_events:
                relative_second = max(0, int(event["timestamp"] - first_timestamp))
                time_buckets[relative_second] += 1

            timeline_seconds = sorted(time_buckets.keys())
            timeline_counts = [time_buckets[sec] for sec in timeline_seconds]

            plt.plot(timeline_seconds, timeline_counts, marker="o", color="#EF5350")
            plt.xlabel("时间 (秒)")
            plt.ylabel("告警触发次数")
            plt.title("课堂告警时间线")
            plt.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "warning_timeline.png"), dpi=300)
        plt.close()

    def generate_summary_report(self, save_dir):
        def _metrics_from_logs(logs):
            total = len(logs)
            if total <= 0:
                return {
                    "total_frames": 0,
                    "focus_rate": 0.0,
                    "head_up_rate": 0.0,
                    "head_down_rate": 0.0,
                    "head_turn_rate": 0.0,
                    "distracted_rate": 0.0,
                    "habit_label": "数据不足",
                }

            focused = sum(1 for row in logs if row.get("focused"))
            head_up = sum(1 for row in logs if row.get("head_up"))
            head_down = sum(1 for row in logs if row.get("head_down"))
            head_turn = sum(1 for row in logs if row.get("head_turned"))
            distracted = total - focused
            metrics = {
                "total_frames": total,
                "focus_rate": focused / total * 100.0,
                "head_up_rate": head_up / total * 100.0,
                "head_down_rate": head_down / total * 100.0,
                "head_turn_rate": head_turn / total * 100.0,
                "distracted_rate": distracted / total * 100.0,
            }
            metrics["habit_label"] = self.classify_learning_habit(metrics)
            return metrics

        # 1) 常规模式：按学生编号汇总（支持人脸识别/座位编号/多学生统计）
        all_students = {}
        for person_id, state in self.student_states.items():
            self.ensure_student_state_fields(state)
            if state["student_number"] is None or state["total_frames"] < 2:
                continue
            all_students[state["student_number"]] = {
                "metrics": self.calculate_student_metrics(person_id),
                "identity": state.get("identity", "未知"),
                "alert_count": self.get_student_alert_count(person_id),
            }

        # 2) 兼容本地视频“仅抬头/低头”：当无法按编号汇总时，退化为按会话日志汇总
        #    这样即便禁用了人脸/行为模型、只跑了低头/抬头，也能生成有内容的报告。
        using_fallback = False
        if not all_students:
            using_fallback = True
            seq_number = 0
            for person_id, logs in self.attention_logs.items():
                if len(logs) < 2:
                    continue
                seq_number += 1
                state = self.student_states.get(person_id, {})
                identity = state.get("identity", "未知") if isinstance(state, dict) else "未知"
                all_students[seq_number] = {
                    "metrics": _metrics_from_logs(logs),
                    "identity": identity,
                    "alert_count": self.get_student_alert_count(person_id),
                }

        class_metrics = self.calculate_classroom_metrics()
        if (not using_fallback) and (class_metrics.get("students_analyzed", 0) <= 0):
            class_metrics = self.latest_class_metrics

        if using_fallback:
            # 仅依赖 attention_logs 做班级层汇总，避免被 student_number 过滤导致 0 数据
            items = [row["metrics"] for _, row in all_students.items()] if all_students else []
            if items:
                focus_rate = sum(m["focus_rate"] for m in items) / len(items)
                head_up_rate = sum(m["head_up_rate"] for m in items) / len(items)
                habit_counts = defaultdict(int)
                for m in items:
                    habit_counts[m.get("habit_label", "数据不足")] += 1
                dominant_habit = max(habit_counts.items(), key=lambda it: it[1])[0] if habit_counts else "insufficient_data"
                class_metrics = {
                    "students_analyzed": len(items),
                    "focus_rate": focus_rate,
                    "head_up_rate": head_up_rate,
                    "dominant_habit": dominant_habit,
                }
            else:
                class_metrics = {
                    "students_analyzed": 0,
                    "focus_rate": 0.0,
                    "head_up_rate": 0.0,
                    "dominant_habit": "insufficient_data",
                }

        analyzed_students = len(all_students)
        attendance_rate = analyzed_students / self.total_students * 100 if self.total_students > 0 else 0
        parameter_snapshot = self.get_parameter_snapshot()
        total_alert_count = sum(student["alert_count"] for student in all_students.values())

        with open(os.path.join(save_dir, "attention_report.txt"), "w", encoding="utf-8") as f:
            f.write("课堂行为分析报告\n")
            f.write("====================\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if using_fallback:
                f.write("报告模式: 本地视频兼容（基于会话日志汇总：抬头/低头为主）\n")
            f.write(f"参与统计学生数: {analyzed_students}\n")
            f.write(f"班级总人数: {self.total_students}\n")
            f.write(f"出勤率: {attendance_rate:.2f}%\n")
            f.write(f"平均专注度: {class_metrics['focus_rate']:.2f}%\n")
            f.write(f"平均抬头率: {class_metrics['head_up_rate']:.2f}%\n")
            f.write(f"主导听课习惯: {class_metrics['dominant_habit']}\n\n")

            f.write("参数快照\n")
            f.write("--------------------\n")
            f.write(f"低头检测阈值: {parameter_snapshot['head_down_threshold']}\n")
            f.write(f"分心时间阈值: {parameter_snapshot['time_threshold']} 秒\n")
            f.write(f"转头角度阈值: {parameter_snapshot['head_turn_threshold']}\n")
            f.write(f"检测置信度阈值: {parameter_snapshot['confidence_threshold']}\n")
            f.write(f"平滑窗口大小: {parameter_snapshot['filter_size']}\n")
            f.write(f"班级总人数设定: {parameter_snapshot['total_students']}\n")
            f.write(f"调试信息开关: {'开启' if parameter_snapshot['debug'] else '关闭'}\n")
            f.write(f"声音提醒开关: {'开启' if parameter_snapshot['beep_enabled'] else '关闭'}\n")
            f.write(f"桌面物品检测: {'开启' if parameter_snapshot['object_detection_enabled'] else '关闭'}\n")
            f.write(f"人脸识别: {'开启' if parameter_snapshot['face_recognition_enabled'] else '关闭'}\n")
            f.write("\n")

            f.write("班级摘要\n")
            f.write("--------------------\n")
            f.write(f"总告警次数: {total_alert_count}\n\n")

            f.write("学生指标明细\n")
            f.write("--------------------\n")
            if not all_students:
                f.write("无有效学生统计数据（可能未检测到人体或视频过短）。\n")
            else:
                for number in sorted(all_students.keys()):
                    student = all_students[number]
                    metrics = student["metrics"]
                    identity = student["identity"]
                    identity_str = f" ({identity})" if identity != "未知" else ""
                    f.write(
                        f"学生 #{number}{identity_str}: 专注度={metrics['focus_rate']:.2f}%, "
                        f"抬头率={metrics['head_up_rate']:.2f}%, "
                        f"低头占比={metrics['head_down_rate']:.2f}%, "
                        f"转头占比={metrics['head_turn_rate']:.2f}%, "
                        f"习惯={metrics['habit_label']}, "
                        f"告警次数={student['alert_count']}\n"
                    )

            needs_attention = [
                (num, student) for num, student in all_students.items()
                if student["metrics"]["focus_rate"] < 70 or student["metrics"]["head_up_rate"] < 60
            ]

            f.write("\n需要重点关注的学生\n")
            f.write("--------------------\n")
            if needs_attention:
                for number, student in sorted(needs_attention, key=lambda item: item[1]["metrics"]["focus_rate"]):
                    metrics = student["metrics"]
                    identity = student["identity"]
                    identity_str = f" ({identity})" if identity != "未知" else ""
                    f.write(
                        f"学生 #{number}{identity_str}: 专注度={metrics['focus_rate']:.2f}%, "
                        f"抬头率={metrics['head_up_rate']:.2f}%, "
                        f"习惯={metrics['habit_label']}, "
                        f"告警次数={student['alert_count']}\n"
                    )
            else:
                f.write("当前所有学生表现均在正常范围内。\n")

            det_txt = os.path.join(save_dir, "detection_session_report.txt")
            det_png = os.path.join(save_dir, "detection_session_overview.png")
            if os.path.exists(det_txt):
                f.write("\n行为识别会话报告\n--------------------\n")
                f.write(f"文本: {os.path.basename(det_txt)}\n")
                if os.path.exists(det_png):
                    f.write(f"图表: {os.path.basename(det_png)}\n")

    def set_total_students(self, total):
        self.total_students = total

    def reset_runtime_state(self):
        self.attention_logs.clear()
        self.student_states.clear()
        self.id_mapping = {}
        self.next_id = 0
        self.current_count = 0
        self.detected_objects = []
        self.latest_class_metrics = {
            "students_analyzed": 0,
            "focus_rate": 0.0,
            "head_up_rate": 0.0,
            "dominant_habit": "insufficient_data"
        }
        self.frame_count = 0
        self.actual_fps = 0
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self._frame_processing_times = []
        self._cached_detected_objects = []
        self._cached_behavior_detections = []
        self._cached_face_results = ([], [], [])
        self._cached_pose_result = None
        self.session_behavior_samples.clear()
        self._finalize_report_done = False

    def _sync_target_fps_with_source(self):
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if source_fps and source_fps > 1:
            self.target_fps = min(60, max(1, round(source_fps)))
        else:
            self.target_fps = 30

    def set_video_source(self, video_source, reset_state=True):
        new_cap = cv2.VideoCapture(video_source)
        if not new_cap.isOpened():
            new_cap.release()
            return False, "无法打开所选视频源。"

        # 切换视频源时关闭上一个实时保存句柄，避免文件占用/写入混乱
        try:
            self.stop_realtime_recording()
        except Exception as e:
            print(f"切换视频源释放实时视频写入器失败: {e}")

        if self.cap is not None:
            self.cap.release()

        self.cap = new_cap
        self.video_source = video_source
        self._sync_target_fps_with_source()

        if reset_state:
            self.reset_runtime_state()

        return True, None

    def release(self):
        """释放资源"""
        # 稳妥落盘：不管如何退出，尽量释放实时视频写入句柄
        try:
            self.stop_realtime_recording()
        except Exception as e:
            print(f"释放实时视频写入器失败: {e}")
        self.cap.release()
        cv2.destroyAllWindows()


def draw_chinese_text(img, text, position, font_size=24, color=(255, 0, 0)):
    """在图像上绘制中文文本"""
    # 兼容旧接口：内部走缓存字体，避免每帧重复加载字体
    return draw_chinese_texts(img, [(text, position, font_size, color)])


class ClassroomMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("课堂教学行为分析系统")
        self.root.geometry("1420x920")
        self.root.minsize(1280, 820)
        self.bg_color = "#F5F7FB"
        self.panel_bg = "#FFFFFF"
        self.panel_fg = "#37474F"
        self.root.configure(bg=self.bg_color)

        self.total_students_var = tk.IntVar(value=30)

        self.monitor = ClassroomMonitor(total_students=self.total_students_var.get())
        atexit.register(self.monitor.close_finalize_reports)
        self._closing = False
        self.running = False

        self.video_source_mode = "camera"
        self.current_video_path = None
        self._last_video_frame_bgr = None
        self._video_progress_visible = False
        self._video_progress_dragging = False
        self._video_progress_seekable = False
        self._video_progress_total_frames = 0
        self._video_progress_fps = 0.0

        self.create_menu()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.control_frame = tk.Frame(self.root, bg=self.bg_color)
        self.control_frame.pack(fill="x", padx=18, pady=(12, 8))

        self.start_btn = tk.Button(self.control_frame, text="开始监测", command=self.start,
                                   width=12, height=2, bg="#4CAF50", fg="white", font=("黑体", 10, "bold"),
                                   relief="flat", bd=0)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.pause_btn = tk.Button(self.control_frame, text="暂停监测", command=self.pause,
                                   width=12, height=2, bg="#F44336", fg="white", font=("黑体", 10, "bold"),
                                   relief="flat", bd=0)
        self.pause_btn.grid(row=0, column=1, padx=5)

        self.open_video_btn = tk.Button(self.control_frame, text="打开视频", command=self.open_local_video,
                                        width=12, height=2, bg="#00ACC1", fg="white", font=("黑体", 10, "bold"),
                                        relief="flat", bd=0)
        self.open_video_btn.grid(row=0, column=2, padx=5)

        self.record_btn = tk.Button(self.control_frame, text="开始录制", command=self.toggle_recording,
                                    width=12, height=2, bg="#9C27B0", fg="white", font=("黑体", 10, "bold"),
                                    relief="flat", bd=0)
        self.record_btn.grid(row=0, column=3, padx=5)

        self.report_btn = tk.Button(self.control_frame, text="生成报告", command=self.generate_report,
                                    width=12, height=2, bg="#FF9800", fg="white", font=("黑体", 10, "bold"),
                                    relief="flat", bd=0)
        self.report_btn.grid(row=0, column=4, padx=5)

        # 添加人脸注册按钮
        self.register_face_btn = tk.Button(self.control_frame, text="注册人脸", command=self.register_face,
                                           width=12, height=2, bg="#E91E63", fg="white", font=("黑体", 10, "bold"),
                                           relief="flat", bd=0)
        self.register_face_btn.grid(row=0, column=5, padx=5)

        self.manage_face_btn = tk.Button(self.control_frame, text="人脸库管理", command=self.open_face_database_manager,
                                         width=12, height=2, bg="#607D8B", fg="white", font=("黑体", 10, "bold"),
                                         relief="flat", bd=0)
        self.manage_face_btn.grid(row=0, column=6, padx=5)

        self.total_frame = tk.Frame(self.control_frame, bg=self.bg_color)
        self.total_frame.grid(row=0, column=7, padx=10)

        tk.Label(self.total_frame, text="班级总人数", font=("黑体", 10), bg=self.bg_color).pack(side="left")
        self.total_entry = tk.Entry(self.total_frame, textvariable=self.total_students_var, width=5, font=("黑体", 10))
        self.total_entry.pack(side="left", padx=5)

        self.update_total_btn = tk.Button(self.total_frame, text="更新", command=self.update_total_students,
                                          bg="#2196F3", fg="white", font=("黑体", 9), relief="flat", bd=0)
        self.update_total_btn.pack(side="left")

        self.sensitivity_frame = tk.LabelFrame(
            self.root,
            text="检测参数调整",
            font=("微软雅黑", 10, "bold"),
            bg=self.panel_bg,
            fg=self.panel_fg,
            padx=10,
            pady=6
        )
        self.sensitivity_frame.pack(padx=18, pady=(0, 8), fill="x")

        tk.Label(self.sensitivity_frame, text="低头检测灵敏度:", bg=self.panel_bg).grid(row=0, column=0, padx=5, sticky="e")
        self.sensitivity_slider = tk.Scale(self.sensitivity_frame, from_=5, to=45, orient="horizontal",
                                           length=190, command=self.update_sensitivity, bg=self.panel_bg,
                                           highlightthickness=0)
        self.sensitivity_slider.set(self.monitor.head_down_threshold)
        self.sensitivity_slider.grid(row=0, column=1, padx=5, sticky="w")

        tk.Label(self.sensitivity_frame, text="分心时间阈值(秒):", bg=self.panel_bg).grid(row=0, column=2, padx=5, sticky="e")
        self.time_slider = tk.Scale(self.sensitivity_frame, from_=0.5, to=10, orient="horizontal",
                                    resolution=0.5, length=190, command=self.update_time_threshold, bg=self.panel_bg,
                                    highlightthickness=0)
        self.time_slider.set(self.monitor.time_threshold)
        self.time_slider.grid(row=0, column=3, padx=5, sticky="w")

        tk.Label(self.sensitivity_frame, text="转头角度阈值(度):", bg=self.panel_bg).grid(row=1, column=0, padx=5, sticky="e")
        self.turn_slider = tk.Scale(self.sensitivity_frame, from_=10, to=90, orient="horizontal",
                                    length=190, command=self.update_turn_threshold, bg=self.panel_bg,
                                    highlightthickness=0)
        self.turn_slider.set(self.monitor.head_turn_threshold)
        self.turn_slider.grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(self.sensitivity_frame, text="检测置信度阈值:", bg=self.panel_bg).grid(row=1, column=2, padx=5, sticky="e")
        self.conf_slider = tk.Scale(self.sensitivity_frame, from_=0.1, to=0.9, orient="horizontal",
                                    resolution=0.05, length=190, command=self.update_confidence, bg=self.panel_bg,
                                    highlightthickness=0)
        self.conf_slider.set(self.monitor.confidence_threshold)
        self.conf_slider.grid(row=1, column=3, padx=5, sticky="w")

        tk.Label(self.sensitivity_frame, text="平滑窗口大小:", bg=self.panel_bg).grid(row=2, column=0, padx=5, sticky="e")
        self.filter_slider = tk.Scale(self.sensitivity_frame, from_=1, to=10, orient="horizontal",
                                      length=190, command=self.update_filter_size, bg=self.panel_bg,
                                      highlightthickness=0)
        self.filter_slider.set(self.monitor.filter_size)
        self.filter_slider.grid(row=2, column=1, padx=5, sticky="w")

        self.options_frame = tk.Frame(self.sensitivity_frame, bg=self.panel_bg)
        self.options_frame.grid(row=2, column=2, columnspan=2, pady=5, sticky="w")

        self.debug_var = tk.BooleanVar(value=self.monitor.debug)
        self.debug_check = tk.Checkbutton(self.options_frame, text="显示调试信息",
                                          variable=self.debug_var, command=self.toggle_debug, bg=self.panel_bg)
        self.debug_check.pack(side="left", padx=(0, 10))

        self.beep_var = tk.BooleanVar(value=False)
        self.beep_check = tk.Checkbutton(self.options_frame, text="启用声音提醒",
                                         variable=self.beep_var, command=self.toggle_beep, bg=self.panel_bg)
        self.beep_check.pack(side="left", padx=10)
        self.monitor.beep_enabled = self.beep_var.get()

        self.object_var = tk.BooleanVar(value=False)
        self.object_check = tk.Checkbutton(self.options_frame, text="桌面物品检测",
                                           variable=self.object_var, command=self.toggle_object_detection, bg=self.panel_bg)
        self.object_check.pack(side="left", padx=10)
        self.monitor.object_detection_enabled = self.object_var.get()

        # 人脸识别开关
        self.face_var = tk.BooleanVar(value=False)
        self.face_check = tk.Checkbutton(self.options_frame, text="人脸识别",
                                         variable=self.face_var, command=self.toggle_face_recognition, bg=self.panel_bg)
        self.face_check.pack(side="left", padx=10)
        self.monitor.face_recognition_enabled = self.face_var.get()

        self.content_frame = tk.Frame(self.root, bg=self.bg_color)
        self.content_frame.pack(fill="both", expand=True, padx=18, pady=(0, 10))

        self.video_frame = tk.LabelFrame(
            self.content_frame,
            text="课堂画面",
            font=("微软雅黑", 10, "bold"),
            bg=self.panel_bg,
            fg=self.panel_fg,
            padx=10,
            pady=10
        )
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.status_frame = tk.LabelFrame(
            self.content_frame,
            text="实时状态",
            font=("微软雅黑", 10, "bold"),
            bg=self.panel_bg,
            fg=self.panel_fg,
            padx=12,
            pady=10,
            width=320
        )
        self.status_frame.pack(side="right", fill="y")
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_frame,
            text="系统就绪，等待开始...",
            font=("黑体", 10),
            bg=self.panel_bg,
            fg=self.panel_fg,
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.status_label.pack(fill="x", pady=(0, 8))

        self.attendance_label = tk.Label(self.status_frame,
                                         text=f"出勤率: 0/{self.total_students_var.get()} (0.0%)",
                                         font=("黑体", 13, "bold"), fg="#1976D2",
                                         bg=self.panel_bg, anchor="w", justify="left", wraplength=280)
        self.attendance_label.pack(fill="x", pady=4)
        
        self.analysis_label = tk.Label(self.status_frame,
                                       text="专注度 0.0% | 抬头率 0.0%",
                                       font=("TkDefaultFont", 10), fg="#455A64",
                                       bg=self.panel_bg, anchor="w", justify="left", wraplength=280)
        self.analysis_label.pack(fill="x", pady=2)
        
        self.habit_label = tk.Label(self.status_frame,
                                    text="主导习惯: 数据不足",
                                    font=("TkDefaultFont", 10), fg="#455A64",
                                    bg=self.panel_bg, anchor="w", justify="left", wraplength=280)
        self.habit_label.pack(fill="x", pady=2)

        self.source_label = tk.Label(self.status_frame,
                                     text="视频源: 摄像头",
                                     font=("黑体", 10), fg="#0097A7",
                                     bg=self.panel_bg, anchor="w", justify="left", wraplength=280)
        self.source_label.pack(fill="x", pady=4)

        self.record_label = tk.Label(self.status_frame,
                                     text="录制: 已关闭",
                                     font=("黑体", 10), fg="#9E9E9E",
                                     bg=self.panel_bg, anchor="w", justify="left", wraplength=280)
        self.record_label.pack(fill="x", pady=4)

        self.object_label = tk.Label(self.status_frame,
                                     text="桌面物品: 未检测",
                                     font=("黑体", 10), fg="#FF9800",
                                     bg=self.panel_bg, anchor="w", justify="left", wraplength=280)
        self.object_label.pack(fill="x", pady=4)

        # 人脸识别状态
        self.face_label = tk.Label(self.status_frame,
                                   text="人脸识别: 已关闭",
                                   font=("黑体", 10), fg="#9C27B0",
                                   bg=self.panel_bg, anchor="w", justify="left", wraplength=280)
        self.face_label.pack(fill="x", pady=4)

        self.canvas = tk.Canvas(self.video_frame, bg="#101418", highlightthickness=0, width=980, height=640)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas_image_id = None

        self.video_progress_var = tk.DoubleVar(value=0)
        self.video_progress_frame = tk.Frame(self.video_frame, bg=self.panel_bg)
        self.video_progress_scale = tk.Scale(
            self.video_progress_frame,
            from_=0,
            to=1,
            orient="horizontal",
            variable=self.video_progress_var,
            showvalue=False,
            resolution=1,
            bg=self.panel_bg,
            highlightthickness=0,
            state="disabled",
            command=self._on_video_progress_drag
        )
        self.video_progress_scale.pack(side="left", fill="x", expand=True)
        self.video_progress_scale.bind("<ButtonPress-1>", self._on_video_progress_press)
        self.video_progress_scale.bind("<ButtonRelease-1>", self._on_video_progress_release)
        self.video_time_label = tk.Label(
            self.video_progress_frame,
            text="",
            font=("TkDefaultFont", 10),
            fg="#455A64",
            bg=self.panel_bg,
            width=26,
            anchor="e"
        )
        self.video_time_label.pack(side="right", padx=(10, 0))

        self.footer_frame = tk.Frame(self.root, bg=self.bg_color, height=24)
        self.footer_frame.pack(fill="x", side="bottom")
        self.footer_label = tk.Label(self.footer_frame, text="开发者: zkm202105100625",
                                     bg=self.bg_color, fg="#7A869A")
        self.footer_label.pack(side="right", padx=10)

        self._refresh_face_status()
        self._refresh_recording_status()
        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def update_total_students(self):
        """更新总学生数"""
        try:
            total = int(self.total_students_var.get())
            if total <= 0:
                raise ValueError("人数必须大于 0")

            self.monitor.set_total_students(total)
            self.attendance_label.config(
                text=f"出勤率: {self.monitor.current_count}/{total} ({self.monitor.current_count / total * 100:.1f}%)")
            self.status_label.config(text=f"班级总人数已更新为: {total}")
        except ValueError as e:
            self.status_label.config(text=f"错误: {str(e)}")
            self.total_students_var.set(self.monitor.total_students)

    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)

        self.file_menu = tk.Menu(menubar, tearoff=0)
        self.file_menu.add_command(label="开始监测", command=self.start)
        self.file_menu.add_command(label="暂停监测", command=self.pause)
        self.file_menu.add_command(label="打开本地视频", command=self.open_local_video)
        self.file_menu.add_command(label="切换到摄像头", command=self.use_camera_source)
        self.file_menu.add_command(label="开始录制", command=self.toggle_recording)
        self.record_menu_index = self.file_menu.index("end")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="生成报告", command=self.generate_report)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.on_closing)
        menubar.add_cascade(label="文件", menu=self.file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="恢复默认设置", command=self.reset_settings)
        settings_menu.add_command(label="保存当前设置", command=self.save_settings)
        settings_menu.add_command(label="加载设置", command=self.load_settings)
        menubar.add_cascade(label="设置", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.config(menu=menubar)

    def _refresh_recording_status(self):
        recording = bool(self.monitor.realtime_save_enabled)
        if recording:
            label = "录制: 已开启，等待开始"
            if self.monitor.realtime_video_path:
                label = f"录制: 进行中 ({os.path.basename(self.monitor.realtime_video_path)})"
            self.record_label.config(text=label, fg="#4CAF50")
            self.record_btn.config(text="停止录制", bg="#D32F2F")
            if hasattr(self, "file_menu") and hasattr(self, "record_menu_index"):
                self.file_menu.entryconfig(self.record_menu_index, label="停止录制")
        else:
            self.record_label.config(text="录制: 已关闭", fg="#9E9E9E")
            self.record_btn.config(text="开始录制", bg="#9C27B0")
            if hasattr(self, "file_menu") and hasattr(self, "record_menu_index"):
                self.file_menu.entryconfig(self.record_menu_index, label="开始录制")

    def _refresh_run_controls(self):
        """根据当前视频源和运行状态刷新开始/暂停控件文案。"""
        is_local_video = self.video_source_mode == "file"
        start_text = "继续播放" if is_local_video else "开始监测"
        pause_text = "暂停播放" if is_local_video else "暂停监测"

        self.start_btn.config(text=start_text)
        self.pause_btn.config(text=pause_text)

        if hasattr(self, "file_menu"):
            self.file_menu.entryconfig(0, label=start_text)
            self.file_menu.entryconfig(1, label=pause_text)

    def start(self):
        """开始监测"""
        self.running = True
        self.status_label.config(text="本地视频播放中..." if self.video_source_mode == "file" else "系统运行中...")
        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self._refresh_run_controls()

    def pause(self):
        """暂停监测"""
        self.running = False
        self.status_label.config(text="视频播放已暂停。" if self.video_source_mode == "file" else "监测已暂停。")
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self._refresh_run_controls()
        if self.video_source_mode == "file":
            self._refresh_paused_local_video_frame()

    def _refresh_paused_local_video_frame(self):
        """暂停本地视频时，对当前静止帧做一次行为识别预览。"""
        if self.video_source_mode != "file":
            return
        raw = self._last_video_frame_bgr
        if raw is None:
            return
        try:
            processed_frame, used_behavior_model = self.monitor.process_local_behavior_preview(raw.copy())
            self.current_frame = processed_frame.copy()
            self._render_frame_on_canvas(processed_frame)
            if used_behavior_model:
                self.status_label.config(text="视频已暂停，已对当前画面完成行为识别预览。")
            else:
                self.status_label.config(text="视频已暂停，未加载到行为模型，保持抬头/低头预览。")
        except Exception as e:
            self.status_label.config(text=f"暂停帧行为识别预览失败: {e}")

    def _update_source_label(self):
        """刷新当前视频源显示。"""
        if self.video_source_mode == "file" and self.current_video_path:
            self.source_label.config(text=f"视频源: 本地视频 - {os.path.basename(self.current_video_path)}")
        else:
            self.source_label.config(text="视频源: 摄像头")

    def _reset_runtime_panels(self):
        """切换视频源后重置界面统计显示。"""
        attendance_rate = (self.monitor.current_count / self.monitor.total_students * 100
                           if self.monitor.total_students > 0 else 0)
        self.attendance_label.config(
            text=f"出勤率: {self.monitor.current_count}/{self.monitor.total_students} ({attendance_rate:.1f}%)")
        self.analysis_label.config(text="专注度 0.0% | 抬头率 0.0%")
        self.habit_label.config(text="主导习惯: 数据不足")
        if self.video_source_mode == "file" and self.object_var.get():
            self.object_label.config(text="桌面物品检测: 本地视频模式已停用", fg="#FF9800")
        elif self.object_var.get():
            self.object_label.config(text="桌面物品: 未检测", fg="#FF9800")
        else:
            self.object_label.config(text="桌面物品检测: 已关闭", fg="#9E9E9E")
        self._refresh_face_status()
        self._refresh_recording_status()
        self.canvas.delete("all")
        self._canvas_image_id = None
        self.canvas.imgtk = None
        self._last_video_frame_bgr = None
        if hasattr(self, "current_frame"):
            del self.current_frame

    def _switch_video_source(self, video_source, source_mode, status_text, video_path=None):
        """切换到新的摄像头或本地视频源。"""
        self.running = False
        self.pause_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self._refresh_run_controls()

        success, error = self.monitor.set_video_source(video_source)
        if not success:
            self.status_label.config(text=error)
            return False

        # 本地视频上传检测：只做抬头/低头，禁用其它姿态/行为
        if source_mode == "file":
            self.monitor.simple_pose_only = True
            self.monitor.object_detection_enabled = False
            self.monitor.face_recognition_enabled = False
            self.monitor.behavior_model_enabled = False
        else:
            self.monitor.simple_pose_only = False
            # 恢复 UI 开关状态（行为模型按是否成功加载决定）
            self.monitor.object_detection_enabled = bool(self.object_var.get()) if hasattr(self, "object_var") else self.monitor.object_detection_enabled
            self.monitor.face_recognition_enabled = bool(self.face_var.get()) if hasattr(self, "face_var") else self.monitor.face_recognition_enabled
            self.monitor.behavior_model_enabled = os.path.exists(self.monitor.behavior_model_path)

        self.video_source_mode = source_mode
        self.current_video_path = video_path if source_mode == "file" else None
        self._update_source_label()
        self._reset_runtime_panels()
        self._show_or_hide_video_progress()
        self.start()
        self.status_label.config(text=status_text)
        return True

    def open_local_video(self):
        """选择并开始分析本地视频。"""
        video_path = filedialog.askopenfilename(
            title="选择本地视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("所有文件", "*.*")
            ]
        )
        if not video_path:
            return

        self._switch_video_source(
            video_path,
            "file",
            f"已加载本地视频: {os.path.basename(video_path)}",
            video_path=video_path
        )

    def use_camera_source(self):
        """切换回默认摄像头。"""
        self._switch_video_source(0, "camera", "已切换到摄像头实时监测。")

    def _safe_cap_get(self, prop_id):
        try:
            value = self.monitor.cap.get(prop_id)
            if value is None or not np.isfinite(value):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def _format_video_time(self, seconds):
        seconds = max(0, int(round(seconds)))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _refresh_video_progress_metadata(self):
        if self.video_source_mode != "file":
            self._video_progress_seekable = False
            self._video_progress_total_frames = 0
            self._video_progress_fps = 0.0
            return

        total_frames = self._safe_cap_get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self._safe_cap_get(cv2.CAP_PROP_FPS)
        self._video_progress_total_frames = int(total_frames) if total_frames > 1 else 0
        self._video_progress_fps = fps if fps > 0.01 else 0.0
        self._video_progress_seekable = self._video_progress_total_frames > 1 and self._video_progress_fps > 0.01

        if self._video_progress_seekable:
            self.video_progress_scale.config(
                from_=0,
                to=max(1, self._video_progress_total_frames - 1),
                state="normal"
            )
        else:
            self.video_progress_scale.config(from_=0, to=1, state="disabled")
            self.video_progress_var.set(0)

    def _show_or_hide_video_progress(self):
        if not hasattr(self, "video_progress_frame"):
            return

        if self.video_source_mode == "file":
            if not self._video_progress_visible:
                self.video_progress_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
                self._video_progress_visible = True
            self._video_progress_dragging = False
            self._refresh_video_progress_metadata()
            self._update_video_progress()
        else:
            self._video_progress_dragging = False
            self._video_progress_seekable = False
            self._video_progress_total_frames = 0
            self._video_progress_fps = 0.0
            self.video_progress_var.set(0)
            self.video_time_label.config(text="")
            self.video_progress_scale.config(from_=0, to=1, state="disabled")
            if self._video_progress_visible:
                self.video_progress_frame.grid_remove()
                self._video_progress_visible = False

    def _video_elapsed_seconds(self, frame_value=None):
        pos_msec = self._safe_cap_get(cv2.CAP_PROP_POS_MSEC)
        if frame_value is None:
            frame_value = self._safe_cap_get(cv2.CAP_PROP_POS_FRAMES)

        if pos_msec > 0 and frame_value is None:
            return pos_msec / 1000
        if self._video_progress_fps > 0.01:
            return max(0.0, float(frame_value) / self._video_progress_fps)
        if pos_msec > 0:
            return pos_msec / 1000
        return 0.0

    def _set_video_progress_label(self, frame_value=None, elapsed_seconds=None, force_end=False):
        if self._video_progress_seekable:
            total_frames = max(1, self._video_progress_total_frames)
            fps = self._video_progress_fps
            if force_end:
                frame_value = total_frames
            elif frame_value is None:
                frame_value = self._safe_cap_get(cv2.CAP_PROP_POS_FRAMES)

            frame_value = max(0.0, min(float(frame_value), float(total_frames)))
            elapsed = frame_value / fps
            total_seconds = total_frames / fps
            percent = 100.0 if force_end else min(100.0, max(0.0, frame_value / total_frames * 100))
            self.video_time_label.config(
                text=f"{self._format_video_time(elapsed)} / {self._format_video_time(total_seconds)} ({percent:.1f}%)"
            )
        else:
            if elapsed_seconds is None:
                elapsed_seconds = self._video_elapsed_seconds(frame_value)
            self.video_time_label.config(text=f"已播放 {self._format_video_time(elapsed_seconds)}")

    def _update_video_progress(self, force_end=False):
        if self.video_source_mode != "file" or not hasattr(self, "video_progress_scale"):
            return
        if self._video_progress_dragging and not force_end:
            return

        self._refresh_video_progress_metadata()
        if self._video_progress_seekable:
            if force_end:
                slider_frame = max(0, self._video_progress_total_frames - 1)
                self.video_progress_var.set(slider_frame)
                self._set_video_progress_label(force_end=True)
            else:
                pos_frames = self._safe_cap_get(cv2.CAP_PROP_POS_FRAMES)
                slider_frame = max(0, min(int(round(pos_frames)), self._video_progress_total_frames - 1))
                self.video_progress_var.set(slider_frame)
                self._set_video_progress_label(pos_frames)
        else:
            elapsed_seconds = self._video_elapsed_seconds()
            self._set_video_progress_label(elapsed_seconds=elapsed_seconds)

    def _on_video_progress_press(self, event):
        if self.video_source_mode != "file":
            return
        self._refresh_video_progress_metadata()
        if self._video_progress_seekable:
            self._video_progress_dragging = True

    def _on_video_progress_drag(self, value):
        if not self._video_progress_dragging or not self._video_progress_seekable:
            return
        try:
            frame_value = float(value)
        except (TypeError, ValueError):
            return
        self._set_video_progress_label(frame_value)

    def _reset_panels_after_video_seek(self):
        attendance_rate = (self.monitor.current_count / self.monitor.total_students * 100
                           if self.monitor.total_students > 0 else 0)
        self.attendance_label.config(
            text=f"出勤率: {self.monitor.current_count}/{self.monitor.total_students} ({attendance_rate:.1f}%)")
        self.analysis_label.config(text="专注度 0.0% | 抬头率 0.0%")
        self.habit_label.config(text="主导习惯: 数据不足")
        self.object_label.config(text="桌面物品检测: 本地视频模式已停用", fg="#FF9800")
        self._refresh_face_status()
        self.canvas.delete("all")
        self._canvas_image_id = None
        self.canvas.imgtk = None
        if hasattr(self, "current_frame"):
            del self.current_frame

    def _seek_local_video(self, target_frame):
        if self.video_source_mode != "file":
            return

        self._refresh_video_progress_metadata()
        if not self._video_progress_seekable:
            self._update_video_progress()
            return

        target_frame = max(0, min(int(round(target_frame)), self._video_progress_total_frames - 1))
        was_running = self.running
        try:
            if not self.monitor.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame):
                raise RuntimeError("视频源不支持精确跳转")
        except Exception as e:
            self.status_label.config(text=f"视频跳转失败: {e}")
            self._update_video_progress()
            return

        self.monitor.reset_runtime_state()
        self._reset_panels_after_video_seek()

        if not was_running:
            try:
                success, preview_frame = self.monitor.cap.read()
                if success:
                    self._last_video_frame_bgr = preview_frame.copy()
                    self._render_frame_on_canvas(preview_frame)
                    self.monitor.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            except Exception as e:
                print(f"跳转预览帧失败: {e}")

        self.video_progress_var.set(target_frame)
        self._set_video_progress_label(target_frame)
        self.status_label.config(text=f"已跳转到 {self._format_video_time(target_frame / self._video_progress_fps)}，本次视频统计已重置。")

    def _on_video_progress_release(self, event):
        if not self._video_progress_dragging:
            return
        self._video_progress_dragging = False
        if self.video_source_mode != "file" or not self._video_progress_seekable:
            self._update_video_progress()
            return
        self._seek_local_video(self.video_progress_var.get())

    def _render_frame_on_canvas(self, frame):
        """将处理后的帧按比例绘制到主画布。"""
        h, w = frame.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 10 or canvas_h <= 10:
            return

        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0:
            return

        # 性能优化：使用 INTER_AREA 更适合缩小；并避免每帧 delete 重建 Canvas 对象
        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        x_pos = (canvas_w - new_w) // 2
        y_pos = (canvas_h - new_h) // 2
        if not hasattr(self, "_canvas_image_id") or self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(x_pos, y_pos, anchor="nw", image=imgtk)
        else:
            try:
                self.canvas.coords(self._canvas_image_id, x_pos, y_pos)
                self.canvas.itemconfig(self._canvas_image_id, image=imgtk)
            except tk.TclError:
                self._canvas_image_id = self.canvas.create_image(x_pos, y_pos, anchor="nw", image=imgtk)
        self.canvas.imgtk = imgtk

    def update_sensitivity(self, value):
        """更新低头检测灵敏度"""
        self.monitor.head_down_threshold = float(value)

    def update_time_threshold(self, value):
        """更新分心时间阈值"""
        self.monitor.time_threshold = float(value)

    def update_turn_threshold(self, value):
        """更新转头角度阈值"""
        self.monitor.head_turn_threshold = float(value)

    def update_confidence(self, value):
        """更新置信度阈值"""
        self.monitor.confidence_threshold = float(value)

    def update_filter_size(self, value):
        """更新平滑窗口大小"""
        self.monitor.filter_size = int(float(value))

    def toggle_debug(self):
        """切换调试模式"""
        self.monitor.debug = self.debug_var.get()

    def toggle_beep(self):
        """切换声音提醒"""
        self.monitor.beep_enabled = self.beep_var.get()

    def toggle_object_detection(self):
        """切换桌面物品检测"""
        self.monitor.object_detection_enabled = self.object_var.get()
        if self.object_var.get():
            if self.video_source_mode == "file":
                self.monitor.object_detection_enabled = False
                self.object_label.config(text="桌面物品检测: 切回摄像头后生效", fg="#FF9800")
                self.status_label.config(text="本地视频模式不启用桌面物品检测，切回摄像头后生效。")
                return
            if self.video_source_mode != "file":
                ok, message = self.monitor.ensure_object_model_loaded()
                if not ok:
                    self.object_var.set(False)
                    self.monitor.object_detection_enabled = False
                    self.object_label.config(text="桌面物品检测: 加载失败", fg="#F44336")
                    self.status_label.config(text=message)
                    return
            self.object_label.config(text="桌面物品检测: 已开启", fg="#4CAF50")
        else:
            self.object_label.config(text="桌面物品检测: 已关闭", fg="#9E9E9E")

    def toggle_face_recognition(self):
        """切换人脸识别"""
        self.monitor.face_recognition_enabled = self.face_var.get()
        if self.face_var.get() and self.video_source_mode == "file":
            self.monitor.face_recognition_enabled = False
            self.status_label.config(text="本地视频模式不启用人脸识别，切回摄像头后生效。")
            self.face_label.config(text="人脸识别: 切回摄像头后生效", fg="#FF9800")
            return
        if self.face_var.get() and self.video_source_mode != "file":
            ok, message = self.monitor.ensure_face_models_loaded()
            if not ok:
                self.face_var.set(False)
                self.monitor.face_recognition_enabled = False
                self.status_label.config(text=message)
        self._refresh_face_status()

    def _refresh_face_status(self):
        """刷新人脸识别状态文本。"""
        if self.face_var.get() and self.video_source_mode == "file":
            self.face_label.config(text="人脸识别: 本地视频模式已停用", fg="#FF9800")
            return
        if self.face_var.get():
            self.face_label.config(text="人脸识别: 已开启", fg="#4CAF50")
        else:
            self.face_label.config(text="人脸识别: 已关闭", fg="#9E9E9E")

    def register_face(self):
        """注册新人脸"""
        if not hasattr(self, 'current_frame'):
            self.status_label.config(text="请先开始监测，然后点击注册人脸")
            return
            
        was_running = self.running
        if was_running:
            self.pause()
        
        # 创建注册窗口
        register_window = tk.Toplevel(self.root)
        register_window.title("注册人脸")
        register_window.geometry("400x200")
        
        tk.Label(register_window, text="请输入学生姓名:", font=("黑体", 12)).pack(pady=20)
        
        name_var = tk.StringVar()
        name_entry = tk.Entry(register_window, textvariable=name_var, width=30, font=("黑体", 12))
        name_entry.pack(pady=10)
        name_entry.focus()
        
        def restore_run():
            if was_running: self.start()
            register_window.destroy()
            
        register_window.protocol("WM_DELETE_WINDOW", restore_run)
        
        def do_register():
            name = name_var.get().strip()
            if not name:
                self.status_label.config(text="请输入有效的姓名")
                return
            
            # 从当前帧中提取人脸
            success = self.monitor.add_face_to_database(self.current_frame, name)
            if success:
                self.status_label.config(text=f"成功注册人脸：{name}")
                restore_run()
            else:
                self.status_label.config(text=f"注册失败：未检测到人脸或无法提取特征")
        
        tk.Button(register_window, text="确认注册", command=do_register,
                  bg="#4CAF50", fg="white", font=("黑体", 12)).pack(pady=20)

    def open_face_database_manager(self):
        """打开人脸库管理窗口。"""
        if hasattr(self, "face_db_window") and self.face_db_window.winfo_exists():
            self.face_db_window.lift()
            self.face_db_window.focus_force()
            self._refresh_face_database_list()
            return

        self.face_db_window = tk.Toplevel(self.root)
        self.face_db_window.title("人脸库管理")
        self.face_db_window.geometry("420x420")

        header_frame = tk.Frame(self.face_db_window)
        header_frame.pack(fill="x", padx=12, pady=(12, 6))

        self.face_db_count_label = tk.Label(header_frame, text="已注册人数: 0", font=("黑体", 11, "bold"))
        self.face_db_count_label.pack(side="left")

        refresh_btn = tk.Button(header_frame, text="刷新", command=self._refresh_face_database_list,
                                bg="#2196F3", fg="white", font=("黑体", 10))
        refresh_btn.pack(side="right")

        list_frame = tk.Frame(self.face_db_window)
        list_frame.pack(fill="both", expand=True, padx=12, pady=6)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.face_db_listbox = tk.Listbox(list_frame, font=("微软雅黑", 11), activestyle="dotbox")
        self.face_db_listbox.pack(side="left", fill="both", expand=True)
        self.face_db_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.face_db_listbox.yview)

        action_frame = tk.Frame(self.face_db_window)
        action_frame.pack(fill="x", padx=12, pady=8)

        rename_btn = tk.Button(action_frame, text="重命名", command=self._rename_selected_face,
                               bg="#FF9800", fg="white", font=("黑体", 10), width=10)
        rename_btn.pack(side="left", padx=4)

        delete_btn = tk.Button(action_frame, text="删除", command=self._delete_selected_face,
                               bg="#F44336", fg="white", font=("黑体", 10), width=10)
        delete_btn.pack(side="left", padx=4)

        close_btn = tk.Button(action_frame, text="关闭", command=self.face_db_window.destroy,
                              bg="#9E9E9E", fg="white", font=("黑体", 10), width=10)
        close_btn.pack(side="right", padx=4)

        self.face_db_status_label = tk.Label(self.face_db_window, text="请选择一个已注册身份进行管理。",
                                             font=("微软雅黑", 10), fg="#455A64", anchor="w")
        self.face_db_status_label.pack(fill="x", padx=12, pady=(0, 12))

        self._refresh_face_database_list()

    def _refresh_face_database_list(self):
        """刷新人脸库列表。"""
        if not hasattr(self, "face_db_listbox") or not self.face_db_listbox.winfo_exists():
            return

        self.face_db_entries = self.monitor.get_face_database_entries()
        self.face_db_listbox.delete(0, tk.END)

        for entry in self.face_db_entries:
            self.face_db_listbox.insert(tk.END, f"{entry['index'] + 1}. {entry['name']}")

        count = len(self.face_db_entries)
        self.face_db_count_label.config(text=f"已注册人数: {count}")
        if count == 0:
            self.face_db_status_label.config(text="当前人脸库为空。")
        else:
            self.face_db_status_label.config(text="请选择一个已注册身份进行管理。")

    def _get_selected_face_entry(self):
        """获取当前选中的人脸库条目。"""
        if not hasattr(self, "face_db_listbox") or not self.face_db_listbox.curselection():
            messagebox.showwarning("未选择身份", "请先在列表中选择一个身份。")
            return None

        selection = self.face_db_listbox.curselection()[0]
        if selection >= len(getattr(self, "face_db_entries", [])):
            messagebox.showwarning("选择无效", "当前选中项无效，请刷新后重试。")
            return None
        return self.face_db_entries[selection]

    def _delete_selected_face(self):
        """删除选中的人脸库身份。"""
        entry = self._get_selected_face_entry()
        if entry is None:
            return

        confirmed = messagebox.askyesno(
            "确认删除",
            f"确定要删除身份“{entry['name']}”吗？此操作会立即写入人脸库文件。"
        )
        if not confirmed:
            return

        success, result = self.monitor.delete_face_from_database(entry["index"])
        if success:
            self._refresh_face_database_list()
            self.face_db_status_label.config(text=f"已删除身份：{result}")
            self.status_label.config(text=f"人脸库已删除：{result}")
        else:
            messagebox.showerror("删除失败", result)
            self.face_db_status_label.config(text=result)

    def _rename_selected_face(self):
        """重命名选中的人脸库身份。"""
        entry = self._get_selected_face_entry()
        if entry is None:
            return

        new_name = simpledialog.askstring(
            "重命名身份",
            f"请输入“{entry['name']}”的新姓名：",
            initialvalue=entry["name"],
            parent=self.face_db_window
        )
        if new_name is None:
            return

        normalized_name = new_name.strip()
        if not normalized_name:
            messagebox.showwarning("姓名无效", "姓名不能为空。")
            return

        duplicate_exists = any(
            item["name"] == normalized_name and item["index"] != entry["index"]
            for item in self.monitor.get_face_database_entries()
        )
        if duplicate_exists:
            confirmed = messagebox.askyesno(
                "存在同名身份",
                f"人脸库中已存在“{normalized_name}”。是否仍然继续重命名？"
            )
            if not confirmed:
                return

        success, result = self.monitor.rename_face_in_database(entry["index"], normalized_name)
        if success:
            self._refresh_face_database_list()
            self.face_db_status_label.config(text=f"已重命名身份：{result} -> {normalized_name}")
            self.status_label.config(text=f"人脸库已重命名：{result} -> {normalized_name}")
        else:
            messagebox.showerror("重命名失败", result)
            self.face_db_status_label.config(text=result)

    def take_snapshot(self):
        """保存当前画面截图"""
        self.status_label.config(text="截图功能未提供。")

    def toggle_recording(self):
        """切换录像状态"""
        enabled = not self.monitor.realtime_save_enabled
        final_path = self.monitor.set_recording_enabled(enabled)
        self._refresh_recording_status()
        if enabled:
            self.status_label.config(text="录制已开启，下一帧开始保存检测视频。")
        else:
            if final_path:
                self.status_label.config(text=f"录制已停止，文件已保存: {final_path}")
            else:
                self.status_label.config(text="录制已关闭。")

    def generate_report(self):
        """生成报告"""
        report_thread = threading.Thread(target=self._generate_report_thread)
        report_thread.daemon = True
        report_thread.start()
        self.status_label.config(text="正在生成报告，请稍候...")

    def _generate_report_thread(self):
        """在工作线程中生成报告"""
        # 新版报告生成（替代旧版学习/行为报告）
        generate_new_classroom_report(self.monitor, save_dir="attention_logs")
        self.root.after(0, self._report_done)

    def _report_done(self):
        """报告生成完成后的回调"""
        self.status_label.config(text="报告已生成，请查看 attention_logs 文件夹。")

        report_path = os.path.join("attention_logs", "classroom_report.txt")
        if os.path.exists(report_path):
            self.show_report(report_path)

    def show_report(self, report_path):
        """显示报告内容"""
        report_window = tk.Toplevel(self.root)
        report_window.title("课堂分析报告")
        report_window.geometry("600x500")

        report_text = tk.Text(report_window, wrap=tk.WORD, font=("微软雅黑", 11))
        report_text.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(report_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        report_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=report_text.yview)

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
                report_text.insert(tk.END, report_content)
                report_text.config(state="disabled")
        except Exception as e:
            report_text.insert(tk.END, f"无法读取报告: {str(e)}")

    def reset_settings(self):
        """重置所有设置为默认值"""
        self.sensitivity_slider.set(18)
        self.time_slider.set(2.0)
        self.turn_slider.set(35)
        self.conf_slider.set(0.45)
        self.filter_slider.set(3)
        self.debug_var.set(True)
        self.beep_var.set(False)
        self.object_var.set(False)
        self.face_var.set(False)
        self.total_students_var.set(30)

        self.monitor.head_down_threshold = 18
        self.monitor.time_threshold = 2.0
        self.monitor.head_turn_threshold = 35
        self.monitor.confidence_threshold = 0.45
        self.monitor.filter_size = 3
        self.monitor.debug = True
        self.monitor.beep_enabled = False
        self.monitor.object_detection_enabled = False
        self.monitor.face_recognition_enabled = False
        self.monitor.set_recording_enabled(False)
        self.monitor.set_total_students(30)

        attendance_rate = self.monitor.current_count / self.monitor.total_students * 100
        self.attendance_label.config(
            text=f"出勤率: {self.monitor.current_count}/{self.monitor.total_students} ({attendance_rate:.1f}%)")

        self.analysis_label.config(text="专注度 0.0% | 抬头率 0.0%")
        self.habit_label.config(text="主导习惯: 数据不足")
        self.object_label.config(text="桌面物品检测: 已关闭", fg="#9E9E9E")
        self._refresh_face_status()
        self._refresh_recording_status()
        self.status_label.config(text="所有设置已恢复默认值。")

    def save_settings(self):
        """保存当前设置到文件"""
        settings = {
            "head_down_threshold": self.monitor.head_down_threshold,
            "time_threshold": self.monitor.time_threshold,
            "head_turn_threshold": self.monitor.head_turn_threshold,
            "confidence_threshold": self.monitor.confidence_threshold,
            "filter_size": self.monitor.filter_size,
            "debug": self.monitor.debug,
            "beep_enabled": self.beep_var.get(),
            "object_detection_enabled": self.object_var.get(),
            "face_recognition_enabled": self.face_var.get(),
            "recording_enabled": self.monitor.realtime_save_enabled,
            "total_students": self.monitor.total_students
        }

        try:
            os.makedirs("settings", exist_ok=True)
            with open("settings/config.txt", "w") as f:
                for key, value in settings.items():
                    f.write(f"{key}={value}\n")
            self.status_label.config(text="设置已保存。")
        except Exception as e:
            self.status_label.config(text=f"保存设置失败: {str(e)}")

    def load_settings(self):
        """从文件加载设置"""
        try:
            if not os.path.exists("settings/config.txt"):
                self.status_label.config(text="未找到已保存的设置文件。")
                return

            settings = {}
            with open("settings/config.txt", "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        if value.lower() == "true":
                            settings[key] = True
                        elif value.lower() == "false":
                            settings[key] = False
                        else:
                            try:
                                settings[key] = float(value)
                            except:
                                settings[key] = value

            if "head_down_threshold" in settings:
                self.sensitivity_slider.set(settings["head_down_threshold"])
                self.monitor.head_down_threshold = settings["head_down_threshold"]

            if "time_threshold" in settings:
                self.time_slider.set(settings["time_threshold"])
                self.monitor.time_threshold = settings["time_threshold"]

            if "head_turn_threshold" in settings:
                self.turn_slider.set(settings["head_turn_threshold"])
                self.monitor.head_turn_threshold = settings["head_turn_threshold"]

            if "confidence_threshold" in settings:
                self.conf_slider.set(settings["confidence_threshold"])
                self.monitor.confidence_threshold = settings["confidence_threshold"]

            if "filter_size" in settings:
                self.filter_slider.set(settings["filter_size"])
                self.monitor.filter_size = int(settings["filter_size"])

            if "debug" in settings:
                self.debug_var.set(settings["debug"])
                self.monitor.debug = settings["debug"]

            if "beep_enabled" in settings:
                self.beep_var.set(settings["beep_enabled"])
                self.monitor.beep_enabled = settings["beep_enabled"]

            if "object_detection_enabled" in settings:
                self.object_var.set(settings["object_detection_enabled"])
                self.monitor.object_detection_enabled = settings["object_detection_enabled"]
                if settings["object_detection_enabled"]:
                    self.object_label.config(text="桌面物品检测: 已开启", fg="#4CAF50")
                else:
                    self.object_label.config(text="桌面物品检测: 已关闭", fg="#9E9E9E")

            if "face_recognition_enabled" in settings:
                self.face_var.set(settings["face_recognition_enabled"])
                self.monitor.face_recognition_enabled = settings["face_recognition_enabled"]
                self._refresh_face_status()

            if "recording_enabled" in settings:
                self.monitor.set_recording_enabled(settings["recording_enabled"])
                self._refresh_recording_status()

            if "total_students" in settings:
                self.total_students_var.set(int(settings["total_students"]))
                self.monitor.set_total_students(int(settings["total_students"]))

            attendance_rate = self.monitor.current_count / self.monitor.total_students * 100 if self.monitor.total_students > 0 else 0
            self.attendance_label.config(
                text=f"出勤率: {self.monitor.current_count}/{self.monitor.total_students} ({attendance_rate:.1f}%)")

            self._refresh_face_status()
            self.status_label.config(text="设置已加载。")
        except Exception as e:
            self.status_label.config(text=f"加载设置失败: {str(e)}")

    def show_help(self):
        """显示帮助信息"""
        help_window = tk.Toplevel(self.root)
        help_window.title("使用说明")
        help_window.geometry("600x500")

        help_text = tk.Text(help_window, wrap=tk.WORD, font=("微软雅黑", 11))
        help_text.pack(fill="both", expand=True, padx=10, pady=10)

        help_content = """课堂教学行为分析系统使用说明

1. 基本操作
   - 点击"开始监测"启动实时检测
   - 点击"暂停监测"暂停当前分析
   - 点击"打开视频"可加载本地视频进行分析
   - 点击"开始录制"可保存带检测叠加的画面
   - 再次点击可停止录制，文件保存在 realtime_videos 文件夹

2. 人脸识别功能
   - 点击"注册人脸"添加学生人脸
   - 点击"人脸库管理"可查看、重命名或删除已注册身份
   - 系统会自动识别已注册的学生
   - 在"检测参数调整"区域可开启/关闭此功能
   - 人脸模型会在首次启用时按需加载

3. 桌面物品检测
   - 系统可自动检测手机、书籍、笔记本电脑等物品
   - 在"检测参数调整"区域可开启/关闭此功能
   - 检测到的物品会用橙色框标注
   - 物品模型会在首次启用时按需加载

4. 本地视频说明
   - 本地视频模式只做抬头/低头分析
   - 本地视频模式不会启用桌面物品检测、人脸识别和行为模型

5. 参数说明
   - 低头检测灵敏度：值越小越敏感（推荐15-20）
   - 分心时间阈值：持续分心多久后触发提醒
   - 转头角度阈值：值越小越容易判定为转头（推荐30-40）
   - 检测置信度阈值：值越高越严格（推荐0.4-0.5）
   - 平滑窗口大小：值越大越稳定，但会增加延迟

6. 统计指标
   - 出勤率：检测到的学生人数/总人数
   - 专注度：学生专注时间占比
   - 抬头率：学生抬头时间占比
   - 听课习惯偏好：稳定听讲型、前倾低头型、侧向关注型、易分心型

7. 报告输出
   - 文本报告保存在 attention_logs/classroom_report.txt
   - 班级专注率趋势图保存在 attention_logs/class_focus_timeline.png
   - 抬头/低头趋势图保存在 attention_logs/class_headpose_timeline.png
   - 专注分布饼图保存在 attention_logs/class_focus_pie.png
   - 学生排行图保存在 attention_logs/student_ranking.png
   - 告警时间线图保存在 attention_logs/warning_timeline.png
"""
        help_text.insert(tk.END, help_content)
        help_text.config(state="disabled")

    def show_about(self):
        """显示关于信息"""
        about_window = tk.Toplevel(self.root)
        about_window.title("关于")
        about_window.geometry("400x300")

        about_frame = tk.Frame(about_window, padx=20, pady=20)
        about_frame.pack(fill="both", expand=True)

        title_label = tk.Label(about_frame, text="课堂教学行为分析系统", font=("TkDefaultFont", 16, "bold"))
        title_label.pack(pady=10)

        version_label = tk.Label(about_frame, text="版本 3.0 (人脸识别版)", font=("微软雅黑", 10))
        version_label.pack()

        desc_label = tk.Label(about_frame, text="基于 YOLOv8 姿态检测和人脸识别的\n课堂行为分析系统",
                              font=("微软雅黑", 11), justify="center")
        desc_label.pack(pady=10)

        feature_label = tk.Label(about_frame,
                                 text="支持人脸识别\n支持低头/转头检测\n支持本地视频分析\n支持专注度、抬头率统计\n支持桌面物品检测\n支持按需录制检测结果",
                                 font=("微软雅黑", 10), justify="left")
        feature_label.pack(pady=10)

        copyright_label = tk.Label(about_frame, text="2025 课堂监测项目组", font=("TkDefaultFont", 9))
        copyright_label.pack(side="bottom", pady=10)

    def update_video(self):
        """更新视频帧"""
        if self.running:
            frame_start = time.perf_counter()

            success, frame = self.monitor.cap.read()
            if not success:
                final_path = self.monitor.stop_realtime_recording()
                if self.video_source_mode == "file":
                    self.running = False
                    self.start_btn.config(state="normal")
                    self.pause_btn.config(state="disabled")
                    self._update_video_progress(force_end=True)
                    if final_path:
                        self.status_label.config(text=f"本地视频已播放结束，录制文件已保存: {final_path}")
                    else:
                        self.status_label.config(text="本地视频已播放结束，可重新打开视频或切换回摄像头。")
                else:
                    if final_path:
                        self.status_label.config(text=f"当前摄像头无法读取画面，录制文件已保存: {final_path}")
                    else:
                        self.status_label.config(text="当前摄像头无法读取画面。")
                self._refresh_recording_status()
                self.root.after(30, self.update_video)
                return

            try:
                current_time = time.time()
                if self.video_source_mode == "file":
                    self._last_video_frame_bgr = frame.copy()
                processed_frame = self.monitor.process_frame(frame, current_time)
                self.current_frame = processed_frame.copy()
            except Exception as e:
                print(f"帧处理错误: {str(e)}")
                self.root.after(30, self.update_video)
                return

            # 实时保存检测视频
            try:
                if getattr(self.monitor, "realtime_save_enabled", False):
                    if self.monitor.realtime_video_writer is None:
                        os.makedirs("realtime_videos", exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        src_tag = "file" if self.video_source_mode == "file" else "camera"
                        out_path = os.path.join("realtime_videos", f"detect_{src_tag}_{timestamp}.mp4")

                        h, w = processed_frame.shape[:2]
                        fps = int(round(self.monitor.target_fps or 30))
                        fps = max(1, min(60, fps))
                        self.monitor.realtime_video_fps = fps
                        writer = cv2.VideoWriter(out_path, self.monitor.realtime_video_fourcc, fps, (w, h))
                        if writer.isOpened():
                            self.monitor.realtime_video_writer = writer
                            self.monitor.realtime_video_path = out_path
                            self.status_label.config(text=f"检测视频实时保存中: {out_path}")
                            self._refresh_recording_status()
                        else:
                            writer.release()
                            self.monitor.realtime_video_writer = None
                            self.monitor.realtime_video_path = None
                            self._refresh_recording_status()
                    if self.monitor.realtime_video_writer is not None:
                        self.monitor.realtime_video_writer.write(processed_frame)
            except Exception as e:
                print(f"实时保存检测视频失败: {e}")

            attendance_rate = (self.monitor.current_count / self.monitor.total_students * 100
                               if self.monitor.total_students > 0 else 0)
            self.attendance_label.config(
                text=f"出勤率: {self.monitor.current_count}/{self.monitor.total_students} ({attendance_rate:.1f}%)"
            )

            class_metrics = self.monitor.calculate_classroom_metrics()
            self.analysis_label.config(
                text=f"专注度 {class_metrics['focus_rate']:.1f}% | 抬头率 {class_metrics['head_up_rate']:.1f}% | FPS {self.monitor.actual_fps:.0f}/{self.monitor.target_fps}"
            )
            self.habit_label.config(text=f"主导习惯: {class_metrics['dominant_habit']}")
            
            if self.monitor.detected_objects:
                object_names = [obj["class_name"] for obj in self.monitor.detected_objects]
                self.object_label.config(
                    text=f"检测到桌面物品: {', '.join(set(object_names))}",
                    fg="#FF9800"
                )
            else:
                self.object_label.config(text="桌面物品: 未检测", fg="#FF9800")
            
            # 更新人脸识别状态
            if self.monitor.face_recognition_enabled:
                recognized_count = sum(1 for state in self.monitor.student_states.values() 
                                     if state.get("identity", "未知") != "未知")
                self.face_label.config(
                    text=f"已识别学生: {recognized_count}人",
                    fg="#4CAF50"
                )
            else:
                self._refresh_face_status()

            self._render_frame_on_canvas(processed_frame)
            if self.video_source_mode == "file":
                self._update_video_progress()

            process_time = time.perf_counter() - frame_start
            self.monitor._frame_processing_times.append(process_time)
            if len(self.monitor._frame_processing_times) > 5:
                self.monitor._frame_processing_times.pop(0)

            self.monitor._fps_counter += 1
            if time.time() - self.monitor._fps_last_time >= 1.0:
                self.monitor.actual_fps = self.monitor._fps_counter
                self.monitor._fps_counter = 0
                self.monitor._fps_last_time = time.time()

            target_delay = max(1, int((1 / self.monitor.target_fps - process_time) * 1000))
            self.root.after(target_delay, self.update_video)
        else:
            self.root.after(30, self.update_video)

    def on_closing(self):
        """关闭窗口时的处理"""
        if getattr(self, "_closing", False):
            return
        self._closing = True

        if self.running:
            self.running = False
            time.sleep(0.5)

        try:
            self.status_label.config(text="正在生成最终报告...")
            self.monitor.close_finalize_reports()
        except Exception as e:
            print(f"生成最终报告时出错: {str(e)}")

        self.monitor.release()
        self.root.destroy()


if __name__ == "__main__":
    ClassroomMonitorGUI()
