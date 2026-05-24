import os
import time

import cv2


def stop_realtime_recording(monitor):
    final_path = monitor.realtime_video_path
    if monitor.realtime_video_writer is not None:
        monitor.realtime_video_writer.release()
        monitor.realtime_video_writer = None
    monitor.realtime_video_path = None
    monitor.realtime_video_fps = 0
    return final_path


def set_recording_enabled(monitor, enabled):
    monitor.realtime_save_enabled = bool(enabled)
    if not monitor.realtime_save_enabled:
        return stop_realtime_recording(monitor)
    return monitor.realtime_video_path


def ensure_recording_writer(monitor, processed_frame, source_mode, generation):
    events = []
    if not getattr(monitor, "realtime_save_enabled", False):
        if monitor.realtime_video_writer is not None:
            final_path = stop_realtime_recording(monitor)
            events.append({
                "type": "recording_stopped",
                "generation": generation,
                "path": final_path,
                "message": f"录制已停止，文件已保存: {final_path}" if final_path else "录制已关闭。",
            })
        return events

    if monitor.realtime_video_writer is None:
        os.makedirs("realtime_videos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        src_tag = "file" if source_mode == "file" else "camera"
        out_path = os.path.join("realtime_videos", f"detect_{src_tag}_{timestamp}.mp4")

        h, w = processed_frame.shape[:2]
        fps = int(round(monitor.target_fps or 30))
        fps = max(1, min(60, fps))
        monitor.realtime_video_fps = fps
        writer = cv2.VideoWriter(out_path, monitor.realtime_video_fourcc, fps, (w, h))
        if writer.isOpened():
            monitor.realtime_video_writer = writer
            monitor.realtime_video_path = out_path
            events.append({
                "type": "recording_started",
                "generation": generation,
                "path": out_path,
                "message": f"检测视频实时保存中: {out_path}",
            })
        else:
            writer.release()
            monitor.realtime_video_writer = None
            monitor.realtime_video_path = None
            monitor.realtime_save_enabled = False
            events.append({
                "type": "recording_stopped",
                "generation": generation,
                "path": None,
                "message": "实时保存检测视频失败：无法创建视频写入器。",
            })
            return events

    if monitor.realtime_video_writer is not None:
        monitor.realtime_video_writer.write(processed_frame)
    return events
