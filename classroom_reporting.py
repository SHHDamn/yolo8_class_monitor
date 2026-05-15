import os
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_new_classroom_report(monitor, save_dir="attention_logs"):
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    logs_by_person = {}
    for person_id, logs in getattr(monitor, "attention_logs", {}).items():
        if not logs or len(logs) < 2:
            continue
        logs_by_person[person_id] = logs

    if not logs_by_person:
        txt_path = os.path.join(save_dir, "classroom_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("课堂分析报告（新版）\n")
            f.write("====================\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("本次会话无有效检测数据（未开始监测或尚未检测到学生）。\n")
        return

    first_ts = min(rows[0].get("timestamp", 0) for rows in logs_by_person.values())
    last_ts = max(rows[-1].get("timestamp", 0) for rows in logs_by_person.values())
    duration_sec = max(1, int(last_ts - first_ts))

    bins = [[0, 0, 0, 0] for _ in range(duration_sec + 1)]
    total_frames_all = 0
    focused_frames_all = 0
    head_up_frames_all = 0
    head_down_frames_all = 0

    for _pid, rows in logs_by_person.items():
        for row in rows:
            ts = float(row.get("timestamp", 0))
            t = int(max(0, min(duration_sec, ts - first_ts)))
            focused = bool(row.get("focused", False))
            head_up = bool(row.get("head_up", False))
            head_down = bool(row.get("head_down", False))

            bins[t][1] += 1
            bins[t][0] += int(focused)
            bins[t][2] += int(head_up)
            bins[t][3] += int(head_down)

            total_frames_all += 1
            focused_frames_all += int(focused)
            head_up_frames_all += int(head_up)
            head_down_frames_all += int(head_down)

    def safe_pct(n, d):
        return 0.0 if d <= 0 else (100.0 * float(n) / float(d))

    overall_focus_pct = safe_pct(focused_frames_all, total_frames_all)
    overall_head_up_pct = safe_pct(head_up_frames_all, total_frames_all)
    overall_head_down_pct = safe_pct(head_down_frames_all, total_frames_all)

    student_rows = []
    for pid, rows in logs_by_person.items():
        state = getattr(monitor, "student_states", {}).get(pid, {}) if hasattr(monitor, "student_states") else {}
        number = state.get("student_number")
        identity = state.get("identity", "未知") if isinstance(state, dict) else "未知"

        total = len(rows)
        focus_n = sum(1 for r in rows if r.get("focused"))
        head_up_n = sum(1 for r in rows if r.get("head_up"))
        head_down_n = sum(1 for r in rows if r.get("head_down"))
        alert_n = 0
        prev = False
        for r in rows:
            cur = bool(r.get("alert", False))
            if cur and not prev:
                alert_n += 1
            prev = cur

        label = f"#{number}" if number is not None else f"ID{pid}"
        if identity and identity != "未知":
            label = f"{label}-{identity}"

        student_rows.append(
            {
                "label": label,
                "total": total,
                "focus_pct": safe_pct(focus_n, total),
                "head_up_pct": safe_pct(head_up_n, total),
                "head_down_pct": safe_pct(head_down_n, total),
                "alert_count": alert_n,
            }
        )

    timeline = list(range(duration_sec + 1))
    focus_series = [safe_pct(b[0], b[1]) for b in bins]
    head_up_series = [safe_pct(b[2], b[1]) for b in bins]
    head_down_series = [safe_pct(b[3], b[1]) for b in bins]

    plt.figure(figsize=(12, 5))
    plt.plot(timeline, focus_series, color="#1976D2", linewidth=2, label="专注率(%)")
    plt.ylim(0, 100)
    plt.xlabel("时间(秒)")
    plt.ylabel("比例(%)")
    plt.title("班级专注率趋势（按秒聚合）")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "班级专注率趋势图.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(timeline, head_up_series, color="#2E7D32", linewidth=2, label="抬头率(%)")
    plt.plot(timeline, head_down_series, color="#FB8C00", linewidth=2, label="低头率(%)")
    plt.ylim(0, 100)
    plt.xlabel("时间(秒)")
    plt.ylabel("比例(%)")
    plt.title("班级抬头/低头趋势（按秒聚合）")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "班级抬头低头趋势图.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(6.6, 6.2))
    sizes = [focused_frames_all, max(0, total_frames_all - focused_frames_all)]
    if sum(sizes) <= 0:
        plt.text(0.5, 0.5, "无专注度样本", ha="center", va="center")
        plt.axis("off")
    else:
        labels = [
            f"专注 {overall_focus_pct:.1f}%",
            f"非专注 {100.0 - overall_focus_pct:.1f}%",
        ]
        plt.pie(
            sizes,
            labels=labels,
            colors=["#1976D2", "#FF9800"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.35},
        )
        plt.title("专注分布（全量样本）")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "班级专注分布饼图.png"), dpi=220)
    plt.close()

    student_rows_sorted = sorted(student_rows, key=lambda r: r["focus_pct"])
    top_n = min(12, len(student_rows_sorted))
    worst = student_rows_sorted[:top_n]
    labels = [r["label"] for r in worst]
    focus_vals = [r["focus_pct"] for r in worst]
    head_up_vals = [r["head_up_pct"] for r in worst]

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, focus_vals, width, label="专注度(%)", color="#1976D2")
    plt.bar(x + width / 2, head_up_vals, width, label="抬头率(%)", color="#2E7D32")
    plt.ylim(0, 100)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("比例(%)")
    plt.title("学生表现排行（专注度/抬头率，低到高）")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "学生表现排行图.png"), dpi=220)
    plt.close()

    try:
        warning_events = monitor.get_warning_events() if hasattr(monitor, "get_warning_events") else []
    except Exception as e:
        print(f"获取告警事件失败: {e}")
        warning_events = []

    plt.figure(figsize=(12, 5))
    if not warning_events:
        plt.text(0.5, 0.5, "当前无告警数据", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        bucket = defaultdict(int)
        for ev in warning_events:
            t = int(max(0, ev.get("timestamp", 0) - first_ts))
            bucket[t] += 1
        xs = sorted(bucket.keys())
        ys = [bucket[x] for x in xs]
        plt.plot(xs, ys, marker="o", color="#EF5350")
        plt.xlabel("时间(秒)")
        plt.ylabel("告警触发次数")
        plt.title("课堂告警时间线")
        plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "课堂告警时间线图.png"), dpi=220)
    plt.close()

    try:
        parameter_snapshot = monitor.get_parameter_snapshot() if hasattr(monitor, "get_parameter_snapshot") else {}
    except Exception as e:
        print(f"获取参数快照失败: {e}")
        parameter_snapshot = {}

    txt_path = os.path.join(save_dir, "classroom_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("课堂分析报告（新版）\n")
        f.write("====================\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析时长: {duration_sec} 秒\n")
        f.write(f"参与统计对象数: {len(student_rows)}\n\n")

        f.write("班级概览\n")
        f.write("--------------------\n")
        f.write(f"总体专注度: {overall_focus_pct:.2f}%\n")
        f.write(f"总体抬头率: {overall_head_up_pct:.2f}%\n")
        f.write(f"总体低头率: {overall_head_down_pct:.2f}%\n\n")

        f.write("图表输出\n")
        f.write("--------------------\n")
        f.write("1) 班级专注率趋势图.png：班级专注率趋势\n")
        f.write("2) 班级抬头低头趋势图.png：抬头/低头趋势\n")
        f.write("3) 班级专注分布饼图.png：专注分布饼图\n")
        f.write("4) 学生表现排行图.png：学生专注度/抬头率排行\n")
        f.write("5) 课堂告警时间线图.png：告警时间线\n\n")

        if parameter_snapshot:
            f.write("参数快照\n")
            f.write("--------------------\n")
            for k, v in parameter_snapshot.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        f.write("学生明细（按专注度从低到高）\n")
        f.write("--------------------\n")
        for row in student_rows_sorted:
            f.write(
                f"{row['label']}: 专注度={row['focus_pct']:.2f}%, "
                f"抬头率={row['head_up_pct']:.2f}%, "
                f"低头占比={row['head_down_pct']:.2f}%, "
                f"告警次数={row['alert_count']}\n"
            )
        f.write("\n")

        f.write("解释与注意事项\n")
        f.write("--------------------\n")
        f.write("1) 本系统指标适合做“群体趋势”分析，不建议用作对单个学生的唯一评价依据。\n")
        f.write("2) 低头并不必然代表分心：可能是书写、阅读或记笔记；抬头也不必然代表专注。\n")
        f.write("3) 遮挡、光照、摄像机角度会影响识别准确性。\n")

