from collections import defaultdict


def ensure_student_state_fields(state):
    state.setdefault("total_frames", 0)
    state.setdefault("focused_frames", 0)
    state.setdefault("head_up_frames", 0)
    state.setdefault("head_down_frames", 0)
    state.setdefault("head_turn_frames", 0)
    state.setdefault("distracted_frames", 0)
    state.setdefault("habit_label", "insufficient_data")
    state.setdefault("identity", "未知")


def classify_learning_habit(metrics):
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


def calculate_student_metrics(student_states, person_id):
    state = student_states[person_id]
    ensure_student_state_fields(state)
    total_frames = state["total_frames"]

    if total_frames <= 0:
        return {
            "total_frames": 0,
            "focus_rate": 0.0,
            "head_up_rate": 0.0,
            "head_down_rate": 0.0,
            "head_turn_rate": 0.0,
            "distracted_rate": 0.0,
            "habit_label": "数据不足",
        }

    metrics = {
        "total_frames": total_frames,
        "focus_rate": state["focused_frames"] / total_frames * 100,
        "head_up_rate": state["head_up_frames"] / total_frames * 100,
        "head_down_rate": state["head_down_frames"] / total_frames * 100,
        "head_turn_rate": state["head_turn_frames"] / total_frames * 100,
        "distracted_rate": state["distracted_frames"] / total_frames * 100,
    }
    metrics["habit_label"] = classify_learning_habit(metrics)
    return metrics


def update_student_statistics(student_states, person_id, student_number, is_head_down, is_head_turned, is_focused):
    state = student_states[person_id]
    ensure_student_state_fields(state)
    state["student_number"] = student_number
    state["total_frames"] += 1
    state["focused_frames"] += int(is_focused)
    state["head_up_frames"] += int(not is_head_down)
    state["head_down_frames"] += int(is_head_down)
    state["head_turn_frames"] += int(is_head_turned)
    state["distracted_frames"] += int(not is_focused)

    metrics = calculate_student_metrics(student_states, person_id)
    state["habit_label"] = metrics["habit_label"]
    return metrics


def calculate_classroom_metrics(student_states):
    student_metrics = []
    habit_counts = defaultdict(int)

    for person_id, state in student_states.items():
        ensure_student_state_fields(state)
        if state["student_number"] is None or state["total_frames"] <= 0:
            continue
        metrics = calculate_student_metrics(student_states, person_id)
        student_metrics.append(metrics)
        habit_counts[metrics["habit_label"]] += 1

    if not student_metrics:
        return {
            "students_analyzed": 0,
            "focus_rate": 0.0,
            "head_up_rate": 0.0,
            "dominant_habit": "insufficient_data",
        }

    focus_rate = sum(item["focus_rate"] for item in student_metrics) / len(student_metrics)
    head_up_rate = sum(item["head_up_rate"] for item in student_metrics) / len(student_metrics)
    dominant_habit = max(habit_counts.items(), key=lambda item: item[1])[0] if habit_counts else "mixed"

    return {
        "students_analyzed": len(student_metrics),
        "focus_rate": focus_rate,
        "head_up_rate": head_up_rate,
        "dominant_habit": dominant_habit,
    }


def get_parameter_snapshot(monitor):
    return {
        "head_down_threshold": monitor.head_down_threshold,
        "time_threshold": monitor.time_threshold,
        "head_turn_threshold": monitor.head_turn_threshold,
        "confidence_threshold": monitor.confidence_threshold,
        "filter_size": monitor.filter_size,
        "total_students": monitor.total_students,
        "debug": monitor.debug,
        "beep_enabled": monitor.beep_enabled,
        "object_detection_enabled": monitor.object_detection_enabled,
        "face_recognition_enabled": monitor.face_recognition_enabled,
    }


def get_student_alert_count(attention_logs, person_id):
    logs = attention_logs.get(person_id, [])
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


def get_warning_events(attention_logs):
    warning_events = []
    for person_id, logs in attention_logs.items():
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
