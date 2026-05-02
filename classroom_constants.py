BEHAVIOR_CLASS_ZH = {
    "reading": "阅读",
    "sleep": "睡觉",
    "hand_raising": "举手",
    "turn_head": "转头",
    "low_head": "低头",
    "head_down": "低头",
    "writing": "写作",
    "leaning_on_desk": "靠桌子",
    "closed_book": "合书",
    "electronic_book": "电子书",
    "no_book": "无书本",
    "opened_book": "开卷阅读",
    "student_answers": "起立作答",
    "student_reads": "阅读",
    "student_writes": "写作",
    "teacher_explains": "教师讲解",
    "teacher_follows_up_students": "教师辅导",
    "worksheet": "课堂作业",
}


def normalize_behavior_class_name(name):
    if name is None:
        return ""
    s = str(name).strip().lower().replace("-", "_")
    s = " ".join(s.replace("_", " ").split())
    return s.replace(" ", "_")

