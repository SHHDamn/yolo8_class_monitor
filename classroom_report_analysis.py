def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_pct(value):
    return f"{_to_float(value):.2f}%"


def _safe_average(values):
    clean_values = [_to_float(value) for value in values]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def _split_three_parts(values):
    clean_values = [_to_float(value) for value in values]
    if len(clean_values) < 3:
        return [], [], []

    part_size = max(1, len(clean_values) // 3)
    first_part = clean_values[:part_size]
    middle_part = clean_values[part_size:-part_size]
    last_part = clean_values[-part_size:]
    if not middle_part:
        middle_part = clean_values[part_size:part_size * 2]
    return first_part, middle_part, last_part


def _build_data_validity_lines(duration_sec, student_rows, focus_series):
    lines = ["数据有效性说明"]
    warnings = []
    if duration_sec < 60:
        warnings.append("检测时长较短")
    if len(student_rows) < 2:
        warnings.append("参与统计对象较少")
    if len(focus_series) < 3:
        warnings.append("趋势序列样本不足")

    if warnings:
        lines.append(f"- 本次数据存在{ '、'.join(warnings) }的情况，分析结果仅供参考。")
    else:
        lines.append("- 本次检测样本具备基本趋势分析条件，仍建议结合课堂实际观察理解结果。")
    return lines


def _build_overall_lines(overall_focus_pct, overall_head_up_pct, overall_head_down_pct):
    focus_pct = _to_float(overall_focus_pct)
    head_up_pct = _to_float(overall_head_up_pct)
    head_down_pct = _to_float(overall_head_down_pct)

    lines = ["班级整体状态分析"]
    if focus_pct >= 80:
        lines.append(f"- 总体专注度为 {_format_pct(focus_pct)}，提示整体课堂状态较好。")
    elif focus_pct >= 60:
        lines.append(f"- 总体专注度为 {_format_pct(focus_pct)}，提示整体较稳定，但可能存在一定分心现象。")
    else:
        lines.append(f"- 总体专注度为 {_format_pct(focus_pct)}，提示注意力水平偏低，建议结合课堂实际观察关注。")

    if head_up_pct >= 70:
        lines.append(f"- 总体抬头率为 {_format_pct(head_up_pct)}，提示大部分时间能观察到抬头或面向课堂的外显行为。")
    elif head_down_pct >= 40:
        lines.append(
            f"- 总体低头率为 {_format_pct(head_down_pct)}，提示低头行为占比较高，"
            "需结合书写、阅读、看资料、看手机或疲劳等课堂情境判断。"
        )
    else:
        lines.append(
            f"- 总体抬头率为 {_format_pct(head_up_pct)}，总体低头率为 {_format_pct(head_down_pct)}，"
            "提示抬头和低头行为分布相对均衡。"
        )
    return lines


def _build_trend_lines(focus_series):
    lines = ["趋势变化分析"]
    first_part, middle_part, last_part = _split_three_parts(focus_series)
    if not first_part or not middle_part or not last_part:
        lines.append("- 专注率趋势样本不足，无法形成稳定的前段、中段、后段比较结论。")
        return lines

    first_avg = _safe_average(first_part)
    middle_avg = _safe_average(middle_part)
    last_avg = _safe_average(last_part)
    if first_avg is None or middle_avg is None or last_avg is None:
        lines.append("- 专注率趋势样本不足，无法形成稳定的前段、中段、后段比较结论。")
        return lines

    delta = last_avg - first_avg
    lines.append(
        f"- 前段平均专注率为 {_format_pct(first_avg)}，中段为 {_format_pct(middle_avg)}，"
        f"后段为 {_format_pct(last_avg)}。"
    )
    if delta <= -15:
        lines.append("- 后段较前段下降超过 15 个百分点，提示后半段注意力可能下降明显。")
    elif abs(delta) < 10:
        lines.append("- 前后段变化小于 10 个百分点，提示课堂状态整体较稳定。")
    elif delta > 0:
        lines.append("- 后段专注率较前段有所提升，提示课堂后段状态可能有所改善。")
    else:
        lines.append("- 后段专注率较前段有所下降，建议结合课堂节奏变化观察原因。")

    low_focus_count = sum(1 for value in focus_series if _to_float(value) < 50)
    if low_focus_count >= max(3, int(len(focus_series) * 0.2)):
        lines.append("- 存在较多低于 50% 的时间点，提示课堂状态可能存在波动。")
    return lines


def _build_headpose_lines(overall_head_up_pct, overall_head_down_pct):
    lines = ["抬头/低头结构分析"]
    head_up_pct = _to_float(overall_head_up_pct)
    head_down_pct = _to_float(overall_head_down_pct)

    if head_down_pct >= 40:
        lines.append(
            f"- 低头率为 {_format_pct(head_down_pct)}，提示低头行为占比较高，"
            "但低头不必然代表分心。"
        )
    elif head_up_pct >= 70:
        lines.append(f"- 抬头率为 {_format_pct(head_up_pct)}，提示抬头行为占比较高。")
    else:
        lines.append(
            f"- 抬头率为 {_format_pct(head_up_pct)}，低头率为 {_format_pct(head_down_pct)}，"
            "提示课堂外显姿态分布较为分散。"
        )
    lines.append("- 低头可能来自书写、阅读、看资料、看手机或疲劳等多种情况，建议结合课堂实际观察判断。")
    return lines


def _build_student_difference_lines(student_rows):
    lines = ["个体差异分析"]
    focus_rows = []
    for row in student_rows:
        focus_pct = _to_float(row.get("focus_pct"))
        head_up_pct = _to_float(row.get("head_up_pct"))
        alert_count = int(_to_float(row.get("alert_count"), 0))
        reasons = []
        if focus_pct < 60:
            reasons.append(f"专注度 {_format_pct(focus_pct)} 低于 60%")
        if head_up_pct < 50:
            reasons.append(f"抬头率 {_format_pct(head_up_pct)} 低于 50%")
        if alert_count >= 2:
            reasons.append(f"告警次数 {alert_count} 次")
        if reasons:
            focus_rows.append((focus_pct, row, reasons))

    if not focus_rows:
        lines.append("- 未发现明显达到重点关注条件的对象，建议继续结合课堂实际观察。")
        return lines

    sorted_rows = sorted(focus_rows, key=lambda item: item[0])
    lines.append(f"- 共有 {len(sorted_rows)} 个对象达到重点关注条件，完整指标见后文“学生明细”。")
    lines.append("- 示例对象（最多 5 个，仅列编号/标签和触发条件，不作主观评价）：")
    for _focus_pct, row, reasons in sorted_rows[:5]:
        lines.append(f"  - {row.get('label', '未知对象')}: {'；'.join(reasons)}")
    if len(sorted_rows) > 5:
        lines.append("- 其余对象请参考“学生明细”，建议结合现场观察进行非惩罚性关注。")
    return lines


def _build_warning_lines(warning_events):
    lines = ["告警事件分析"]
    if not warning_events:
        lines.append("- 本次检测未出现持续告警。")
    else:
        lines.append(
            f"- 本次检测记录到 {len(warning_events)} 次告警提示，"
            "可能存在阶段性低头或分心状态，建议结合课堂实际观察复核。"
        )
    return lines


def _build_suggestion_lines(
    focus_series,
    student_rows,
    overall_focus_pct,
    overall_head_down_pct,
):
    lines = ["教学建议"]
    suggestions = []

    first_part, _middle_part, last_part = _split_three_parts(focus_series)
    first_avg = _safe_average(first_part)
    last_avg = _safe_average(last_part)
    if first_avg is not None and last_avg is not None and last_avg - first_avg <= -15:
        suggestions.append("- 后半段专注率下降较明显，建议在后半段增加互动、提问或节奏变化。")

    if _to_float(overall_head_down_pct) >= 40:
        suggestions.append("- 低头率较高，建议结合课堂内容判断是否处于书写、阅读或资料查看环节。")

    focus_targets = [
        row for row in student_rows
        if _to_float(row.get("focus_pct")) < 60
        or _to_float(row.get("head_up_pct")) < 50
        or int(_to_float(row.get("alert_count"), 0)) >= 2
    ]
    if focus_targets:
        suggestions.append("- 个体差异较明显，建议教师进行非惩罚性关注，并结合现场观察了解原因。")

    if _to_float(overall_focus_pct) >= 80 and not suggestions:
        suggestions.append("- 整体课堂状态较好，建议保持当前课堂组织方式，并持续观察后续变化。")

    if not suggestions:
        suggestions.append("- 建议结合课堂目标、教学环节和现场观察，对本次数据进行综合理解。")

    lines.extend(suggestions)
    return lines


def generate_classroom_analysis(
    duration_sec,
    student_rows,
    focus_series,
    head_up_series,
    head_down_series,
    overall_focus_pct,
    overall_head_up_pct,
    overall_head_down_pct,
    warning_events,
    parameter_snapshot,
):
    student_rows = list(student_rows or [])
    focus_series = list(focus_series or [])
    warning_events = list(warning_events or [])

    if not student_rows or not focus_series:
        return [
            "课堂状态分析",
            "--------------------",
            "数据有效性说明",
            "- 有效样本不足，无法形成稳定分析结论，分析结果仅供参考。",
            "班级整体状态分析",
            "- 当前缺少足够的有效检测数据，无法稳定判断班级整体状态。",
            "趋势变化分析",
            "- 当前缺少足够的趋势序列，无法形成稳定的前段、中段、后段比较结论。",
            "个体差异分析",
            "- 当前缺少足够的学生明细数据，无法形成稳定的个体差异分析。",
            "教学建议",
            "- 建议结合课堂实际观察和后续更长时段数据，再判断课堂状态变化。",
            "",
        ]

    lines = [
        "课堂状态分析",
        "--------------------",
    ]
    lines.extend(_build_data_validity_lines(duration_sec, student_rows, focus_series))
    lines.extend(_build_overall_lines(overall_focus_pct, overall_head_up_pct, overall_head_down_pct))
    lines.extend(_build_trend_lines(focus_series))
    lines.extend(_build_headpose_lines(overall_head_up_pct, overall_head_down_pct))
    lines.extend(_build_student_difference_lines(student_rows))
    lines.extend(_build_warning_lines(warning_events))
    lines.extend(_build_suggestion_lines(focus_series, student_rows, overall_focus_pct, overall_head_down_pct))
    lines.append("")
    return lines
