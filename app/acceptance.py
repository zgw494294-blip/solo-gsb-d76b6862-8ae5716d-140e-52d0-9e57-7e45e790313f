"""步道坡线规划台 —— 施工验收复核。

施工方现场测出各测点的实测标高后，与此前**冻结**的可行方案做验收核对。
本模块只做核对，**不对实测、也不对冻结方案重新求优**：

1. 入参为原地面 ``ground``、已冻结的方案 ``plan``、与方案等数量的实测
   标高 ``measured``、容差 ``tolerance``（0～20 的整数厘米）。
2. 先独立复核冻结方案本身对原地面合法：
   测点 3～16 个、标高 0～200 整数、数量与地面一致、首/尾标高分别等于
   原地面首/尾、相邻高差 ≤ 8 cm、任意连续 5 段至少 1 段水平。规则判定
   全部复用 :func:`app.planner.step_rule`（与规划、可锁定分析同源，不另写
   一套），但不调用 DP 求解器，也不要求方案是最优解。
3. 再核对实测坡线，输出四类违规：

   - 逐点偏差 = 实测 − 方案，``|偏差| > 容差`` 的点为超容差点
     （恰好等于容差不算超）；
   - 实测相邻点高差绝对值超过 8 cm 的段；
   - 实测坡线中不含任何水平段的连续 5 段窗口（不足 5 段时不触发，
     语义与规划规则一致）；
   - 实测首、尾标高与方案首、尾不相等的点。

   四类违规全部为空时 ``passed = true``。实测标高本身仍限 0～200 的整数
   厘米；实测违反坡度/水平段约束只如实列入报告，由调用方决定是否放行，
   本函数不会据此改动任何输入。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .planner import (
    H_MAX,
    H_MIN,
    INITIAL_MASK,
    MAX_GRADE,
    WINDOW,
    PlanError,
    _validate,
    step_rule,
)

TOL_MIN = 0       # 容差下限（厘米）
TOL_MAX = 20      # 容差上限（厘米）


def _validate_levels(values: Any, name: str) -> None:
    """校验标高序列：数组、每项 0～200 的整数厘米（布尔不算整数）。"""
    if not isinstance(values, list):
        raise PlanError(f"{name}必须是数组。")
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, int):
            raise PlanError(f"第 {i + 1} 个{name}必须是整数厘米。")
        if not (H_MIN <= v <= H_MAX):
            raise PlanError(
                f"第 {i + 1} 个{name} {v} 超出范围"
                f"（{H_MIN}～{H_MAX} 厘米）。"
            )


def _validate_frozen_plan(ground: List[int], plan: Any) -> None:
    """独立复核冻结方案对原地面合法（不调用求解器，不要求最优）。

    检查数量与地面一致、标高 0～200 整数、首尾分别等于地面首尾，并逐段
    经 :func:`planner.step_rule` 校验相邻高差 ≤ 8 cm 与连续 5 段水平窗口。
    """
    if not isinstance(plan, list):
        raise PlanError("冻结方案标高必须是数组。")
    if len(plan) != len(ground):
        raise PlanError(
            f"冻结方案测点数量（{len(plan)} 个）必须与原地面"
            f"（{len(ground)} 个）一致。"
        )
    _validate_levels(plan, "冻结方案标高")

    n = len(plan)
    if plan[0] != ground[0] or plan[n - 1] != ground[n - 1]:
        raise PlanError("冻结方案首、尾标高必须分别等于原地面首、尾标高。")

    mask = INITIAL_MASK
    for i in range(n - 1):
        h, nh = plan[i], plan[i + 1]
        nm = step_rule(mask, h, nh, i)
        if nm is None:
            if abs(nh - h) > MAX_GRADE:
                sign = "+" if nh - h > 0 else ""
                raise PlanError(
                    f"冻结方案第 {i + 1} 段高差 {sign}{nh - h} 厘米，"
                    f"绝对值超过 {MAX_GRADE} 厘米。"
                )
            # step_rule 中只有凑满连续 5 段且全非水平才会因窗口约束拒绝。
            raise PlanError(
                f"冻结方案第 {max(1, i - 3)}～{i + 1} 段这一连续 {WINDOW} 段"
                "窗口内没有水平段，不满足水平段约束。"
            )
        mask = nm


def _validate_tolerance(tolerance: Any) -> int:
    """校验容差：0～20 的整数厘米（布尔不算整数）。"""
    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise PlanError("容差必须是 0～20 的整数厘米。")
    if not (TOL_MIN <= tolerance <= TOL_MAX):
        raise PlanError(
            f"容差必须在 {TOL_MIN}～{TOL_MAX} 厘米之间"
            f"（当前 {tolerance} 厘米）。"
        )
    return tolerance


def acceptance_review(
    ground: Any, plan: Any, measured: Any, tolerance: Any
) -> Dict[str, Any]:
    """施工验收复核：核对实测与冻结方案，全程不重新求优。

    可行时（指输入合法、冻结方案本身合法）返回::

        {
          "passed": bool,
          "ground": [...], "plan": [...], "measured": [...],
          "tolerance": int,
          "deviations": [实测−方案, ...],
          "over_tolerance_points": [{"index", "plan", "measured",
                                     "deviation"}],
          "over_grade_segments": [{"index", "from", "to", "delta"}],
          "no_horizontal_windows": [{"index", "from_segment",
                                     "to_segment", "segments": [...]}],
          "endpoint_mismatches": [{"index", "plan", "measured"}],
        }

    四类违规列表全部为空时 ``passed`` 为 ``True``。任何输入非法抛
    :class:`app.planner.PlanError`（地面/数量/标高复用规划校验）。
    """
    _validate(ground)
    _validate_frozen_plan(ground, plan)

    if not isinstance(measured, list):
        raise PlanError("实测标高必须是数组。")
    if len(measured) != len(plan):
        raise PlanError(
            f"实测标高数量（{len(measured)} 个）必须与冻结方案"
            f"（{len(plan)} 个）一致。"
        )
    _validate_levels(measured, "实测标高")
    tol = _validate_tolerance(tolerance)

    n = len(plan)

    # 1) 逐点偏差（实测减方案）与超容差点
    deviations = [measured[i] - plan[i] for i in range(n)]
    over_tolerance_points: List[Dict[str, Any]] = []
    for i, d in enumerate(deviations):
        if abs(d) > tol:
            over_tolerance_points.append({
                "index": i + 1,
                "plan": plan[i],
                "measured": measured[i],
                "deviation": d,
            })

    # 2) 实测相邻高差超过 8 cm 的段，并记录每段是否水平供窗口检查
    over_grade_segments: List[Dict[str, Any]] = []
    horizontal_flags: List[bool] = []
    for i in range(n - 1):
        delta = measured[i + 1] - measured[i]
        horizontal_flags.append(delta == 0)
        if abs(delta) > MAX_GRADE:
            over_grade_segments.append({
                "index": i + 1, "from": i + 1, "to": i + 2, "delta": delta,
            })

    # 3) 不含水平段的连续 5 段窗口（段数不足 5 时 range 为空，不触发）
    no_horizontal_windows: List[Dict[str, Any]] = []
    m = n - 1
    for start in range(0, m - WINDOW + 1):
        window = list(range(start + 1, start + WINDOW + 1))
        if not any(horizontal_flags[start:start + WINDOW]):
            no_horizontal_windows.append({
                "index": len(no_horizontal_windows) + 1,
                "from_segment": window[0],
                "to_segment": window[-1],
                "segments": window,
            })

    # 4) 实测首尾与方案首尾不相等的点
    endpoint_mismatches: List[Dict[str, Any]] = []
    for i in (0, n - 1):
        if measured[i] != plan[i]:
            endpoint_mismatches.append({
                "index": i + 1, "plan": plan[i], "measured": measured[i],
            })

    passed = not (
        over_tolerance_points
        or over_grade_segments
        or no_horizontal_windows
        or endpoint_mismatches
    )

    return {
        "passed": passed,
        "ground": list(ground),
        "plan": list(plan),
        "measured": list(measured),
        "tolerance": tol,
        "deviations": deviations,
        "over_tolerance_points": over_tolerance_points,
        "over_grade_segments": over_grade_segments,
        "no_horizontal_windows": no_horizontal_windows,
        "endpoint_mismatches": endpoint_mismatches,
    }
