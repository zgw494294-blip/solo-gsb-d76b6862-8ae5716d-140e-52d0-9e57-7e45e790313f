"""步道坡线规划台 —— 测点可锁定标高分析。

沿用 planner 的全部规则（3～16 个测点、标高 0～200 整数厘米、首尾固定、
相邻高差 ≤ 8 厘米、任意连续 5 段至少 1 段水平），逐点回答：

    若只把第 i 个测点的方案标高锁定为 h，其余测点自由选择，
    是否仍存在一条完整可行的坡线？

不限总调整量，也不要求与现有最优方案一致，因此这是纯可行性判定。
每个测点返回精确的可锁定值集合，并压缩为升序、互不相邻的整数闭区间
（如 [[92, 108], [110, 112]]；空档不填平，绝不只用最小/最大值概括）。

算法
----
与最优求解共用同一套状态与转移语义，不另写规则：状态为
(pos, mask, h)，mask 打包到达 pos 时最近 4 段的水平状态（bit 0 为最近
一段），不存在的更早段以“水平(1)”填充（planner.INITIAL_MASK），使不足
5 段时不误触发约束；每条候选转移是否合法只由 planner.step_rule 判定，
相邻高差与 0～200 值域只由 planner.height_bounds 夹取。

- fwd[pos][mask][h]：能否从首点（标高固定为 ground[0]）走到 pos 点取 h
  （前向布尔可达性，本模块计算）；
- 后向可达性直接复用 planner.backward_cost_dp：
  dp[pos][mask][h] < INF 当且仅当从 pos 点取 h 出发能合法走到尾点
  （标高固定为 ground[-1]）。这正是最优求解所用的同一张后向 DP 表。

跨过 pos 的连续 5 段窗口由两侧共享的 mask 校验，因此第 i 点可锁定 h
当且仅当存在 mask 使 fwd[i][mask][h] 与 dp[i][mask][h] < INF 同时成立。
可行性判定与 /api/plan 同源，同一地面数据不可能在两种结果中冲突。
复杂度 O(n · 2^4 · 201 · (2·8+1))，16 个测点约两百万次基本操作。
"""

from __future__ import annotations

from typing import List

from .planner import (
    H_MAX,
    H_MIN,
    INF,
    RECENT,
    backward_cost_dp,
    height_bounds,
    _step_rules,
    INITIAL_MASK,
    _infeasible_reason,
    _validate,
)


def _forward_reachable(ground: List[int]) -> list:
    """fwd[pos][mask][h]：前缀 0..pos 可行且 plan[pos] == h。

    与最优求解的重建过程走同一套转移（planner 的 step_rule 规则核心
    ``_step_rules``）：从 INITIAL_MASK、ground[0] 出发，逐段枚举合法
    下一标高（相邻高差与值域由 height_bounds 夹取）。
    """
    n = len(ground)
    size = H_MAX + 1
    fwd = [[[False] * size for _ in range(1 << RECENT)] for _ in range(n)]
    fwd[0][INITIAL_MASK][ground[0]] = True
    for pos in range(n - 1):
        nxt = fwd[pos + 1]
        rules = _step_rules(pos)
        for mask in range(1 << RECENT):
            flat_mask, slope_mask = rules[mask]
            row = fwd[pos][mask]
            if flat_mask is None and slope_mask is None:
                continue
            for h in range(H_MIN, H_MAX + 1):
                if not row[h]:
                    continue
                if flat_mask is not None:
                    nxt[flat_mask][h] = True
                if slope_mask is not None:
                    lo, hi = height_bounds(h)
                    for nh in range(lo, h):
                        nxt[slope_mask][nh] = True
                    for nh in range(h + 1, hi + 1):
                        nxt[slope_mask][nh] = True
    return fwd


def _compress_intervals(values: List[int]) -> List[List[int]]:
    """把升序整数序列压缩为升序且互不相邻的闭区间 [[lo, hi], ...]。

    只合并真正连续的整数；相邻区间之间至少隔一个缺失值，
    不会用首尾最小/最大值把空档填平。
    """
    intervals: List[List[int]] = []
    for v in values:
        if intervals and v == intervals[-1][1] + 1:
            intervals[-1][1] = v
        else:
            intervals.append([v, v])
    return intervals


def lockable_analysis(ground: List[int]) -> dict:
    """逐测点计算可锁定标高区间。

    可行时返回::

        {
          "feasible": True,
          "ground": [...],
          "points": [{"index": 1, "intervals": [[lo, hi], ...]}, ...],
        }

    ``intervals`` 为升序、互不相邻的整数闭区间；锁定该测点方案标高为
    区间内任意整数时，其余测点必能补全为一条完整可行坡线，区间外则不能。
    全线无解时返回 {"feasible": False, "reason": "..."}；非法输入抛
    :class:`app.planner.PlanError`（复用 planner 的校验）。
    """
    _validate(ground)
    n = len(ground)
    fwd = _forward_reachable(ground)
    dp = backward_cost_dp(ground)     # 与 /api/plan 同一张后向 DP 表

    points = []
    for pos in range(n):
        f_layer = fwd[pos]
        b_layer = dp[pos]
        values = []
        for h in range(H_MIN, H_MAX + 1):
            for mask in range(1 << RECENT):
                if f_layer[mask][h] and b_layer[mask][h] < INF:
                    values.append(h)
                    break
        points.append({"index": pos + 1, "intervals": _compress_intervals(values)})

    if not points[0]["intervals"]:
        # 首点只能取 ground[0]；它都无可锁定值即全线无解。
        return {"feasible": False, "reason": _infeasible_reason(ground)}

    return {"feasible": True, "ground": list(ground), "points": points}
