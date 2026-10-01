"""步道坡线规划台 —— 规划算法。

规则
----
1. 测点等间距 1 m，共 n 个测点（3 <= n <= 16）。
2. 地面标高 ground[i] 为 0~200 的整数厘米；方案标高 plan[i] 同为 0~200 的
   整数厘米。
3. 首、尾方案标高必须分别等于首、尾地面标高。
4. 相邻方案点高差的绝对值不超过 8 cm。
5. 任意连续 5 段中至少有 1 段水平段（相邻两点方案标高相等的段）。
6. 目标：在所有可行方案中最小化总调整量
       sum_i |plan[i] - ground[i]|；
   总调整量相同则取方案标高序列字典序最小者。

算法
----
带“最近 4 段是否水平”位掩码状态的动态规划。
向后 DP：dp[pos][mask][h] 表示第 pos 点方案标高取 h、其前 4 段
（pos-4..pos-3、…、pos-1..pos）的水平状态打包为 mask（bit 0 为最近一段）
时，pos 点之后（含本点）的最小代价；不存在的更早段以“水平(1)”填充，
使不足 5 段时不会触发水平段约束。整体最优值为 dp[0][0b1111][ground[0]]，
为 INF 即无解。重建时逐点取最小可行标高，天然得到同分下字典序最小的序列。

转移与边界语义只有一份实现：:func:`step_rule` 统一判定一条候选转移
（前点 mask/标高 → 后点标高）是否合法（标高值域、相邻高差、连续 5 段
水平窗口、起点之前虚拟段视为水平、首点初始掩码），并给出到达后点的
新掩码；:func:`height_bounds` 统一相邻高差与 0～200 值域的夹取。
最优求解（:func:`backward_cost_dp`）与方案重建（:func:`plan_trail`）
都只经过这两个函数，可锁定分析（:mod:`app.lockable`）的前向可达性
同样只调用它们，二者因此不可能对同一地面数据给出可行性冲突。
"""

from __future__ import annotations

from typing import List

N_MIN = 3
N_MAX = 16
H_MIN = 0
H_MAX = 200
MAX_GRADE = 8          # 相邻方案点允许的最大高差（厘米）
WINDOW = 5             # 任意连续 5 段
RECENT = WINDOW - 1    # 状态中记录最近 4 段
_MASK_RECENT = (1 << RECENT) - 1     # 0b1111
_MASK_WINDOW = (1 << WINDOW) - 1     # 0b11111
INITIAL_MASK = _MASK_RECENT          # 首点之前的虚拟段全部视为水平
INF = 10 ** 18


class PlanError(ValueError):
    """输入非法或问题无可行解。"""


def _validate(ground: List[int]) -> None:
    if not isinstance(ground, list):
        raise PlanError("地面标高必须是数组。")
    if not (N_MIN <= len(ground) <= N_MAX):
        raise PlanError(
            f"测点数量必须在 {N_MIN}～{N_MAX} 个之间（当前 {len(ground)} 个）。"
        )
    for i, g in enumerate(ground):
        if isinstance(g, bool) or not isinstance(g, int):
            raise PlanError(f"第 {i + 1} 个地面标高必须是整数厘米。")
        if not (H_MIN <= g <= H_MAX):
            raise PlanError(
                f"第 {i + 1} 个地面标高 {g} 超出范围"
                f"（{H_MIN}～{H_MAX} 厘米）。"
            )


def _relaxed_reachable(ground: List[int]) -> bool:
    """只受“相邻高差 <= 8”与首尾固定约束时是否可行（双向区间传播）。"""
    n = len(ground)
    lo = [H_MIN] * n
    hi = [H_MAX] * n
    lo[0] = hi[0] = ground[0]
    lo[n - 1] = hi[n - 1] = ground[n - 1]
    for i in range(1, n - 1):
        lo[i] = max(H_MIN, lo[i - 1] - MAX_GRADE)
        hi[i] = min(H_MAX, hi[i - 1] + MAX_GRADE)
    for i in range(n - 2, 0, -1):
        lo[i] = max(lo[i], lo[i + 1] - MAX_GRADE)
        hi[i] = min(hi[i], hi[i + 1] + MAX_GRADE)
        if lo[i] > hi[i]:
            return False
    return lo[0] <= hi[0] and lo[n - 1] <= hi[n - 1]


def _infeasible_reason(ground: List[int]) -> str:
    n = len(ground)
    if not _relaxed_reachable(ground):
        end_gap = abs(ground[-1] - ground[0])
        limit = MAX_GRADE * (n - 1)
        extra = (
            f"：首尾高差 {end_gap} 厘米，超过 {n - 1} 段最多消化的 {limit} 厘米。"
            if end_gap > limit
            else "。"
        )
        return (
            "无解：在“相邻方案点高差不超过 8 厘米、首尾固定”的条件下，"
            "无法从首点标高连续过渡到尾点标高" + extra
        )
    return (
        "无解：满足相邻高差限制时，无法同时满足“任意连续 5 段中至少有 1 段"
        "水平段”，可尝试增加测点或调整地面标高。"
    )


def _window_mask(mask: int, horizontal: int) -> int:
    """加入新段后，最近 5 段的水平位（bit0 为最近一段）。"""
    return ((mask << 1) | horizontal) & _MASK_WINDOW


def height_bounds(h: int) -> tuple[int, int]:
    """从标高 h 放下一段时，下一点允许的标高闭区间（高差与值域的交集）。"""
    return max(H_MIN, h - MAX_GRADE), min(H_MAX, h + MAX_GRADE)


def _step_outcome(mask: int, horizontal: int, seg_index: int) -> int | None:
    """给定新段是否水平（0/1），返回合法时的到达掩码，否则 ``None``。

    窗口合法性与掩码更新只取决于 (seg_index, mask, horizontal)，与具体
    标高无关；:func:`step_rule` 与 :func:`_step_rules` 都汇聚到这里。
    """
    if seg_index >= WINDOW - 1 and _window_mask(mask, horizontal) == 0:
        return None
    return horizontal | ((mask << 1) & _MASK_RECENT)


def step_rule(mask: int, h: int, nh: int, seg_index: int) -> int | None:
    """唯一的合法转移判定（最优求解、重建、前向可达性共用）。

    判定从第 ``seg_index`` 段（0-based，连接测点 ``seg_index`` 与
    ``seg_index + 1``）放下标高 h → nh 的候选转移是否合法，并返回到达
    后点时的最近 4 段水平掩码；不合法返回 ``None``。规则集中在此：

    1. 方案标高必须在 ``H_MIN``～``H_MAX`` 内，相邻高差 ≤ ``MAX_GRADE``；
    2. 当该段凑满一个完整的连续 5 段窗口（``seg_index >= 4``）时，
       窗口 5 段中至少有 1 段水平（掩码与窗口判定见 :func:`_step_outcome`）；
    3. 起点之前不存在的段以“水平(1)”填充（调用方以 :data:`INITIAL_MASK`
       进入），使不足 5 段时不会误触发窗口约束。

    首尾固定不在此处理（由各 DP 的初值/边界保证：首点只从
    ``INITIAL_MASK``、ground[0] 出发，末点只接受 ground[-1]）。
    """
    if nh < H_MIN or nh > H_MAX or abs(nh - h) > MAX_GRADE:
        return None
    return _step_outcome(mask, 1 if nh == h else 0, seg_index)


def _step_rules(seg_index: int) -> list:
    """每个入口掩码在第 ``seg_index`` 段上的合法转移预算表（热点加速）。

    窗口合法性与掩码更新只取决于 (seg_index, mask)，与具体标高无关，
    因此整张表只算一次；结论仍逐字段取自 :func:`_step_outcome`
    （即 :func:`step_rule` 的规则核心），不另行复制语义，保证最优求解
    与可达性循环用的是同一套转移规则。

    每项为 ``(flat_mask | None, slope_mask | None)``：分别对应水平段
    （nh == h）与非水平段（nh != h，允许的标高范围由调用方按
    :func:`height_bounds` 去掉 h 后枚举）。
    """
    return [
        (
            _step_outcome(mask, 1, seg_index),
            _step_outcome(mask, 0, seg_index),
        )
        for mask in range(1 << RECENT)
    ]


def backward_cost_dp(ground: List[int]) -> list:
    """后向成本 DP：dp[pos][mask][h] 为该状态后缀（含本点）最小代价。

    取 INF 即表示“从该 (pos, mask, h) 出发无法合法走到固定尾点”，
    因此同一张表既是最优求解器的成本表，也是后向可达性表，
    被 :func:`plan_trail` 与可锁定分析共用。转移合法性来自
    :func:`step_rule`（经 :func:`_step_rules` 预算），边界夹取来自
    :func:`height_bounds`。
    """
    n = len(ground)
    size = H_MAX + 1
    dp = [
        [[INF] * size for _ in range(1 << RECENT)]
        for _ in range(n)
    ]
    glast = ground[n - 1]
    for mask in range(1 << RECENT):
        dp[n - 1][mask][glast] = abs(glast - ground[n - 1])

    for pos in range(n - 2, -1, -1):
        layer = dp[pos]
        next_layer = dp[pos + 1]
        rules = _step_rules(pos)
        cost_h = [abs(h - ground[pos]) for h in range(size)]
        for mask in range(1 << RECENT):
            flat_mask, slope_mask = rules[mask]
            cell = layer[mask]
            for h in range(H_MIN, H_MAX + 1):
                lo, hi = height_bounds(h)
                base = cost_h[h]
                best = INF

                if flat_mask is not None:
                    cand = base + next_layer[flat_mask][h]
                    if cand < best:
                        best = cand

                if slope_mask is not None:
                    nl = next_layer[slope_mask]
                    for nh in range(lo, h):
                        cand = base + nl[nh]
                        if cand < best:
                            best = cand
                    for nh in range(h + 1, hi + 1):
                        cand = base + nl[nh]
                        if cand < best:
                            best = cand
                cell[h] = best
    return dp


def plan_trail(ground: List[int]) -> dict:
    """计算最优步道坡线方案。

    可行时返回::

        {
          "feasible": True,
          "ground": [...], "plan": [...],
          "total_adjustment": int,
          "segments": [{"index", "from", "to", "delta", "horizontal"}],
          "horizontal_indices": [...],   # 1-based 段号
        }

    无解时返回 {"feasible": False, "reason": "..."}；非法输入抛 PlanError。
    """
    _validate(ground)
    n = len(ground)
    g0, glast = ground[0], ground[-1]

    dp = backward_cost_dp(ground)
    total = dp[0][INITIAL_MASK][g0]
    if total >= INF:
        return {"feasible": False, "reason": _infeasible_reason(ground)}

    # 逐点选择最小可行标高，得到同分下字典序最小的方案。
    # 可行性与成本表同源于 step_rule，因此这里能选中的转移必然是
    # dp 表认可的合法后缀，二者不会互相矛盾。
    plan = [0] * n
    plan[0] = g0
    mask = INITIAL_MASK
    spent = abs(g0 - ground[0])
    for pos in range(0, n - 1):
        h = plan[pos]
        chosen = None
        lo, hi = height_bounds(h)
        for nh in range(lo, hi + 1):
            nmask = step_rule(mask, h, nh, pos)
            if nmask is not None and spent + dp[pos + 1][nmask][nh] == total:
                chosen = (nh, nmask)
                break
        assert chosen is not None, "重建最优方案失败"
        plan[pos + 1], mask = chosen
        spent += abs(plan[pos + 1] - ground[pos + 1])

    assert plan[-1] == glast and spent == total

    segments = []
    horizontal_indices = []
    for i in range(n - 1):
        delta = plan[i + 1] - plan[i]
        is_horizontal = delta == 0
        if is_horizontal:
            horizontal_indices.append(i + 1)
        segments.append(
            {"index": i + 1, "from": i + 1, "to": i + 2,
             "delta": delta, "horizontal": is_horizontal}
        )

    return {
        "feasible": True,
        "ground": list(ground),
        "plan": plan,
        "total_adjustment": total,
        "segments": segments,
        "horizontal_indices": horizontal_indices,
    }
