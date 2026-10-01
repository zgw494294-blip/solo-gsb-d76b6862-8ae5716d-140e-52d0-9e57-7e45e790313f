"""步道坡线规划台 —— 土方量积分与沿中点调配核算。

模型
----
1. 相邻测点等间距 1 m。可行方案中，每段地面标高与方案标高均按线性变化，
   定义段内高差（米）::

       d(x) = (ground(x) - plan(x)) / 100,   x ∈ [0, 1]

   ``d > 0`` 为挖深、``d < 0`` 为填深。``d`` 在段内为线性函数：
   两端同号（含零）时按梯形积分；两端异号时在零点把该段拆成两个
   三角形分别积分。体积 = 净宽 × 截面积分，结果以立方米计。

2. 每段土方视为集中在该段中点（位置 0.5, 1.5, … m）。同段挖方与填方
   先抵消，净量 ``net = cut - fill``（正为余、负为缺）。

3. 调配网络为一条链：起点（x = 0）— 段1中点 — 段2中点 — … — 段m中点。
   全线净余/净缺只能经起点进出，末端不对外。设边流量 ``q[k]`` 为离开
   第 k 个中点朝终点方向的有向流量（``q[0]`` 为起点至段1中点），由
   各中点质量守恒与末端 ``q[m] = 0`` 唯一确定::

       q[k-1] = q[k] - net_k

   ``q[0] > 0`` 表示经起点向线内运入（全线净缺），``q[0] < 0`` 表示
   余量回流至起点运出（全线净余）。

4. 运量距离 = Σ |q| × 距离（起点至首段中点 0.5 m，相邻中点间 1 m），
   单位 m³·m。计算全程使用原始浮点结果，不提前取整。
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from .planner import PlanError, plan_trail

WIDTH_MIN = 0.5           # 步道净宽下限（米）
WIDTH_MAX = 10.0          # 步道净宽上限（米）
WIDTH_DECIMALS = 2        # 净宽最多两位小数
SEGMENT_LENGTH = 1.0      # 相邻测点间距 / 相邻中点间距（米）
START_TO_MIDPOINT = 0.5   # 起点到首段中点的距离（米）


def validate_width(width: Any) -> float:
    """校验步道净宽：数字、0.5～10 米、最多两位小数。"""
    if isinstance(width, bool) or not isinstance(width, (int, float)):
        raise PlanError("步道净宽必须是数字（单位米）。")
    w = float(width)
    if not math.isfinite(w):
        raise PlanError("步道净宽必须是有限数字。")
    if not (WIDTH_MIN <= w <= WIDTH_MAX):
        raise PlanError(
            f"步道净宽必须在 {WIDTH_MIN:g}～{WIDTH_MAX:g} 米之间"
            f"（当前 {w:g} 米）。"
        )
    dec = Decimal(str(w))
    if dec.as_tuple().exponent < -WIDTH_DECIMALS:
        raise PlanError("步道净宽最多保留两位小数。")
    return w


def _cut_fill_areas(d0: float, d1: float) -> Tuple[float, float, float | None]:
    """线性高差在一个 1 m 段上的挖、填截面积（m²）。

    ``d0``、``d1`` 为段起、止点的（地面−方案）高差（米，正挖负填）。
    两端异号时在零点拆成两个三角形分别积分；返回
    ``(挖方面积, 填方面积, 变号交点距段起点的距离)``，无变号时交点为 ``None``。
    """
    if d0 * d1 < 0.0:
        # d(x) = d0 + (d1 - d0) x = 0
        x = d0 / (d0 - d1)
        cut = 0.0
        fill = 0.0
        if d0 > 0.0:
            cut += 0.5 * d0 * x
        else:
            fill += 0.5 * (-d0) * x
        if d1 > 0.0:
            cut += 0.5 * d1 * (1.0 - x)
        else:
            fill += 0.5 * (-d1) * (1.0 - x)
        return cut, fill, x

    cut = 0.5 * (max(d0, 0.0) + max(d1, 0.0))
    fill = 0.5 * (max(-d0, 0.0) + max(-d1, 0.0))
    return cut, fill, None


def compute_earthwork(ground: List[int], plan: List[int], width: Any) -> Dict[str, Any]:
    """计算分段挖填、累计净量与沿中点调配流量。

    入参 ``ground``/``plan`` 为厘米标高（长度一致、首尾相等即可，调用方
    通常传入 :func:`app.planner.plan_trail` 的可行结果），``width`` 为
    步道净宽（米）。返回结构::

        {
          "width": float,
          "total_cut", "total_fill", "total_net": float,   # m³
          "segments": [{"index", "cut", "fill", "net",
                        "cumulative_net", "cross_at"}],
          "flows": [{"index", "from", "to", "flow",
                     "distance"}],                         # m³, m
          "transport_work": float,                         # m³·m
        }
    """
    w = validate_width(width)
    if len(ground) != len(plan):
        raise PlanError("地面标高与方案标高数量不一致，无法核算土方。")

    m = len(plan) - 1
    segments: List[Dict[str, Any]] = []
    nets: List[float] = []
    total_cut = total_fill = 0.0
    cumulative = 0.0

    for i in range(m):
        d0 = (ground[i] - plan[i]) / 100.0
        d1 = (ground[i + 1] - plan[i + 1]) / 100.0
        cut_area, fill_area, cross_at = _cut_fill_areas(d0, d1)
        cut = cut_area * w + 0.0     # + 0.0 归一化浮点 -0
        fill = fill_area * w + 0.0
        net = cut - fill
        cumulative += net
        total_cut += cut
        total_fill += fill
        nets.append(net)
        segments.append({
            "index": i + 1,
            "cut": cut,
            "fill": fill,
            "net": net,
            "cumulative_net": cumulative,
            "cross_at": cross_at,
        })

    total_net = total_cut - total_fill

    # q[k]：离开第 k 个中点朝终点方向的有向流量；
    # q[0] 为起点至段1中点，q[m] 为越过末端的虚拟边（末端封闭，恒为 0）。
    q = [0.0] * (m + 1)
    for j in range(m, 0, -1):
        q[j - 1] = q[j] - nets[j - 1]

    flows: List[Dict[str, Any]] = []
    transport_work = 0.0
    for k in range(m):
        if k == 0:
            frm, to, distance = "start", "midpoint:1", START_TO_MIDPOINT
        else:
            frm, to, distance = f"midpoint:{k}", f"midpoint:{k + 1}", SEGMENT_LENGTH
        flows.append({
            "index": k + 1,
            "from": frm,
            "to": to,
            "flow": q[k],
            "distance": distance,
        })
        transport_work += abs(q[k]) * distance

    return {
        "width": w,
        "total_cut": total_cut,
        "total_fill": total_fill,
        "total_net": total_net,
        "segments": segments,
        "flows": flows,
        "transport_work": transport_work,
    }


def plan_with_earthwork(ground: List[int], width: Any) -> Dict[str, Any]:
    """复用现有坡线求解器，可行时附加土方调配核算结果。

    非法输入（含净宽非法）抛 :class:`PlanError`；无可行方案时原样返回
    ``{"feasible": False, "reason": ...}``，不附土方核算。
    """
    w = validate_width(width)
    result = plan_trail(ground)
    if not result.get("feasible"):
        return result
    result["width"] = w
    result["earthwork"] = compute_earthwork(result["ground"], result["plan"], w)
    return result
