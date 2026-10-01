"""土方量积分与沿中点调配核算测试。"""

from __future__ import annotations

import pytest

from app.earthwork import (
    compute_earthwork,
    plan_with_earthwork,
    validate_width,
)
from app.planner import PlanError, plan_trail

APPROX = pytest.approx


# ---------- 净宽校验 ----------

@pytest.mark.parametrize("w", [0.5, 10.0, 1.5, 2, 0.55, 9.99, 10])
def test_valid_width(w):
    assert validate_width(w) == float(w)


@pytest.mark.parametrize(
    "w",
    [0.49, 10.01, 0, -1.5, 1.555, "1.5", True, None, [], float("nan"),
     float("inf"), float("-inf")],
)
def test_invalid_width(w):
    with pytest.raises(PlanError):
        validate_width(w)


# ---------- 分段积分（手算几何） ----------

def test_flat_equal_no_earthwork():
    # 地面＝方案：挖填均为 0，所有流量为 0
    ew = compute_earthwork([100, 100, 100], [100, 100, 100], 1.0)
    assert ew["total_cut"] == 0
    assert ew["total_fill"] == 0
    assert ew["total_net"] == 0
    assert all(s["cut"] == 0 and s["fill"] == 0 and s["net"] == 0
               for s in ew["segments"])
    assert all(f["flow"] == 0 for f in ew["flows"])
    assert ew["transport_work"] == 0


def test_trapezoid_single_segment():
    # d0 = 0.10 m（挖）, d1 = 0.04 m（挖），净宽 2 m，长 1 m
    # 挖方 = 2 × (0.10 + 0.04)/2 = 0.14 m³
    ew = compute_earthwork([110, 104], [100, 100], 2.0)
    seg = ew["segments"][0]
    assert seg["cut"] == APPROX(0.14)
    assert seg["fill"] == 0
    assert seg["net"] == APPROX(0.14)
    assert seg["cross_at"] is None
    assert ew["total_net"] == APPROX(0.14)
    # 唯一一段净余：q0 = -0.14（回流起点运出），距离 0.5
    assert ew["flows"][0]["flow"] == APPROX(-0.14)
    assert ew["transport_work"] == APPROX(0.14 * 0.5)


def test_fill_trapezoid_negative_flow():
    # d0 = -0.02, d1 = -0.08（填），净宽 1
    # 填方 = (0.02 + 0.08)/2 = 0.05 m³，全线净缺，q0 = +0.05
    ew = compute_earthwork([98, 92], [100, 100], 1.0)
    seg = ew["segments"][0]
    assert seg["cut"] == 0
    assert seg["fill"] == APPROX(0.05)
    assert seg["net"] == APPROX(-0.05)
    assert ew["flows"][0]["flow"] == APPROX(0.05)
    assert ew["transport_work"] == APPROX(0.05 * 0.5)


def test_sign_change_split_triangles():
    # d0 = +0.10（挖）, d1 = -0.06（填），净宽 0.5
    # 零点 x = 0.10/0.16 = 0.625
    # 挖三角 0.5×0.10×0.625 = 0.03125 m²，填三角 0.5×0.06×0.375 = 0.01125 m²
    ew = compute_earthwork([110, 94], [100, 100], 0.5)
    seg = ew["segments"][0]
    assert seg["cross_at"] == APPROX(0.625)
    assert seg["cut"] == APPROX(0.5 * 0.03125)
    assert seg["fill"] == APPROX(0.5 * 0.01125)
    assert seg["net"] == APPROX(0.5 * (0.03125 - 0.01125))


def test_sign_change_reversed():
    # d0 = -0.06（填）, d1 = +0.10（挖）
    ew = compute_earthwork([94, 110], [100, 100], 0.5)
    seg = ew["segments"][0]
    assert seg["cross_at"] == APPROX(0.375)
    assert seg["fill"] == APPROX(0.5 * 0.01125)
    assert seg["cut"] == APPROX(0.5 * 0.03125)


def test_zero_at_endpoint_no_split():
    # d0 = 0, d1 = 0.06：退化为三角形挖方，不应误判为变号
    ew = compute_earthwork([100, 106], [100, 100], 1.0)
    seg = ew["segments"][0]
    assert seg["cross_at"] is None
    assert seg["cut"] == APPROX(0.03)
    assert seg["fill"] == 0


# ---------- 沿中点调配链 ----------

def test_three_segment_chain_balance():
    # net1 = +0.05（余）, net2 = -0.02（缺）, net3 = -0.03（缺），全线平衡
    # 净宽 1 m：d0=0.10→d1=0（挖三角 .05）；d1=0→d2=-0.04（填三角 .02）；
    # d2=-0.04→d3=-0.02（填梯形 .03）
    ground = [110, 100, 96, 98]
    plan = [100, 100, 100, 100]
    ew = compute_earthwork(ground, plan, 1.0)
    nets = [s["net"] for s in ew["segments"]]
    assert nets == APPROX([0.05, -0.02, -0.03])
    assert ew["total_net"] == APPROX(0.0)

    flows = [f["flow"] for f in ew["flows"]]
    # q3=0; q2=q3-net3=+0.03; q1=q2-net2=+0.05; q0=q1-net1=0
    assert flows == APPROX([0.0, 0.05, 0.03])
    assert ew["transport_work"] == APPROX(0.0 * 0.5 + 0.05 * 1 + 0.03 * 1)
    # 链路几何
    assert ew["flows"][0]["distance"] == 0.5
    assert [f["distance"] for f in ew["flows"][1:]] == [1, 1]
    assert ew["flows"][1]["from"] == "midpoint:1"
    assert ew["flows"][1]["to"] == "midpoint:2"


def test_cumulative_net_sequence():
    ground = [110, 100, 96, 98]
    plan = [100, 100, 100, 100]
    ew = compute_earthwork(ground, plan, 1.0)
    cums = [s["cumulative_net"] for s in ew["segments"]]
    assert cums == APPROX([0.05, 0.03, 0.0])
    # q[k] = 前 k 段累计净量 − 全线总净量（k=0 时前缀为 0）
    prefix = 0.0
    for k, f in enumerate(ew["flows"]):
        assert f["flow"] == APPROX(prefix - ew["total_net"])
        prefix = cums[k]


def test_width_scales_volumes_and_work_linearly():
    ground, plan = [110, 96, 102], [100, 100, 100]
    ew1 = compute_earthwork(ground, plan, 1.0)
    ew2 = compute_earthwork(ground, plan, 2.37)
    ratio = 2.37
    assert ew2["total_cut"] == APPROX(ew1["total_cut"] * ratio)
    assert ew2["total_fill"] == APPROX(ew1["total_fill"] * ratio)
    assert ew2["transport_work"] == APPROX(ew1["transport_work"] * ratio)
    # 交点位置与净宽无关
    for a, b in zip(ew1["segments"], ew2["segments"]):
        assert (a["cross_at"] is None) == (b["cross_at"] is None)
        if a["cross_at"] is not None:
            assert b["cross_at"] == APPROX(a["cross_at"])


def test_mismatched_lengths():
    with pytest.raises(PlanError):
        compute_earthwork([100, 100, 100], [100, 100], 1.0)


# ---------- 与求解器集成 ----------

def test_plan_with_earthwork_feasible():
    r = plan_with_earthwork([100, 107, 114, 119, 124, 130], 1.5)
    assert r["feasible"] is True
    assert r["width"] == 1.5
    ew = r["earthwork"]
    nseg = len(r["plan"]) - 1
    assert len(ew["segments"]) == nseg
    assert len(ew["flows"]) == nseg

    # 与分段数据自洽
    assert ew["total_cut"] == APPROX(sum(s["cut"] for s in ew["segments"]))
    assert ew["total_fill"] == APPROX(sum(s["fill"] for s in ew["segments"]))
    assert ew["total_net"] == APPROX(ew["total_cut"] - ew["total_fill"])

    # 独立按线性公式重算每段体积
    w = 1.5
    for i, s in enumerate(ew["segments"]):
        d0 = (r["ground"][i] - r["plan"][i]) / 100
        d1 = (r["ground"][i + 1] - r["plan"][i + 1]) / 100
        cut = fill = 0.0
        if d0 * d1 < 0:
            x = d0 / (d0 - d1)
            cut += 0.5 * max(d0, 0) * x + 0.5 * max(d1, 0) * (1 - x)
            fill += 0.5 * max(-d0, 0) * x + 0.5 * max(-d1, 0) * (1 - x)
        else:
            cut = 0.5 * (max(d0, 0) + max(d1, 0))
            fill = 0.5 * (max(-d0, 0) + max(-d1, 0))
        assert s["cut"] == APPROX(cut * w)
        assert s["fill"] == APPROX(fill * w)

    # 逐中点质量守恒：q[k-1] + net_k = q[k]
    nets = [s["net"] for s in ew["segments"]]
    q = [f["flow"] for f in ew["flows"]] + [0.0]
    for i in range(nseg):
        assert q[i] + nets[i] == APPROX(q[i + 1])

    # 运量距离重算
    work = sum(abs(f["flow"]) * f["distance"] for f in ew["flows"])
    assert ew["transport_work"] == APPROX(work)
    # 起点流量 = -全线净量（唯一对外通道）
    assert ew["flows"][0]["flow"] == APPROX(-ew["total_net"])


def test_plan_with_earthwork_infeasible_has_no_earthwork():
    r = plan_with_earthwork([0, 8, 16, 24, 32, 40], 1.5)
    assert r["feasible"] is False
    assert "reason" in r
    assert "earthwork" not in r
    assert "width" not in r


def test_plan_with_earthwork_bad_width():
    with pytest.raises(PlanError):
        plan_with_earthwork([100, 100, 100], 0.1)


# ---------- 小规模穷举：调配守恒与积分一致 ----------

def test_exhaustive_conservation():
    """枚举小段方案，校验交点拆分积分与不分段的有向面积公式一致。"""
    import random

    rng = random.Random(20260929)
    for _ in range(300):
        n = rng.randint(3, 6)
        ground = [rng.randint(0, 200) for _ in range(n)]
        plan = [rng.randint(0, 200) for _ in range(n)]
        width = round(rng.uniform(0.5, 10), 2)
        ew = compute_earthwork(ground, plan, width)

        total_cut = total_fill = 0.0
        for i in range(n - 1):
            d0 = (ground[i] - plan[i]) / 100
            d1 = (ground[i + 1] - plan[i + 1]) / 100
            # 直接用端点三角法重算参考值
            if d0 * d1 < 0:
                x = d0 / (d0 - d1)
                c = 0.5 * max(d0, 0) * x + 0.5 * max(d1, 0) * (1 - x)
                f = 0.5 * max(-d0, 0) * x + 0.5 * max(-d1, 0) * (1 - x)
            else:
                c = 0.5 * (max(d0, 0) + max(d1, 0))
                f = 0.5 * (max(-d0, 0) + max(-d1, 0))
            cut_ref = width * c
            fill_ref = width * f
            s = ew["segments"][i]
            assert s["cut"] == APPROX(cut_ref)
            assert s["fill"] == APPROX(fill_ref)
            total_cut += s["cut"]
            total_fill += s["fill"]

        assert ew["total_cut"] == APPROX(total_cut)
        assert ew["total_fill"] == APPROX(total_fill)

        # 质量守恒链
        nets = [s["net"] for s in ew["segments"]]
        q_next = 0.0
        for i in range(n - 2, -1, -1):
            q_prev = q_next - nets[i]
            assert ew["flows"][i]["flow"] == APPROX(q_prev)
            q_next = q_prev


def test_exhaustive_on_solver_outputs():
    """对求解器实际输出做守恒与量纲检查（净宽变化时结果按比例）。"""
    import random

    rng = random.Random(7)
    cases = [
        [100, 107, 114, 119, 124, 130],
        [0, 0, 0, 8, 8, 0],
        [200, 194, 188, 194, 200, 200],
    ]
    cases += [[rng.randint(0, 200) for _ in range(rng.randint(3, 12))]
              for _ in range(10)]
    for ground in cases:
        base = plan_trail(ground)
        if not base["feasible"]:
            continue
        for width in (0.5, 1.5, 9.99):
            ew = compute_earthwork(base["ground"], base["plan"], width)
            for s in ew["segments"]:
                assert s["cut"] >= -1e-12 and s["fill"] >= -1e-12
                assert s["net"] == APPROX(s["cut"] - s["fill"])
            assert ew["flows"][0]["flow"] == APPROX(-ew["total_net"])
            assert ew["transport_work"] >= -1e-12
