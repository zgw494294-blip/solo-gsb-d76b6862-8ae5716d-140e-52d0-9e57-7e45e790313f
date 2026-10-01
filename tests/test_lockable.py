"""测点可锁定标高分析测试：定点用例 + 穷举对拍。"""

from __future__ import annotations

import random

import pytest

from app.lockable import _compress_intervals, lockable_analysis
from app.planner import (
    H_MAX,
    H_MIN,
    MAX_GRADE,
    N_MAX,
    WINDOW,
    PlanError,
    plan_trail,
)


def brute_force_lockable_sets(ground):
    """穷举全部可行方案，收集每个测点出现过的方案标高（精确集合）。"""
    n = len(ground)
    sets = [set() for _ in range(n)]
    plan = [0] * n
    plan[0] = ground[0]
    plan[n - 1] = ground[n - 1]

    def prefix_windows_ok(upto):
        """已放下到段 upto（0-based）时，已完整的 5 段窗口都需含水平段。"""
        for k in range(upto - WINDOW + 2):
            if not any(plan[k + j] == plan[k + j + 1] for j in range(WINDOW)):
                return False
        return True

    def dfs(idx):
        if idx == n:
            for i, h in enumerate(plan):
                sets[i].add(h)
            return
        if idx == n - 1:
            candidates = (plan[n - 1],)
        else:
            prev = plan[idx - 1]
            candidates = range(
                max(H_MIN, prev - MAX_GRADE),
                min(H_MAX, prev + MAX_GRADE) + 1,
            )
        prev = plan[idx - 1]
        for h in candidates:
            if abs(h - prev) > MAX_GRADE:
                continue
            plan[idx] = h
            if prefix_windows_ok(idx - 1):
                dfs(idx + 1)

    dfs(1)
    return sets


def expand(intervals):
    """把压缩区间还原为整数集合。"""
    s = set()
    for lo, hi in intervals:
        s.update(range(lo, hi + 1))
    return s


def assert_intervals_well_formed(intervals):
    """区间必须升序、互不相邻（相邻区间至少隔一个缺失值）、在值域内。"""
    for lo, hi in intervals:
        assert H_MIN <= lo <= hi <= H_MAX
    for k in range(1, len(intervals)):
        assert intervals[k][0] >= intervals[k - 1][1] + 2


def assert_matches_bruteforce(ground):
    r = lockable_analysis(ground)
    bf = brute_force_lockable_sets(ground)
    assert r["feasible"] == bool(bf[0])
    if not r["feasible"]:
        assert all(not s for s in bf)
        assert "无解" in r["reason"]
        return
    assert r["ground"] == list(ground)
    assert len(r["points"]) == len(ground)
    for i, point in enumerate(r["points"]):
        assert point["index"] == i + 1
        assert_intervals_well_formed(point["intervals"])
        assert expand(point["intervals"]) == bf[i]


# ---------- 定点用例 ----------

def test_n3_middle_point_full_slope_range():
    # n=3 不触发 5 段窗口；中点只受两侧高差 ≤ 8 限制
    r = lockable_analysis([0, 8, 0])
    assert r["feasible"]
    assert [p["intervals"] for p in r["points"]] == [
        [[0, 0]], [[0, 8]], [[0, 0]],
    ]


def test_flat_ground_endpoints_fixed():
    r = lockable_analysis([100] * 6)
    assert r["feasible"]
    assert r["points"][0]["intervals"] == [[100, 100]]
    assert r["points"][-1]["intervals"] == [[100, 100]]
    # 中间点可在坡度与水平段规则内偏离地面
    for point in r["points"][1:-1]:
        assert [100, 100] in point["intervals"] or any(
            lo <= 100 <= hi for lo, hi in point["intervals"]
        )


def test_sample_intervals_have_gaps_not_filled():
    # 该用例第 2 点的可锁定集合为 {100, 106, 107, 108}：
    # 必须给出两个区间，不能用 [100, 108] 填平 101～105 的空档
    ground = [100, 107, 114, 119, 124, 130]
    r = lockable_analysis(ground)
    assert r["feasible"]
    assert r["points"][1]["intervals"] == [[100, 100], [106, 108]]
    for point in r["points"]:
        assert_intervals_well_formed(point["intervals"])


def test_infeasible_grade_limit():
    r = lockable_analysis([0, 100, 100, 100, 100, 41])
    assert not r["feasible"]
    assert "无解" in r["reason"]
    assert "points" not in r


def test_infeasible_window_rule():
    r = lockable_analysis([0, 8, 16, 24, 32, 40])
    assert not r["feasible"]
    assert "水平段" in r["reason"]


def test_optimal_plan_values_are_lockable():
    # 最优方案每一点的取值必然可锁定（最优方案本身就是见证）
    ground = [100, 107, 114, 119, 124, 130]
    plan = plan_trail(ground)["plan"]
    r = lockable_analysis(ground)
    for i, point in enumerate(r["points"]):
        assert any(lo <= plan[i] <= hi for lo, hi in point["intervals"])


# ---------- 区间压缩 ----------

@pytest.mark.parametrize(
    "values,expected",
    [
        ([], []),
        ([5], [[5, 5]]),
        ([1, 2, 3], [[1, 3]]),
        ([1, 3], [[1, 1], [3, 3]]),          # 互不相邻：2 缺失
        ([1, 2, 4, 5, 7], [[1, 2], [4, 5], [7, 7]]),
        ([0, 2, 3, 4, 200], [[0, 0], [2, 4], [200, 200]]),
    ],
)
def test_compress_intervals(values, expected):
    assert _compress_intervals(values) == expected


# ---------- 非法输入（复用 planner 校验） ----------

@pytest.mark.parametrize(
    "ground",
    [
        [],
        [1, 2],
        [1] * 17,
        [0, -1, 0],
        [0, 201, 0],
        [0, 1.5, 0],
        [0, "8", 0],
        [0, True, 0],
    ],
)
def test_invalid_inputs(ground):
    with pytest.raises(PlanError):
        lockable_analysis(ground)


# ---------- 穷举对拍 ----------

def test_bruteforce_band_random():
    """窄带地面（0～30）、n ≤ 6，全值域穷举对拍。"""
    rng = random.Random(20260929)
    for _ in range(30):
        n = rng.randint(3, 6)
        ground = [rng.randint(0, 30) for _ in range(n)]
        assert_matches_bruteforce(ground)


def test_bruteforce_full_domain_random():
    """完整 0～200 值域、n ≤ 5，穷举对拍。"""
    rng = random.Random(99)
    for _ in range(12):
        n = rng.randint(3, 5)
        ground = [rng.randint(H_MIN, H_MAX) for _ in range(n)]
        assert_matches_bruteforce(ground)


def test_bruteforce_fixed_cases():
    for ground in (
        [0, 0, 0, 8, 8, 0],
        [8, 8, 0, 0, 8, 8, 0],
        [0, 8, 0],
        [200, 192, 200],
        [0, 0, 0, 0, 0, 0],
        [5, 13, 5, 13, 5, 13, 5],
    ):
        assert_matches_bruteforce(list(ground))


def test_consistency_with_planner_at_max_size():
    """16 点规模：与最优求解器可行性一致，且最优方案逐点可锁定。"""
    rng = random.Random(4242)
    cases = [
        [10 + 11 * i for i in range(N_MAX)],          # 坡度 11，无解
        [100 + (7 if i % 2 else -7) for i in range(N_MAX)],
    ]
    cases += [
        [rng.randint(90, 110) for _ in range(N_MAX)] for _ in range(3)
    ]
    for ground in cases:
        plan_res = plan_trail(ground)
        lock_res = lockable_analysis(ground)
        assert lock_res["feasible"] == plan_res["feasible"]
        if plan_res["feasible"]:
            assert len(lock_res["points"]) == N_MAX
            for i, point in enumerate(lock_res["points"]):
                assert_intervals_well_formed(point["intervals"])
                assert any(
                    lo <= plan_res["plan"][i] <= hi
                    for lo, hi in point["intervals"]
                )
