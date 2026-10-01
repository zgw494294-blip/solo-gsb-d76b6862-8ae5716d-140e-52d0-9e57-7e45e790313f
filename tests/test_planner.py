"""规划算法测试：定点用例 + 小规模穷举对拍。"""

from __future__ import annotations

import itertools

import pytest

from app.planner import (
    H_MAX,
    H_MIN,
    MAX_GRADE,
    N_MAX,
    N_MIN,
    WINDOW,
    PlanError,
    plan_trail,
)


def brute_force_optimal(ground, h_max=H_MAX):
    """穷举求 (feasible, plan, cost)，用于对拍。"""
    n = len(ground)
    best = None  # (cost, plan)
    for p in brute_force_iter(ground, h_max):
        cost = sum(abs(p[i] - ground[i]) for i in range(n))
        key = (cost, tuple(p))
        if best is None or key < best:
            best = key
    if best is None:
        return False, None, None
    return True, list(best[1]), best[0]


def _windows_ok(plan):
    nseg = len(plan) - 1
    if nseg < WINDOW:
        return True
    return all(
        any(plan[k + j] == plan[k + j + 1] for j in range(WINDOW))
        for k in range(nseg - WINDOW + 1)
    )


def brute_force_iter(ground, h_max):
    """按约束生成全部可行方案（剪枝搜索）。"""
    n = len(ground)
    plan = [0] * n
    plan[0], plan[-1] = ground[0], ground[-1]

    def prefix_windows_ok(upto):
        """已放下到段 upto（0-based）时，已完整的 5 段窗口都需含水平段。"""
        for k in range(upto - WINDOW + 2):   # 完整窗口起点 k：k+WINDOW-1 <= upto
            if not any(plan[k + j] == plan[k + j + 1] for j in range(WINDOW)):
                return False
        return True

    def dfs(idx):
        if idx == n:
            if _windows_ok(plan):
                yield plan.copy()
            return
        prev = plan[idx - 1]
        candidates = (
            range(H_MIN, h_max + 1)
            if idx < n - 1
            else [plan[n - 1]]
        )
        for h in candidates:
            if abs(h - prev) <= MAX_GRADE:
                plan[idx] = h
                if prefix_windows_ok(idx - 1):
                    yield from dfs(idx + 1)

    yield from dfs(1)


def assert_matches_bruteforce(ground, h_max=40):
    r = plan_trail(ground)
    feasible, bf_plan, bf_cost = brute_force_optimal(ground, h_max)
    assert r["feasible"] == feasible
    if feasible:
        assert r["plan"] == bf_plan
        assert r["total_adjustment"] == bf_cost
        assert r["plan"][0] == ground[0]
        assert r["plan"][-1] == ground[-1]
        for i in range(len(ground) - 1):
            assert abs(r["plan"][i + 1] - r["plan"][i]) <= MAX_GRADE
        horiz = [s["horizontal"] for s in r["segments"]]
        nseg = len(ground) - 1
        if nseg >= WINDOW:
            for k in range(nseg - WINDOW + 1):
                assert any(horiz[k:k + WINDOW])


# ---------- 定点用例 ----------

def test_flat_ground():
    r = plan_trail([100] * 6)
    assert r["feasible"]
    assert r["plan"] == [100] * 6
    assert r["total_adjustment"] == 0
    assert all(s["horizontal"] for s in r["segments"])


def test_endpoint_too_far():
    # 首尾差 41，5 段最多消化 40
    ground = [0, 100, 100, 100, 100, 41]
    r = plan_trail(ground)
    assert not r["feasible"]
    assert "无解" in r["reason"]


def test_strictly_rising_ground_infeasible():
    # 每点地面抬升 8，最省调整的方案会被强制全部 +8，没有水平段
    ground = [0, 8, 16, 24, 32, 40]
    r = plan_trail(ground)
    assert not r["feasible"]
    assert "水平段" in r["reason"]


def test_sample_has_horizontal_and_cost():
    ground = [100, 107, 114, 119, 124, 130]
    r = plan_trail(ground)
    assert r["feasible"]
    assert r["plan"][0] == 100 and r["plan"][-1] == 130
    assert len(r["horizontal_indices"]) >= 1
    for s in r["segments"]:
        assert abs(s["delta"]) <= 8
    assert r["total_adjustment"] == sum(
        abs(p - g) for p, g in zip(r["plan"], ground)
    )


def test_lex_tiebreak_n4():
    # n=4 时不触发 5 段窗口；cost=2 的方案至少有
    # [0,0,8,8] 与 [0,1,9,8]，字典序最小应为前者
    ground = [0, 0, 10, 8]
    r = plan_trail(ground)
    assert r["feasible"]
    assert r["plan"] == [0, 0, 8, 8]
    assert r["total_adjustment"] == 2


def test_n3_tiny_case():
    assert_matches_bruteforce([0, 8, 0], h_max=H_MAX)


# ---------- 非法输入 ----------

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
        plan_trail(ground)


# ---------- 穷举对拍 ----------

def test_bruteforce_small_domain_random():
    """小值域、限坡度 8 下随机地面，穷举校验最优值与字典序。"""
    import random

    rng = random.Random(20260928)
    h_max = 8
    checked = 0
    for _ in range(60):
        n = rng.randint(N_MIN, 8)
        ground = [rng.randint(0, h_max) for _ in range(n)]
        assert_matches_bruteforce(ground, h_max=h_max)
        checked += 1
    assert checked == 60


def test_bruteforce_full_domain_band():
    """完整 0~200 值域，但地面限制在宽 8 的带状内，穷举可行。"""
    import random

    rng = random.Random(42)
    cases = [
        [0, 0, 0, 8, 8, 0],
        [0, 2, 5, 8, 6, 4, 2, 0],
        [8, 8, 0, 0, 8, 8, 0],
    ]
    for ground in cases:
        assert_matches_bruteforce(ground, h_max=H_MAX)
    for _ in range(20):
        n = rng.randint(3, 7)
        base = rng.randint(0, H_MAX - 8)
        ground = [base + rng.randint(0, 8) for _ in range(n)]
        assert_matches_bruteforce(ground, h_max=H_MAX)


def test_max_point_count_runs():
    """16 点边界规模必须能快速算出且自洽。"""
    import time

    ground = [10 + 11 * i for i in range(N_MAX)]  # 坡度 11，需要大量调整
    start = time.perf_counter()
    r = plan_trail(ground)
    elapsed = time.perf_counter() - start
    assert elapsed < 3
    if r["feasible"]:
        assert r["plan"][0] == ground[0]
        assert r["plan"][-1] == ground[-1]
        horiz = [s["horizontal"] for s in r["segments"]]
        for k in range(len(ground) - 1 - WINDOW + 1):
            assert any(horiz[k:k + WINDOW])
