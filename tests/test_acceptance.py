"""施工验收复核测试（acceptance_review 与 /api/acceptance）。"""

import pytest
from fastapi.testclient import TestClient

from app.acceptance import acceptance_review
from app.main import app
from app.planner import plan_trail

client = TestClient(app)

GROUND = [100, 107, 114, 119, 124, 130]
PLAN = plan_trail(GROUND)["plan"]


# ---------- 通过用例 ----------

def test_pass_when_measured_equals_plan():
    r = acceptance_review(GROUND, PLAN, list(PLAN), 5)
    assert r["passed"] is True
    assert r["deviations"] == [0] * 6
    assert r["over_tolerance_points"] == []
    assert r["over_grade_segments"] == []
    assert r["no_horizontal_windows"] == []
    assert r["endpoint_mismatches"] == []
    assert r["tolerance"] == 5


def test_pass_within_tolerance():
    ground = [50] * 6
    plan = [50] * 6
    measured = [50, 53, 50, 50, 48, 50]
    r = acceptance_review(ground, plan, measured, 3)
    assert r["passed"] is True
    assert r["deviations"] == [0, 3, 0, 0, -2, 0]
    assert r["over_tolerance_points"] == []


def test_tolerance_boundary_equal_is_allowed():
    # |偏差| == 容差不算超
    ground = [50] * 6
    plan = [50] * 6
    measured = [50, 50, 55, 50, 50, 50]
    r = acceptance_review(ground, plan, measured, 5)
    assert r["passed"] is True
    assert r["over_tolerance_points"] == []


def test_tolerance_zero_requires_exact():
    measured = list(PLAN)
    measured[2] -= 1
    r = acceptance_review(GROUND, PLAN, measured, 0)
    assert r["passed"] is False
    bad = r["over_tolerance_points"]
    assert len(bad) == 1
    assert bad[0]["index"] == 3
    assert bad[0]["plan"] == PLAN[2]
    assert bad[0]["measured"] == PLAN[2] - 1
    assert bad[0]["deviation"] == -1


# ---------- 超容差 ----------

def test_over_tolerance_points():
    measured = list(PLAN)
    measured[0] += 6
    measured[3] -= 7
    r = acceptance_review(GROUND, PLAN, measured, 5)
    idx = {p["index"]: p for p in r["over_tolerance_points"]}
    assert set(idx) == {1, 4}
    assert idx[1]["deviation"] == 6
    assert idx[4]["deviation"] == -7
    assert r["passed"] is False


# ---------- 实测高差超 8 ----------

def test_over_grade_segment():
    measured = list(PLAN)
    measured[2] = measured[1] + 10
    r = acceptance_review(GROUND, PLAN, measured, 20)
    segs = r["over_grade_segments"]
    assert len(segs) == 1
    assert segs[0]["index"] == 2
    assert segs[0]["from"] == 2 and segs[0]["to"] == 3
    assert segs[0]["delta"] == 10
    assert r["passed"] is False


def test_grade_boundary_8_allowed():
    ground = [50] * 6
    plan = [50] * 6
    measured = [50, 58, 58, 58, 50, 50]   # 最大段差恰好 8，含水平段
    r = acceptance_review(ground, plan, measured, 20)
    assert r["over_grade_segments"] == []


# ---------- 连续 5 段无水平 ----------

def test_no_horizontal_window_reported():
    # 冻结方案本身必须含水平段：8 个点（7 段）的全斜线不可行，
    # 取一个含水平段的冻结方案，把实测改成连续 7 段都非水平（每段 +1）。
    ground = [50, 50, 50, 50, 50, 50, 50, 50]
    plan = plan_trail(ground)["plan"]
    measured = [50, 51, 52, 53, 54, 55, 56, 57]
    r = acceptance_review(ground, plan, measured, 20)
    windows = r["no_horizontal_windows"]
    # 7 段共有 3 个连续 5 段窗口（1-5、2-6、3-7），全部无水平
    assert [w["from_segment"] for w in windows] == [1, 2, 3]
    assert [w["to_segment"] for w in windows] == [5, 6, 7]
    assert windows[0]["segments"] == [1, 2, 3, 4, 5]
    assert windows[1]["index"] == 2
    assert r["passed"] is False


def test_window_with_horizontal_not_reported():
    ground = [50] * 8
    plan = plan_trail(ground)["plan"]
    measured = [50, 51, 52, 53, 53, 54, 55, 56]  # 第 4 段水平
    r = acceptance_review(ground, plan, measured, 20)
    # 两个窗口（2-6 含段 4、3-7 含段 4）中，只有窗口 1-5 之外……
    # 段4 水平 => 窗口 1-5 与 2-6 合规；窗口 3-7（段3,4,5,6,7）含段 4 也合规
    assert r["no_horizontal_windows"] == []


def test_under_five_segments_window_not_triggered():
    # 3 个测点 2 段：约束不触发，即使全非水平也不报窗口
    ground = [0, 4, 0]
    plan = [0, 4, 0]
    measured = [0, 8, 0]   # 段差均为 8，非水平
    r = acceptance_review(ground, plan, measured, 20)
    assert r["no_horizontal_windows"] == []


# ---------- 首尾不匹配 ----------

def test_endpoint_mismatches():
    measured = list(PLAN)
    measured[0] += 1
    measured[-1] -= 2
    r = acceptance_review(GROUND, PLAN, measured, 20)
    mm = r["endpoint_mismatches"]
    assert [p["index"] for p in mm] == [1, 6]
    assert mm[0]["measured"] == PLAN[0] + 1
    assert mm[1]["measured"] == PLAN[-1] - 2
    assert r["passed"] is False


def test_internal_deviation_is_not_endpoint_mismatch():
    measured = list(PLAN)
    measured[2] += 10
    r = acceptance_review(GROUND, PLAN, measured, 20)
    assert r["endpoint_mismatches"] == []


# ---------- 冻结方案复核：不要求最优，但必须合法 ----------

def test_non_optimal_but_feasible_plan_accepted():
    # 全平地面上的可行方案（有水平段、相邻差 ≤ 8、首尾固定），虽非最优
    # （最优是全平），验收仍照常核对，不重新求优。
    ground = [50, 50, 50, 50, 50, 50]
    plan = [50, 52, 52, 51, 50, 50]
    r = acceptance_review(ground, plan, list(plan), 20)
    assert r["passed"] is True


def test_frozen_plan_wrong_length():
    with pytest.raises(Exception) as exc:
        acceptance_review(GROUND, PLAN[:-1], PLAN[:-1], 5)
    assert "数量" in str(exc.value)


def test_frozen_plan_endpoint_must_equal_ground():
    plan = list(PLAN)
    plan[0] += 1
    with pytest.raises(Exception, match="首、尾"):
        acceptance_review(GROUND, plan, list(plan), 5)


def test_frozen_plan_grade_violation():
    ground = [0, 9, 9, 9, 9, 0]
    plan = [0, 9, 9, 9, 9, 0]
    with pytest.raises(Exception, match="高差"):
        acceptance_review(ground, plan, list(plan), 5)


def test_frozen_plan_missing_horizontal_window():
    # 6 点 5 段：首尾固定为 0，5 段全非水平（+1/+1/-1/-1 需要 6 段，
    # 5 段内构造每段 +1：0,1,2,3,4,? 尾必须为 0）。
    # 直接构造 0→…→0 五段均非水平：0,1,2,1,2,0（差：+1,+1,-1,+1,-2）
    ground = [0, 1, 2, 1, 2, 0]
    plan = [0, 1, 2, 1, 2, 0]
    with pytest.raises(Exception, match="水平段"):
        acceptance_review(ground, plan, list(plan), 5)


def test_frozen_plan_value_out_of_range():
    plan = list(PLAN)
    plan[2] = 201
    with pytest.raises(Exception, match="超出范围"):
        acceptance_review(GROUND, plan, list(PLAN), 5)


def test_frozen_plan_not_int():
    plan = list(PLAN)
    plan[2] = 114.5
    with pytest.raises(Exception, match="整数"):
        acceptance_review(GROUND, plan, list(PLAN), 5)


# ---------- 实测 / 容差 / 地面输入校验 ----------

def test_measured_wrong_length():
    with pytest.raises(Exception, match="实测"):
        acceptance_review(GROUND, PLAN, PLAN[:-1], 5)


def test_measured_range():
    measured = list(PLAN)
    measured[1] = -1
    with pytest.raises(Exception, match="实测"):
        acceptance_review(GROUND, PLAN, measured, 5)


def test_measured_not_int():
    measured = [float(v) for v in PLAN]
    with pytest.raises(Exception, match="实测"):
        acceptance_review(GROUND, PLAN, measured, 5)


@pytest.mark.parametrize("tol", [-1, 21, 5.0, "5", True, False, None])
def test_invalid_tolerance(tol):
    with pytest.raises(Exception):
        acceptance_review(GROUND, PLAN, list(PLAN), tol)


def test_invalid_ground_reused():
    with pytest.raises(Exception):
        acceptance_review([0, 1], [0, 1], [0, 1], 5)


# ---------- HTTP 接口 ----------

def test_api_pass():
    resp = client.post(
        "/api/acceptance",
        json={"ground": GROUND, "plan": PLAN, "measured": list(PLAN),
              "tolerance": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert data["deviations"] == [0] * 6
    assert set(data) >= {
        "over_tolerance_points", "over_grade_segments",
        "no_horizontal_windows", "endpoint_mismatches",
    }


def test_api_violations_200():
    measured = list(PLAN)
    measured[1] += 9
    measured[-1] += 1
    resp = client.post(
        "/api/acceptance",
        json={"ground": GROUND, "plan": PLAN, "measured": measured,
              "tolerance": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is False
    assert any(p["index"] == 2 for p in data["over_tolerance_points"])
    assert any(p["index"] == 6 for p in data["endpoint_mismatches"])


@pytest.mark.parametrize("missing", ["ground", "plan", "measured", "tolerance"])
def test_api_missing_field(missing):
    body = {"ground": GROUND, "plan": PLAN, "measured": list(PLAN),
            "tolerance": 5}
    del body[missing]
    resp = client.post("/api/acceptance", json=body)
    assert resp.status_code == 400
    assert resp.json()["feasible"] is False


def test_api_body_not_json():
    resp = client.post(
        "/api/acceptance", content="not-json",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 400


def test_api_invalid_measured_is_422():
    resp = client.post(
        "/api/acceptance",
        json={"ground": GROUND, "plan": PLAN,
              "measured": [1.0] * 6, "tolerance": 5},
    )
    assert resp.status_code == 422
    assert resp.json()["feasible"] is False


def test_api_invalid_tolerance_is_422():
    resp = client.post(
        "/api/acceptance",
        json={"ground": GROUND, "plan": PLAN,
              "measured": list(PLAN), "tolerance": 21},
    )
    assert resp.status_code == 422
    assert "容差" in resp.json()["reason"]


def test_api_frozen_plan_illegal_is_422():
    plan = list(PLAN)
    plan[0] = GROUND[0] + 1
    resp = client.post(
        "/api/acceptance",
        json={"ground": GROUND, "plan": plan,
              "measured": list(PLAN), "tolerance": 5},
    )
    assert resp.status_code == 422
    assert "首、尾" in resp.json()["reason"]


def test_api_non_optimal_plan_still_reviewed():
    # 接口层同样不重新求优：非最优但可行的冻结方案可以复核并通过
    ground = [50] * 6
    plan = [50, 52, 52, 51, 50, 50]
    resp = client.post(
        "/api/acceptance",
        json={"ground": ground, "plan": plan,
              "measured": list(plan), "tolerance": 3},
    )
    assert resp.status_code == 200
    assert resp.json()["passed"] is True
