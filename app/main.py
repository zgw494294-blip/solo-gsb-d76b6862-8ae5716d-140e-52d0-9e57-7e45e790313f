"""步道坡线规划台 —— FastAPI 入口。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .acceptance import acceptance_review
from .earthwork import plan_with_earthwork
from .lockable import lockable_analysis
from .planner import PlanError

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="步道坡线规划台", version="1.3.0")


def _error(message: str, status: int = 422) -> JSONResponse:
    return JSONResponse(status_code=status, content={"feasible": False, "reason": message})


@app.exception_handler(PlanError)
async def plan_error_handler(_request: Request, exc: PlanError) -> JSONResponse:
    return _error(str(exc))


@app.post("/api/plan")
async def api_plan(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return _error("请求体必须是合法的 JSON。", status=400)
    if not isinstance(payload, dict) or "ground" not in payload:
        return _error("请求 JSON 必须包含 ground 字段（地面标高数组）。", status=400)
    if "width" not in payload:
        return _error("请求 JSON 必须包含 width 字段（步道净宽，单位米）。", status=400)

    result = plan_with_earthwork(payload["ground"], payload["width"])
    return JSONResponse(status_code=200, content=result)


@app.post("/api/lockable")
async def api_lockable(request: Request) -> JSONResponse:
    """测点可锁定标高分析：只接收地面标高，复用 planner 的校验。"""
    try:
        payload = await request.json()
    except Exception:
        return _error("请求体必须是合法的 JSON。", status=400)
    if not isinstance(payload, dict) or "ground" not in payload:
        return _error("请求 JSON 必须包含 ground 字段（地面标高数组）。", status=400)

    result = lockable_analysis(payload["ground"])
    return JSONResponse(status_code=200, content=result)


@app.post("/api/acceptance")
async def api_acceptance(request: Request) -> JSONResponse:
    """施工验收复核：接收原地面、冻结方案、实测标高与容差，只核对不求优。"""
    try:
        payload = await request.json()
    except Exception:
        return _error("请求体必须是合法的 JSON。", status=400)
    if not isinstance(payload, dict) or "ground" not in payload:
        return _error("请求 JSON 必须包含 ground 字段（原地面标高数组）。", status=400)
    if "plan" not in payload:
        return _error("请求 JSON 必须包含 plan 字段（已冻结的方案标高数组）。", status=400)
    if "measured" not in payload:
        return _error("请求 JSON 必须包含 measured 字段（实测标高数组）。", status=400)
    if "tolerance" not in payload:
        return _error("请求 JSON 必须包含 tolerance 字段（容差，0～20 整数厘米）。", status=400)

    result = acceptance_review(
        payload["ground"],
        payload["plan"],
        payload["measured"],
        payload["tolerance"],
    )
    return JSONResponse(status_code=200, content=result)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
