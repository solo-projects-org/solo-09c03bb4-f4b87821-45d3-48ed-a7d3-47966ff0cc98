"""FastAPI application for cross-museum specimen reconciliation."""
from __future__ import annotations

import csv
import io
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db
from .matching import STATUS_LABELS, DecisionConflict, RecomputeError

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = os.environ.get("CATALOG_DB", str(BASE_DIR.parent / "data" / "catalog.db"))
PAGE_SIZE = 50


def get_db_path() -> str:
    return os.environ.get("CATALOG_DB", DEFAULT_DB)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db(get_db_path())
    yield


app = FastAPI(title="跨馆标本对照", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.exception_handler(HTTPException)
async def _html_http_exception(request: Request, exc: HTTPException):
    accepts_html = "text/html" in request.headers.get("accept", "")
    if accepts_html or request.url.path.count("/") <= 2:
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "status_code": exc.status_code,
                "message": exc.detail,
            },
            status_code=exc.status_code,
        )
    return await http_exception_handler(request, exc)


def _safe_next(value: str) -> str:
    """Only allow same-site relative redirects."""
    if "\\" in value or "\r" in value or "\n" in value:
        return ""
    if value.startswith("/") and not value.startswith("//"):
        return value
    return ""


@app.get("/", response_class=HTMLResponse)
def index(request: Request, error: str = ""):
    comparisons = db.list_comparisons(get_db_path())
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "comparisons": comparisons,
            "error": error,
        },
    )


@app.post("/comparisons", response_class=HTMLResponse)
async def upload(
    request: Request,
    a_file: UploadFile = File(...),
    b_file: UploadFile = File(...),
):
    a_content = await a_file.read()
    b_content = await b_file.read()
    a_name = a_file.filename or "甲馆.csv"
    b_name = b_file.filename or "乙馆.csv"
    try:
        comparison_id = db.create_comparison(
            get_db_path(), a_name, a_content, b_name, b_content
        )
    except db.UploadError as exc:
        comparisons = db.list_comparisons(get_db_path())
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "comparisons": comparisons,
                "error": str(exc),
                "details": exc.details,
            },
            status_code=400,
        )
    return RedirectResponse(f"/comparisons/{comparison_id}?uploaded=1", status_code=303)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/comparisons/{comparison_id}", response_class=HTMLResponse)
def comparison_detail(
    request: Request,
    comparison_id: int,
    status: str = "all",
    page: int = 1,
    error: str = "",
    notice: str = "",
    uploaded: int = 0,
):
    comparison = db.get_comparison(get_db_path(), comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="对照任务不存在")
    if status not in ("all", "certain", "uncertain", "unmatched", "excluded"):
        status = "all"
    page = max(1, page)
    rows, total, counts = db.fetch_result_page(
        get_db_path(), comparison_id, status=status, page=page, page_size=PAGE_SIZE
    )
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    if page > pages and total > 0:
        page = pages
        rows, total, counts = db.fetch_result_page(
            get_db_path(), comparison_id, status=status, page=page, page_size=PAGE_SIZE
        )
    if uploaded and not notice:
        notice = "文件已导入并完成首次计算"
    return templates.TemplateResponse(
        request,
        "detail.html",
        {
            "comparison": comparison,
            "rows": rows,
            "total": total,
            "counts": counts,
            "status": status,
            "page": page,
            "pages": pages,
            "status_labels": STATUS_LABELS,
            "error": error,
            "notice": notice,
        },
    )


@app.post("/comparisons/{comparison_id}/decide")
async def decide(request: Request, comparison_id: int):
    form = await request.form()
    try:
        candidate_id = int(str(form.get("candidate_id", "")))
    except ValueError:
        candidate_id = -1
    action = str(form.get("action", ""))
    status = str(form.get("status", "all"))
    if status not in ("all", "certain", "uncertain", "unmatched", "excluded"):
        status = "all"
    try:
        page = max(1, int(str(form.get("page", "1"))))
    except ValueError:
        page = 1
    back = _safe_next(str(form.get("next", "")))
    if not back:
        back = f"/comparisons/{comparison_id}?status={status}&page={page}"
    try:
        db.apply_decision(get_db_path(), comparison_id, candidate_id, action)
    except (DecisionConflict, RecomputeError, sqlite3.OperationalError) as exc:
        separator = "&" if "?" in back else "?"
        return RedirectResponse(
            f"{back}{separator}error={_quote(_failure_text(exc))}", status_code=303
        )
    notice = {"confirm": "已确认该配对", "exclude": "已排除该候选", "clear": "已撤销决定"}.get(
        action, "已更新"
    )
    separator = "&" if "?" in back else "?"
    return RedirectResponse(
        f"{back}{separator}notice={_quote(notice)}", status_code=303
    )


@app.post("/comparisons/{comparison_id}/recompute")
async def recompute(request: Request, comparison_id: int):
    form = await request.form()
    back = _safe_next(str(form.get("next", ""))) or f"/comparisons/{comparison_id}"
    try:
        db.recompute(get_db_path(), comparison_id)
    except (DecisionConflict, RecomputeError, sqlite3.OperationalError) as exc:
        separator = "&" if "?" in back else "?"
        return RedirectResponse(
            f"{back}{separator}error={_quote(_failure_text(exc))}", status_code=303
        )
    separator = "&" if "?" in back else "?"
    return RedirectResponse(
        f"{back}{separator}notice={_quote('已按现有决定重新计算')}", status_code=303
    )


@app.get("/comparisons/{comparison_id}/export.csv")
def export(comparison_id: int):
    comparison = db.get_comparison(get_db_path(), comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="对照任务不存在")
    pairs = db.confirmed_pairs(get_db_path(), comparison_id)
    buffer = io.StringIO()
    buffer.write("﻿")
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "甲馆馆藏号",
            "甲馆原文件行号",
            "乙馆馆藏号",
            "乙馆原文件行号",
            "采集号",
            "县代码",
            "甲馆采集日期",
            "乙馆采集日期",
            "日期差(天)",
            "判定来源",
        ]
    )
    for pair in pairs:
        writer.writerow(
            [
                pair["a_collection"],
                pair["a_line"],
                pair["b_collection"],
                pair["b_line"],
                pair["field_number"],
                pair["county_code"],
                pair["a_date"],
                pair["b_date"],
                pair["gap_days"],
                "馆员确认" if pair["decision"] == "confirm" else "自动确定",
            ]
        )
    content = buffer.getvalue().encode("utf-8")
    filename = f"certain-pairs-{comparison_id}.csv"
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/comparisons/{comparison_id}/source/{museum}", response_class=HTMLResponse)
def source_file(request: Request, comparison_id: int, museum: str):
    if museum not in ("A", "B"):
        raise HTTPException(status_code=404, detail="馆别不存在")
    comparison = db.get_comparison(get_db_path(), comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="对照任务不存在")
    records = db.source_lines(get_db_path(), comparison_id, museum)
    return templates.TemplateResponse(
        request,
        "source.html",
        {
            "comparison": comparison,
            "museum": museum,
            "museum_label": db.MUSEUM_LABELS[museum],
            "records": records,
            "filename": comparison["a_filename"] if museum == "A" else comparison["b_filename"],
        },
    )


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _failure_text(exc: Exception) -> str:
    if isinstance(exc, (DecisionConflict, RecomputeError)):
        return str(exc)
    return "数据库暂时不可用，请稍后重试；原有决定与结果未改变"
