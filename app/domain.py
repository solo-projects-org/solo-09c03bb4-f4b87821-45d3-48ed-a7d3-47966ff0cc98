"""Dates, CSV parsing and candidate detection rules."""
from __future__ import annotations

import csv
import datetime as dt
import io
import re

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
)

# 2024-3-1, 2024/03/01, 2024年3月1日
_DATE_PATTERN = re.compile(
    r"^(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*日?"
    r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$"
)

HEADER_ALIASES = {
    "collection_number": ("馆藏号", "館藏號", "collection number", "collection_number", "catalog number", "catalog_number", "馆藏编号", "館藏編號"),
    "field_number": ("采集号", "採集號", "field number", "field_number", "采集编号", "採集編號"),
    "county_code": ("县代码", "縣代碼", "county code", "county_code", "县代码code", "countycode", "county"),
    "collected_at": ("采集日期", "採集日期", "collection date", "collection_date", "date", "日期"),
}

MAX_DATE_GAP_DAYS = 1


class RowError(ValueError):
    pass


def parse_date(raw: str) -> dt.date:
    text = (raw or "").strip()
    if not text:
        raise RowError("采集日期为空")
    for fmt in DATE_FORMATS:
        try:
            parsed = dt.datetime.strptime(text, fmt)
            return parsed.date()
        except ValueError:
            continue
    match = _DATE_PATTERN.match(text)
    if match:
        year, month, day = (int(part) for part in match.groups()[:3])
        hour = int(match.group(4) or 0)
        minute = int(match.group(5) or 0)
        second = int(match.group(6) or 0)
        try:
            return dt.datetime(year, month, day, hour, minute, second).date()
        except ValueError:
            pass
    raise RowError(f"无法识别采集日期: {text!r}")


def normalise_header(name: str) -> str:
    text = (name or "").replace("﻿", "").strip().lower()
    return re.sub(r"[\s_\-]+", "", text)


def detect_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical names to the actual CSV headers, accepting common aliases."""
    index: dict[str, str] = {}
    for header in fieldnames:
        key = normalise_header(header)
        if key:
            index.setdefault(key, header)
    mapping: dict[str, str] = {}
    for canonical, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            found = index.get(normalise_header(alias))
            if found is not None:
                mapping[canonical] = found
                break
    missing = [name for name in HEADER_ALIASES if name not in mapping]
    if missing:
        labels = {
            "collection_number": "馆藏号",
            "field_number": "采集号",
            "county_code": "县代码",
            "collected_at": "采集日期",
        }
        raise RowError("CSV 缺少列: " + "、".join(labels[m] for m in missing))
    return mapping


def parse_csv(raw: bytes) -> tuple[list[dict], list[str]]:
    """Parse an uploaded CSV into normalised rows.

    Returns (rows, errors). Fatal problems raise RowError with a user-facing
    message; per-row problems are collected and reported without importing.
    """
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gb18030")
        except UnicodeDecodeError:
            raise RowError("文件编码无法识别，请使用 UTF-8 或 GB18030")
    if not text.strip():
        raise RowError("文件为空")

    chosen = None
    last_error: RowError | None = None
    for candidate in (",", ";", "\t"):
        reader = csv.reader(io.StringIO(text), delimiter=candidate)
        try:
            header_cells = next(reader)
        except StopIteration:
            continue
        if not any(cell.strip() for cell in header_cells):
            continue
        try:
            mapping = detect_columns(header_cells)
        except RowError as exc:
            last_error = exc
            continue
        if len(header_cells) > 1:
            chosen = (candidate, header_cells, mapping)
            break
    if chosen is None:
        raise last_error or RowError("CSV 无法读取表头")
    delimiter, header_cells, mapping = chosen
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    next(reader)  # skip header

    rows: list[dict] = []
    errors: list[str] = []
    seen: dict[str, int] = {}
    previous_end = 1  # physical line of the header
    header_len = len(header_cells)
    while True:
        try:
            cells = next(reader)
        except StopIteration:
            break
        start_line = previous_end + 1
        previous_end = max(start_line, reader.line_num)
        offset = start_line
        if not any(cell.strip() for cell in cells):
            continue  # blank line
        if len(cells) > header_len:
            errors.append(f"第 {offset} 行: 列数多于表头，无法确定各列含义")
            continue
        padded = list(cells) + [""] * (header_len - len(cells))
        record = dict(zip(header_cells, padded))
        collection = (record.get(mapping["collection_number"]) or "").strip()
        field = (record.get(mapping["field_number"]) or "").strip()
        county = (record.get(mapping["county_code"]) or "").strip()
        date_raw = (record.get(mapping["collected_at"]) or "").strip()
        if not (collection or field or county or date_raw):
            continue  # blank line
        problems = []
        if not collection:
            problems.append("馆藏号为空")
        if not field:
            problems.append("采集号为空")
        if not county:
            problems.append("县代码为空")
        parsed_date = None
        try:
            parsed_date = parse_date(date_raw)
        except RowError as exc:
            problems.append(str(exc))
        if problems:
            errors.append(f"第 {offset} 行: " + "；".join(problems))
            continue
        if collection in seen:
            errors.append(
                f"第 {offset} 行: 馆藏号 {collection!r} 与本文件第 {seen[collection]} 行重复"
            )
            continue
        seen[collection] = offset
        rows.append(
            {
                "line_no": offset,
                "collection_number": collection,
                "field_number": field,
                "county_code": county,
                "collected_on": parsed_date.isoformat(),
            }
        )
    if errors:
        return rows, errors
    if not rows:
        raise RowError("CSV 没有有效数据行")
    return rows, errors


def date_gap_days(a: str, b: str) -> int:
    return abs((dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days)


def is_candidate(a: dict, b: dict) -> bool:
    return (
        a["field_number"] == b["field_number"]
        and a["county_code"] == b["county_code"]
        and date_gap_days(a["collected_on"], b["collected_on"]) <= MAX_DATE_GAP_DAYS
    )
