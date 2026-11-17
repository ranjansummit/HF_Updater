"""Fetch a complete NEPSE daily floorsheet, then publish its non-broker aggregate.

Raw: ranjansummit/nepse_floorsheet/YYYY-MM-DD.parquet
Aggregated: ranjansummit/testing/combined_dataset.parquet (upsert one date)
Index: official NEPSE API -> ranjansummit/dailyindices/NEPSE_INDEX.csv

Usage: python nepse_daily_v2.py --yes
Repair an aggregate: python nepse_daily_v2.py --date YYYY-MM-DD --aggregate-only --yes
Environment: HF_TOKEN; optional HF_REPO_ID, HF_AGG_REPO_ID, HF_INDEX_REPO_ID.
Requires: pandas>=2.2,<3 numpy<3 pyarrow>=15 requests huggingface_hub>=0.27,<2
Uses the original non-broker feature formulas; S below must match seed training settings.
Model: ranjansummit/nepse-p33-models/combined_production_YYYY-MM-DD.joblib
Filename date is the last usable training date, not the execution date.
Train only: python nepse_daily_v2.py --train-only --yes
Also requires scikit-learn==1.6.1 joblib wasmtime; optional HF_MODEL_REPO_ID.
Official index authentication downloads css.wasm from the official NEPSE site.
No broker features, historical master download, or historical backfill.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi

BASE_URL = "https://nepsealerts.com/api/floorsheet"
PAGE_SIZE = 200
NEPAL = timezone(timedelta(hours=5, minutes=45))
UTC = timezone.utc
REPO_DEFAULT = "ranjansummit/nepse_floorsheet"
# The V2 master assembled through this date. Keep later daily updates separate.
MASTER_LAST_DATE = date(2026, 10, 30)
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "parquet_out"
REPORT_DIR = BASE_DIR / "reports"
SCHEMA = pa.schema([
    pa.field("contract_id", pa.string()),
    pa.field("business_date", pa.string()),
    pa.field("symbol", pa.string()),
    pa.field("buyer", pa.string()),
    pa.field("seller", pa.string()),
    pa.field("qty", pa.int64()),
    pa.field("rate", pa.float64()),
    pa.field("amount", pa.float64()),
    pa.field("traded_at", pa.timestamp("us", tz="UTC")),
])
COLUMNS = SCHEMA.names


def fail(message: str) -> None:
    raise ValueError(message)


def pick(row: dict, *names: str, required=False):
    for key in names:
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return row[key]
    if required:
        fail(f"missing required field (tried {names}); API keys={list(row)}")
    return None


def numeric(value, name: str) -> float:
    if isinstance(value, bool):
        fail(f"invalid {name}: {value!r}")
    try:
        result = float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        fail(f"invalid {name}: {value!r}")
    if not math.isfinite(result) or result < 0:
        fail(f"invalid {name}: {value!r}")
    return result


def parse_timestamp(row: dict, day: str):
    """UTC-aware datetime or None. API `traded_at` is interpreted as UTC.

    Explicit `traded_at_nepal`, `trade_time`, and `time` are interpreted as
    Nepal local when timezone is absent. All timezone-aware values are honored.
    """
    candidates = (
        ("traded_at_utc", UTC), ("traded_at", UTC),
        ("traded_at_nepal", NEPAL), ("trade_time", NEPAL), ("time", NEPAL),
    )
    for field, assumed_tz in candidates:
        value = row.get(field)
        if value is None or str(value).strip().lower() in ("", "none", "nat", "nan", "null"):
            continue
        raw = str(value).strip()
        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?", raw):
            raw = f"{day}T{raw}"
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            fail(f"invalid timestamp {field}={value!r}")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=assumed_tz)
        return parsed.astimezone(UTC)
    return None


def normalize(raw: dict, expected_day: str) -> tuple[dict, dict]:
    """Never infer business_date from contract ID; never manufacture an ID."""
    day_value = pick(raw, "business_date", "businessDate", "calculation_date", required=True)
    business_date = str(day_value)[:10]
    try:
        if date.fromisoformat(business_date).isoformat() != expected_day:
            fail(f"wrong business_date {business_date!r} for {expected_day}")
    except ValueError:
        fail(f"invalid business_date {day_value!r}")
    contract_id = str(pick(raw, "contract_id", "contract_no", "contract_number", required=True)).strip()
    if contract_id.endswith(".0") and contract_id[:-2].isdigit():
        contract_id = contract_id[:-2]
    if contract_id.lower() in {"transactionnumber", "contract_id", "none", "nan", "null"} or not contract_id:
        fail(f"invalid contract_id {contract_id!r}")
    symbol = str(pick(raw, "symbol", "stock_symbol", required=True)).strip()
    buyer = str(pick(raw, "buyer_broker", "buyer", required=True)).strip()
    seller = str(pick(raw, "seller_broker", "seller", required=True)).strip()
    quantity = numeric(pick(raw, "quantity", "qty", required=True), "quantity")
    if quantity <= 0 or not quantity.is_integer():
        fail(f"quantity must be a positive whole number: {quantity!r}")
    rate = numeric(pick(raw, "rate", required=True), "rate")
    amount = numeric(pick(raw, "amount", required=True), "amount")
    repair = None
    if rate == 0 and amount > 0:
        rate = amount / quantity
        repair = "rate=amount/qty"
    if amount == 0 and rate > 0:
        amount = rate * quantity
        repair = "amount=rate*qty"
    if not rate > 0 or not amount > 0:
        fail(f"unrecoverable zero: contract_id={contract_id}, qty={quantity}, rate={rate}, amount={amount}")
    traded_at = parse_timestamp(raw, expected_day)
    return {
        "contract_id": contract_id,
        "business_date": business_date,
        "symbol": symbol,
        "buyer": buyer,
        "seller": seller,
        "qty": int(quantity),
        "rate": rate,
        "amount": amount,
        "traded_at": traded_at,
    }, {"repair": repair}


def matching(existing: dict, fresh: dict) -> tuple[bool, bool]:
    """(core fields match, timestamps compatible within one hour)."""
    core = ("business_date", "symbol", "buyer", "seller", "qty", "rate", "amount")
    for key in core:
        a, b = existing[key], fresh[key]
        if key in ("rate", "amount"):
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-5):
                return False, False
        elif a != b:
            return False, False
    t1, t2 = existing["traded_at"], fresh["traded_at"]
    if t1 is not None and t2 is not None and abs((t1 - t2).total_seconds()) >= 3600:
        return True, False
    return True, True


def page_results(payload: object, page: int) -> tuple[list, int | None]:
    """Only accept a recognizable API payload. Never interpret an error as 0 rows."""
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        fail(f"page {page}: expected JSON object or list, got {type(payload).__name__}")
    if payload.get("error") and payload.get("results") is None:
        fail(f"page {page}: API error: {payload['error']}")
    if "results" not in payload:
        fail(f"page {page}: API response has no 'results'; keys={list(payload)}")
    rows = payload["results"]
    if not isinstance(rows, list):
        fail(f"page {page}: results is not a list")
    total = None
    for node in (payload.get("pagination"), payload.get("meta"), payload):
        if not isinstance(node, dict):
            continue
        for field in ("total_count", "total_records", "total_results", "totalItems", "total"):
            if field in node and node[field] is not None:
                try:
                    total = int(node[field])
                except (TypeError, ValueError):
                    fail(f"page {page}: invalid pagination total {node[field]!r}")
                break
        if total is not None:
            break
    return rows, total


def fetch_page(session, day, page, retries=5):
    """Return (rows, reported_total), or None when the page responds HTTP 404.

    A 404 is NOT a success by itself. collect_day only accepts it as the end
    after earlier valid pages, repeated confirmation and completeness checks.
    Other HTTP errors (including 429/5xx) remain fatal after retrying.
    """
    params = {"date": day, "page": page, "page_size": PAGE_SIZE}
    for attempt in range(1, retries + 1):
        try:
            response = session.get(BASE_URL, params=params, timeout=(10, 45))
            if response.status_code == 404:
                return None
            response.raise_for_status()
            try:
                return page_results(response.json(), page)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f"Bad API response on page {page}: {exc}") from exc
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Page {page} failed after {retries} attempts; refusing upload: {exc}"
                ) from exc
            wait = min(2 ** (attempt - 1), 16)
            print(f"  [retry] page={page} attempt={attempt}/{retries}: {exc}; backoff={wait}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def collect_day(day: str, expected_count: int | None) -> tuple[Path | None, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / f"{day}.parquet"
    partial = OUT_DIR / f"{day}.partial.parquet"
    if partial.exists():
        partial.unlink()  # our own incomplete local output from a prior attempt
    stats = {
        "date": day, "page_size": PAGE_SIZE, "pages_nonempty": 0,
        "raw_api_rows": 0, "unique_rows": 0, "duplicate_rows": 0,
        "repaired_values": 0, "api_reported_total": None,
        "expected_count": expected_count, "timestamp_rows": 0,
        "status": "IN_PROGRESS", "end_reason": None,
    }
    session = requests.Session()
    session.headers.update({"User-Agent": "NEPSE-daily-parquet-v2/1.0", "Accept": "application/json"})
    writer = None
    seen: dict[str, dict] = {}
    staged: list[dict] = []
    page = 1
    last_fingerprint = None
    try:
        while page <= 5000:
            result = fetch_page(session, day, page)
            if result is None:
                # Some APIs return HTTP 404 instead of an empty list when the
                # requested page is beyond the last available page.
                if stats["pages_nonempty"] == 0:
                    fail(f"page {page}: HTTP 404 before any data; NOT treating as completed day")
                time.sleep(1.0)
                if fetch_page(session, day, page) is not None:
                    fail(f"page {page}: transient HTTP 404; pagination is unstable")

                declared_total = stats["api_reported_total"]
                if declared_total is not None and stats["raw_api_rows"] != declared_total:
                    fail(f"page {page}: HTTP 404 with only {stats['raw_api_rows']} of "
                         f"{declared_total} API-reported rows; refusing incomplete upload")

                if declared_total is None:
                    # With no total in the API response, check there is no
                    # reachable next page (a hole in pagination).
                    next_result = fetch_page(session, day, page + 1)
                    if next_result is not None:
                        next_rows, next_total = next_result
                        if next_rows:
                            fail(f"page {page}: HTTP 404 but page {page+1} has rows; refusing incomplete upload")
                        if next_total is not None and next_total != stats["raw_api_rows"]:
                            fail(f"page {page+1}: API total {next_total} disagrees with "
                                 f"{stats['raw_api_rows']} collected rows")
                    print("[note] API did not provide a total: end-of-pages confirmed, "
                          "but an independent count is still recommended")

                stats["end_reason"] = "HTTP_404_CONFIRMED"
                print(f"[done] page={page} returned HTTP 404 twice after "
                      f"{stats['pages_nonempty']} valid pages; pagination ended", flush=True)
                break

            rows, total = result
            if total is not None:
                if total < 0:
                    fail(f"page {page}: invalid negative total {total}")
                if stats["api_reported_total"] is not None and stats["api_reported_total"] != total:
                    fail(f"API total changed between pages: {stats['api_reported_total']} -> {total}")
                stats["api_reported_total"] = total
            if len(rows) > PAGE_SIZE:
                fail(f"page {page}: returned {len(rows)} rows, more than page_size={PAGE_SIZE}")
            if (stats["api_reported_total"] is not None
                    and stats["raw_api_rows"] + len(rows) > stats["api_reported_total"]):
                fail(f"page {page}: API returned more rows than its stated total")
            if not rows:
                # A single transient empty response is not always the true end.
                time.sleep(1.0)
                confirmation = fetch_page(session, day, page)
                if confirmation is None:
                    fail(f"page {page}: empty response became HTTP 404 on retry; pagination is unstable")
                confirm_rows, confirm_total = confirmation
                if confirm_rows:
                    fail(f"page {page}: an empty page became non-empty on retry; pagination is unstable")
                if confirm_total is not None and stats["api_reported_total"] not in (None, confirm_total):
                    fail(f"page {page}: inconsistent API total on empty-page confirmation")
                if confirm_total is not None and stats["api_reported_total"] is None:
                    stats["api_reported_total"] = confirm_total
                stats["end_reason"] = "EMPTY_PAGE_CONFIRMED"
                print(f"[done] page={page} empty twice; pagination ended normally")
                break
            fingerprint = (len(rows), json.dumps(rows[0], sort_keys=True, default=str),
                           json.dumps(rows[-1], sort_keys=True, default=str))
            if fingerprint == last_fingerprint:
                fail(f"page {page}: response repeated previous page; pagination may be broken")
            last_fingerprint = fingerprint
            stats["pages_nonempty"] += 1
            stats["raw_api_rows"] += len(rows)
            for item in rows:
                if not isinstance(item, dict):
                    fail(f"page {page}: non-object transaction row")
                record, note = normalize(item, day)
                cid = record["contract_id"]
                if cid in seen:
                    core_ok, time_ok = matching(seen[cid], record)
                    if not core_ok or not time_ok:
                        fail(f"page {page}: same contract_id {cid!r} with conflicting transaction details/timestamps")
                    stats["duplicate_rows"] += 1
                    # Prefer a provided timestamp if the first copy lacked one.
                    if seen[cid]["traded_at"] is None and record["traded_at"] is not None:
                        seen[cid]["traded_at"] = record["traded_at"]
                        # For correct output, defer writing until all pages complete.
                    continue
                seen[cid] = record
                if note["repair"]:
                    stats["repaired_values"] += 1
            print(f"[page {page}] {len(rows)} rows; cumulative {stats['raw_api_rows']}; unique {len(seen)}", flush=True)
            if (stats["api_reported_total"] is not None
                    and stats["raw_api_rows"] == stats["api_reported_total"]):
                stats["end_reason"] = "API_TOTAL_REACHED"
                print(f"[done] Collected all {stats['raw_api_rows']} API-reported rows "
                      "without requesting an out-of-range page", flush=True)
                break
            page += 1
            time.sleep(0.25)
        else:
            fail("More than 5000 pages; refusing to assume download is complete")

        if stats["api_reported_total"] is not None and stats["raw_api_rows"] != stats["api_reported_total"]:
            fail(f"Fetched {stats['raw_api_rows']} raw rows but API reports {stats['api_reported_total']}")
        if expected_count is not None and len(seen) != expected_count:
            fail(f"Found {len(seen)} unique IDs, but --expected-count says {expected_count}")
        if not seen:
            stats["status"] = "NO_TRADES_NO_UPLOAD"
            return None, stats

        # Keep a one-day dictionary, not 70 million rows in Python RAM.
        stats["unique_rows"] = len(seen)
        stats["timestamp_rows"] = sum(1 for x in seen.values() if x["traded_at"] is not None)
        writer = pq.ParquetWriter(str(partial), SCHEMA, compression="snappy")
        for record in seen.values():
            staged.append(record)
            if len(staged) >= 10000:
                writer.write_table(pa.Table.from_pylist(staged, schema=SCHEMA))
                staged.clear()
        if staged:
            writer.write_table(pa.Table.from_pylist(staged, schema=SCHEMA))
        writer.close()
        writer = None
        pf = pq.ParquetFile(str(partial))
        if pf.metadata.num_rows != stats["unique_rows"] or not pf.schema_arrow.equals(SCHEMA):
            fail("Output Parquet schema/count verification failed")
        # Close underlying file handles before rename (Windows WinError 32 prevention).
        del pf
        os.replace(partial, final)
        stats["status"] = "LOCAL_VALIDATED_NOT_UPLOADED"
        stats["local_file"] = str(final)
        return final, stats
    finally:
        if writer is not None:
            writer.close()
        session.close()
        if partial.exists():
            partial.unlink()


def save_report(day: str, stats: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / f"{day}_upload_audit.json"
    output.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[report] {output}")



# Daily aggregation: original non-broker training features, no model training.
import io
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError

AGG_REPO = os.getenv("HF_AGG_REPO_ID", "ranjansummit/testing")
AGG_FILE = "combined_dataset.parquet"
AGG_META = "combined_dataset.metadata.json"
INDEX_REPO = os.getenv("HF_INDEX_REPO_ID", "ranjansummit/dailyindices")
INDEX_FILE = "NEPSE_INDEX.csv"
AGG_VERSION = "same_day_original_10_guarded_v1"
# Must match the notebook that created the initial combined_dataset.parquet.
S = {
    "timezone": "Asia/Kathmandu",
    "training_window_start": "10:30",
    "training_window_end": "13:00",
    "filter_enabled": True,
    "minimum_transaction_amount": 500000,
    "sequence_fraction": 0.5,
    "neutral_percentage": 0.005,
    "index_daily_return_lookback_days": 1,
    "prediction_min_transactions": 1,
    "spark_max_rows_per_day": 400000,
}
TZ = S["timezone"]
INDEX_DAILY_DAYS = S["index_daily_return_lookback_days"]
FEATS_TX = ['n_elig', 'log_amt', 'w_ret', 'm_ret', 'med_ret', 'up_frac', 'late_w_ret', 'idx_proxy', 'n_sym', 'buy_press']
FEATS_PX = [f"idx_daily_ret_{lag}" for lag in range(1, INDEX_DAILY_DAYS + 1)]
AGG_COLUMNS = ['date','mode','input_method','prev_close','close','chg','chg_pct','cls'] + FEATS_PX + FEATS_TX

def window_by_timestamp(df, start=S['training_window_start'], end=S['training_window_end']):
    """Training-window trades: rows whose Nepal-time HH:MM is within [start, end] (inclusive). Needs traded_at."""
    hm = df.traded_at.dt.tz_convert(TZ).dt.strftime('%H:%M')
    return df[(hm >= start) & (hm <= end)]

def counter_parts(df):
    """Splits 16-digit contract ids (YYYYMMDD + 2-digit stream + 6-digit counter).
    Returns (valid_mask, stream, counter). Inside one stream the counter follows time order."""
    cid = df.contract_id.astype(str)
    ok = cid.str.len().eq(16) & cid.str.isdigit()
    return (ok, cid.str[8:10], pd.to_numeric(cid.str[10:16].where(ok), errors='coerce'))

def window_by_counter(df, frac=S['sequence_fraction']):
    """Counter rule for days WITHOUT trade times: per date AND per stream, find the max counter (before the
    amount filter), cutoff = floor(max * frac), keep trades with counter <= cutoff."""
    ok, stream, ctr = counter_parts(df)
    if not ok.all():
        raise ValueError(f'{(~ok).sum()} ids not 16-digit; other formats are not guessed')
    date = df['business_date'] if 'business_date' in df else pd.Series('one_day', index=df.index)
    mx = ctr.groupby([date, stream]).transform('max')
    return df[ctr <= np.floor(mx * frac)]

def amount_slab(df):
    """minimum_transaction_amount filter (only when filter_enabled). Raw history is never filtered."""
    return df[df.amount >= S['minimum_transaction_amount']] if S['filter_enabled'] else df

def order_key(df):
    """Time order: by traded_at when every row has it, otherwise by contract id (same order inside a stream)."""
    return df.sort_values(['traded_at', 'contract_id'] if df.traded_at.notna().all() else ['contract_id'])

def prev_reference(prev_full):
    """Per-symbol previous close proxy (last trade) and turnover from the previous completed day (all trades)."""
    p = order_key(prev_full)
    g = p.groupby('symbol')
    return pd.DataFrame({'pclose': g.rate.last(), 'pturn': g.amount.sum()})

def daily_index_return_features(closes, date):
    """Return N individual prior daily returns, or None if history is insufficient.

    `closes` is a Series indexed by ISO trading date, with one official close per date.
    Strictly excludes `date` itself: historical training and today's prediction both
    see exactly the same kind of lagged information.
    N daily movements require N+1 *previous* official closes.
    """
    past = closes.loc[closes.index < date].sort_index()
    if len(past) < INDEX_DAILY_DAYS + 1:
        return None
    vals = past.iloc[-(INDEX_DAILY_DAYS + 1):].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals[:-1] <= 0).any():
        return None
    returns = vals[1:] / vals[:-1] - 1.0
    return {f'idx_daily_ret_{lag}': float(returns[-lag]) for lag in range(1, INDEX_DAILY_DAYS + 1)}

def label(chg_pct):
    """Class of a fractional change: bull > +neutral_percentage, bear < -neutral_percentage, deer otherwise."""
    n = S['neutral_percentage']
    return 'bull' if chg_pct > n else 'bear' if chg_pct < -n else 'deer'

def parse_times(df):
    nat = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns, UTC]",
    )

    raw = df["traded_at"].astype("string").str.strip()

    full = raw.where(
        raw.str.contains(
            r"\d{4}-\d{2}-\d{2}", na=False
        ),
        df["business_date"].astype(str) + " " + raw,
    )

    aware = full.str.contains(
        r"(?:Z|[+-]\d{2}:?\d{2})$",
        na=False,
    )

    explicit = pd.to_datetime(
        full.where(aware),
        errors="coerce",
        utc=True,
        format="mixed",
    )

    naive = pd.to_datetime(
        full.where(~aware),
        errors="coerce",
        format="mixed",
    )

    for zone in (S["timezone"], "UTC"):
        candidate = (
            naive.dt.tz_localize(
                zone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
            .dt.tz_convert("UTC")
        )

        candidate = explicit.where(aware, candidate)
        local = candidate.dt.tz_convert(S["timezone"])

        valid = (
            candidate.notna()
            & local.dt.hour.between(10, 15)
            & local.dt.strftime("%Y-%m-%d").eq(
                df["business_date"]
            )
        )

        parsed_count = int(candidate.notna().sum())

        if (
            parsed_count > 0
            and valid.sum() / parsed_count >= 0.95
        ):
            return candidate.where(valid)

    return nat

def tx_features(elig, ref):
    """Eligible trades -> fixed day-level features (FEATS_TX), using the previous day's per-symbol reference.
    Returns a dict; NaN features when no symbol matches."""
    e = order_key(elig).merge(ref, left_on='symbol', right_index=True, how='inner')
    if e.empty:
        return {k: np.nan for k in FEATS_TX} | {'n_elig': len(elig)}
    r = e.rate / e.pclose - 1
    w = e.amount / e.amount.sum()
    late = e.iloc[int(len(e) * 0.75):]
    rl = late.rate / late.pclose - 1
    last = e.groupby('symbol').agg(rate=('rate', 'last'), pclose=('pclose', 'first'), pturn=('pturn', 'first'))
    lr = last.rate / last.pclose - 1
    return {'n_elig': len(elig), 'log_amt': np.log1p(e.amount.sum()), 'w_ret': float((w * r).sum()), 'm_ret': float(r.mean()), 'med_ret': float(r.median()), 'up_frac': float((r > 0).mean()), 'late_w_ret': float((late.amount * rl).sum() / late.amount.sum()), 'idx_proxy': float((last.pturn * lr).sum() / last.pturn.sum()), 'n_sym': len(last), 'buy_press': float(np.sign(r).mul(e.amount).sum() / e.amount.sum())}

_original_tx_features_before_hf_guard = tx_features

def tx_features(elig, ref):
    numeric_ref = ref[["pclose", "pturn"]].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_ref = (
        np.isfinite(numeric_ref).all(axis=1)
        & (numeric_ref["pclose"] > 0)
        & (numeric_ref["pturn"] > 0)
    )

    clean_ref = ref.loc[valid_ref].copy()
    clean_ref[["pclose", "pturn"]] = numeric_ref.loc[valid_ref]

    # Zero-amount trades cannot contribute to weighted features.
    usable = elig.loc[elig["amount"] > 0].copy()

    result = _original_tx_features_before_hf_guard(
        usable, clean_ref
    )

    # Preserve the original eligible-trade count.
    result["n_elig"] = len(elig)

    # An all-zero late slice, for example, becomes a missing feature
    # handled by the existing training imputer.
    return {
        key: (
            float(value)
            if pd.notna(value) and np.isfinite(value)
            else np.nan
        )
        for key, value in result.items()
    }

def _hf_download(api, repo, filename, revision):
    return Path(hf_hub_download(
        repo_id=repo, filename=filename, repo_type="dataset",
        revision=revision, token=api.token,
    ))


def _read_complete_daily(path, day):
    """Daily files from this publisher already have unique contract IDs."""
    frame = pd.read_parquet(path)
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        fail(f"{day}: daily floorsheet lacks {sorted(missing)}")
    if frame.empty or not frame.business_date.astype(str).eq(day).all():
        fail(f"{day}: expected a nonempty single-day floorsheet")
    if frame.contract_id.isna().any() or frame.contract_id.astype(str).duplicated().any():
        fail(f"{day}: missing/duplicate contract IDs; repair the source daily file")
    if len(frame) > S["spark_max_rows_per_day"]:
        fail(f"{day}: exceeds configured daily row limit")
    frame = frame.copy()
    frame['contract_id'] = frame.contract_id.astype(str)
    numeric = frame[['qty','rate','amount']].apply(pd.to_numeric, errors='coerce')
    frame[['qty','rate','amount']] = numeric.replace([np.inf,-np.inf], np.nan).fillna(0).clip(lower=0)
    for col in ('symbol','buyer','seller'):
        frame[col] = frame[col].fillna('<missing>').astype(str)
    frame['traded_at'] = parse_times(frame)
    return frame.sort_values('contract_id').reset_index(drop=True)


# Official website client: same WASM-token approach as the earlier client.py.
# No third-party price/index fallback. The public website session needs no account key.
class OfficialNepseIndex:
    def __init__(self, origin):
        import wasmtime
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.origin = origin.rstrip('/')
        self.session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5,
                      status_forcelist=[429,500,502,503,504], allowed_methods=['GET'])
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        self.session.headers.update({'User-Agent':'Mozilla/5.0',
                                     'Referer':self.origin+'/', 'Accept':'application/json'})
        try:
            response = self.session.get(self.origin+'/assets/prod/css.wasm', timeout=(10,20))
            response.raise_for_status()
            if not response.content.startswith(b'\x00asm'):
                fail('Official css.wasm URL did not return a WebAssembly module')
            engine = wasmtime.Engine()
            self.store = wasmtime.Store(engine)
            module = wasmtime.Module(engine, response.content)
            self.exports = wasmtime.Instance(self.store, module, []).exports(self.store)
            self.authenticate()
        except Exception:
            self.session.close()
            raise

    def authenticate(self):
        self.session.headers.pop('Authorization', None)
        response = self.session.get(self.origin+'/api/authenticate/prove', timeout=(10,20))
        response.raise_for_status()
        proof = response.json()
        s1,s2,s3,s4,s5 = (int(proof[f'salt{i}']) for i in range(1,6))
        token = proof['accessToken']
        salt_orders = [('cdx',(s1,s2,s3,s4,s5)), ('rdx',(s1,s2,s4,s3,s5)),
                       ('bdx',(s1,s2,s4,s3,s5)), ('ndx',(s1,s2,s4,s3,s5)),
                       ('mdx',(s1,s2,s4,s3,s5))]
        cuts = [int(self.exports[name](self.store,*salts)) for name,salts in salt_orders]
        if len(set(cuts)) != 5 or any(i < 0 or i >= len(token) for i in cuts):
            fail('Official token format changed: invalid WASM character positions')
        cutset = set(cuts)
        clean = ''.join(c for i,c in enumerate(token) if i not in cutset)
        self.session.headers['Authorization'] = 'Salter ' + clean

    def history_page(self, page):
        url = self.origin+'/api/nots/index/history/58'
        params = {'size':500, 'page':page, 'sort':'businessDate,desc'}
        response = self.session.get(url, params=params, timeout=(10,30))
        if response.status_code == 401:
            self.authenticate()
            response = self.session.get(url, params=params, timeout=(10,30))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload,dict) or not isinstance(payload.get('content'),list):
            fail('Unexpected official index-history response; expected content list')
        return payload

    def close(self):
        self.session.close()


def _official_day(value):
    # Business dates are explicit dates in the official API, never inferred from IDs.
    result = str(value or '')[:10]
    try:
        return date.fromisoformat(result).isoformat()
    except ValueError:
        fail(f'Invalid official index businessDate: {result!r}')


def fetch_official_index_rows(target_day, since):
    errors = []
    for origin in ('https://www.nepalstock.com','https://nepalstock.com.np'):
        client = None
        try:
            print(f'[INDEX FETCH] Official NEPSE index 58: {origin}; through {target_day}', flush=True)
            client = OfficialNepseIndex(origin)
            records, seen_pages = [], set()
            found_target = False
            for page in range(100):
                payload = client.history_page(page)
                content = payload['content']
                if not content:
                    break
                dates = [_official_day(item.get('businessDate')) for item in content]
                signature = tuple(dates)
                if signature in seen_pages:
                    fail('Official index API repeated a page; refusing incomplete history')
                seen_pages.add(signature)
                records.extend(content)
                found_target = found_target or target_day in dates
                print(f'[INDEX PAGE] {page}: {len(content)} rows; {min(dates)} .. {max(dates)}', flush=True)
                if payload.get('last') or (found_target and min(dates) <= since):
                    break
            else:
                fail('Official index history exceeded pagination limit')
            rows = [row for row in records if since <= _official_day(row.get('businessDate')) <= target_day]
            if not any(_official_day(row.get('businessDate')) == target_day for row in rows):
                fail(f'Official NEPSE history has no completed index row for {target_day} yet')
            return rows, origin+'/api/nots/index/history/58'
        except Exception as exc:
            # Never print response bodies, request headers or API tokens.
            errors.append(f'{origin}: {type(exc).__name__}: {str(exc)[:200]}')
            print(f'[INDEX SOURCE FAILED] {errors[-1]}', flush=True)
        finally:
            if client is not None:
                client.close()
    fail('Official NEPSE index update failed. No third-party index substituted. '+ ' | '.join(errors))


def _official_number(record, aliases):
    for name in aliases:
        if record.get(name) is not None and str(record[name]).strip() != '':
            value = float(str(record[name]).replace(',',''))
            if not math.isfinite(value):
                fail(f'Nonfinite official index field: {name}')
            return value
    return None


def merge_official_index(existing, records, target_day):
    """Preserve history/schema; update supplied official values; leave absent fields blank."""
    result = existing.copy()
    result.columns = result.columns.str.strip().str.lower()
    date_col = 'published_date' if 'published_date' in result.columns else 'date'
    if date_col not in result.columns or 'close' not in result.columns:
        fail('NEPSE_INDEX.csv requires published_date/date and close')
    result[date_col] = pd.to_datetime(result[date_col], errors='raise', format='mixed').dt.strftime('%Y-%m-%d')
    if result[date_col].isna().any() or result[date_col].duplicated().any():
        fail('Existing index CSV has missing/duplicate dates')
    prior_closes = pd.to_numeric(result['close'], errors='raise')
    if not np.isfinite(prior_closes).all() or not (prior_closes > 0).all():
        fail('Existing index CSV contains invalid closes')
    frame = result.set_index(date_col)
    fields = {
        'open':('openingIndex','openIndex','open'),
        'high':('highIndex','high'), 'low':('lowIndex','low'),
        'close':('closingIndex',),
        'per_change':('percentageChange','percentChange','perChange'),
        'traded_quantity':('totalTradedQuantity','totalQuantity','tradedQuantity'),
        'traded_amount':('totalTurnover','totalTradedValue','tradedAmount','turnover'),
    }
    incoming = {}
    for record in records:
        day = _official_day(record.get('businessDate'))
        if day > target_day:
            continue
        close = _official_number(record, ('closingIndex',))
        if close is None or close <= 0:
            fail(f'{day}: missing/invalid closingIndex from official NEPSE history')
        if day in incoming and incoming[day]['closingIndex'] != record['closingIndex']:
            fail(f'{day}: conflicting official index closes')
        incoming[day] = record
    if target_day not in incoming:
        fail(f'Official response lacks requested index date {target_day}')
    for day,record in sorted(incoming.items()):
        if day not in frame.index:
            frame.loc[day] = {col:np.nan for col in frame.columns}
        for col,aliases in fields.items():
            if col not in frame.columns:
                continue
            value = _official_number(record,aliases)
            if value is not None:
                frame.loc[day,col] = value
        # totalTransaction is a trade COUNT, never a share quantity or turnover.
        # Keep source status only if present; do not invent an 'A' status.
        if 'status' in frame.columns and record.get('status') is not None:
            frame.loc[day,'status'] = str(record['status'])
    frame = frame.sort_index()
    if 'per_change' in frame.columns:
        returns = pd.to_numeric(frame['close']).pct_change(fill_method=None)*100
        for day in incoming:
            # Missing percent change can be derived from adjacent official closes.
            if pd.isna(frame.loc[day,'per_change']) and pd.notna(returns.loc[day]):
                frame.loc[day,'per_change'] = float(returns.loc[day])
    return frame.reset_index()[result.columns]


def refresh_official_index(api, target_day):
    revision = api.repo_info(INDEX_REPO, repo_type='dataset').sha
    seed = pd.read_csv(_hf_download(api, INDEX_REPO, INDEX_FILE, revision))
    date_col = 'published_date' if 'published_date' in seed else 'date'
    if date_col not in seed:
        fail('Index CSV has no date column')
    known_dates = pd.to_datetime(seed[date_col],errors='raise',format='mixed').dt.strftime('%Y-%m-%d')
    relevant = known_dates[known_dates <= target_day]
    since = min(target_day, str(relevant.max())) if not relevant.empty else target_day
    # Includes last saved session plus all subsequent official rows through target.
    records,source_url = fetch_official_index_rows(target_day,since)
    for attempt in range(3):
        if attempt:
            revision = api.repo_info(INDEX_REPO, repo_type='dataset').sha
            seed = pd.read_csv(_hf_download(api, INDEX_REPO, INDEX_FILE, revision))
        merged = merge_official_index(seed,records,target_day)
        # Identical values need no redundant commit.
        try:
            pd.testing.assert_frame_equal(seed.reset_index(drop=True),merged.reset_index(drop=True),
                                          check_dtype=False,check_exact=True)
            print(f'[INDEX SUCCESS] {target_day}: official values already match HF; no commit', flush=True)
            return {'status':'UNCHANGED','date':target_day,'source':source_url,'revision':revision}
        except AssertionError:
            pass
        OUT_DIR.mkdir(parents=True,exist_ok=True)
        output = OUT_DIR/INDEX_FILE
        merged.to_csv(output,index=False)
        audit = {'source':source_url,'index_id':58,'requested_date':target_day,
                 'fetched_at':datetime.now(UTC).isoformat(), 'official_records':records}
        try:
            commit = api.create_commit(repo_id=INDEX_REPO,repo_type='dataset',parent_commit=revision,
                commit_message=f'Update official NEPSE index through {target_day}',operations=[
                    CommitOperationAdd(path_in_repo=INDEX_FILE,path_or_fileobj=str(output)),
                    CommitOperationAdd(path_in_repo=f'index_audit/{target_day}.json',
                                       path_or_fileobj=json.dumps(audit,indent=2).encode()),
                ])
        except HfHubHTTPError as exc:
            if getattr(exc.response,'status_code',None) in (409,412) and attempt < 2:
                continue
            raise
        url = str(getattr(commit,'commit_url',''))
        print(f'[INDEX UPLOAD SUCCESS] {INDEX_REPO}/{INDEX_FILE}: {target_day}; {url}',flush=True)
        return {'status':'UPLOADED','date':target_day,'source':source_url,'commit_url':url,
                'revision':getattr(commit,'oid',None)}
    fail('Could not commit the official index update')


def _read_index_file(path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    frame = frame.rename(columns={'published_date':'date'})
    if not {'date','close'} <= set(frame.columns):
        fail('Index CSV requires published_date/date and close')
    frame['date'] = pd.to_datetime(frame.date, errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
    frame['close'] = pd.to_numeric(frame.close, errors='coerce')
    frame = frame.dropna(subset=['date','close'])
    frame = frame[np.isfinite(frame.close) & (frame.close > 0)]
    if frame.groupby('date')['close'].nunique().gt(1).any():
        fail('Conflicting official index closes for the same date')
    return frame.drop_duplicates('date').sort_values('date').set_index('date')['close']


def make_aggregate_row(day, previous, closes, current, prior):
    px = daily_index_return_features(closes, day)
    if px is None:
        fail(f'{day}: not enough previous index closes for return features')
    ref = prev_reference(prior)
    features = None
    method = 'counter'
    if current.traded_at.notna().all():
        features = tx_features(amount_slab(window_by_timestamp(current)), ref)
        if features['n_elig'] >= S['prediction_min_transactions'] and pd.notna(features['w_ret']):
            method = 'ts'
        else:
            features = None
    if features is None:
        features = tx_features(amount_slab(window_by_counter(current)), ref)
    change = float(closes[day] - closes[previous])
    row = {'date':day, 'mode':'combined', 'input_method':method,
           'prev_close':float(closes[previous]), 'close':float(closes[day]),
           'chg':change, 'chg_pct':change / float(closes[previous]),
           'cls':label(change / float(closes[previous])), **px, **features}
    # Persist the same rows as training preparation, including unusable windows.
    # Training's usable_rows filter will exclude n_elig=0 / missing w_ret.
    return row


def merge_aggregate(existing, row):
    missing = set(AGG_COLUMNS) - set(existing.columns)
    extra = set(existing.columns) - set(AGG_COLUMNS)
    if missing or extra:
        fail(f'Aggregate schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}. '
             'This publisher uses the original NON-BROKER training features.')
    if existing.date.isna().any() or existing.date.astype(str).duplicated().any():
        fail('Existing aggregate has missing/duplicate dates; not overwriting it')
    if not existing.empty:
        if not existing['mode'].eq('combined').all() or not existing.input_method.isin(['ts','counter']).all():
            fail('Existing aggregate must have one combined ts/counter row per date')
        if not existing.date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            fail('Existing aggregate dates must use YYYY-MM-DD format')
        returns = pd.to_numeric(existing.chg_pct, errors='coerce')
        if not np.isfinite(returns).all() or not existing.cls.eq(returns.map(label)).all():
            fail('Existing labels are incompatible with the fixed 0.5% neutral threshold')
    incoming = pd.DataFrame([row], columns=existing.columns)
    retained = existing.loc[existing.date.astype(str) != row['date']].copy()
    merged = pd.concat([retained, incoming], ignore_index=True).sort_values('date').reset_index(drop=True)
    old_row = existing.loc[existing.date.astype(str) == row['date']].reset_index(drop=True)
    same = False
    if len(old_row) == 1:
        try:
            pd.testing.assert_frame_equal(old_row, incoming, check_dtype=False, check_exact=True)
            same = True
        except AssertionError:
            pass
    return merged, same


def publish_aggregate(api, floorsheet_repo, day):
    """Upsert one day. Separate commit from raw upload; reruns repair partial success."""
    started = time.monotonic()
    floor_revision = api.repo_info(floorsheet_repo, repo_type='dataset').sha
    index_revision = api.repo_info(INDEX_REPO, repo_type='dataset').sha
    closes = _read_index_file(_hf_download(api, INDEX_REPO, INDEX_FILE, index_revision))
    if day not in closes.index:
        fail(f'{day}: official closing index is not yet in {INDEX_REPO}/{INDEX_FILE}. '
             f'Raw floorsheet is retained. Refresh the index, then rerun --date {day} --aggregate-only.')
    earlier = closes.index[closes.index < day]
    if len(earlier) < INDEX_DAILY_DAYS + 1:
        fail(f'{day}: insufficient prior index history')
    previous = str(earlier[-1])
    floor_files = set(api.list_repo_files(floorsheet_repo, repo_type='dataset', revision=floor_revision))
    for needed in (day, previous):
        if f'{needed}.parquet' not in floor_files:
            fail(f'Missing {floorsheet_repo}/{needed}.parquet. Aggregation needs the full '
                 'previous trading session, not the previous calendar day. '
                 'Publish that daily file first; the historical master is not downloaded by this job.')
    current = _read_complete_daily(_hf_download(api, floorsheet_repo, f'{day}.parquet', floor_revision), day)
    prior = _read_complete_daily(_hf_download(api, floorsheet_repo, f'{previous}.parquet', floor_revision), previous)
    row = make_aggregate_row(day, previous, closes, current, prior)
    identity = {'version':AGG_VERSION, 'settings':S}
    for attempt in range(3):
        revision = api.repo_info(AGG_REPO, repo_type='dataset').sha
        files = set(api.list_repo_files(AGG_REPO, repo_type='dataset', revision=revision))
        if AGG_FILE not in files:
            fail(f'{AGG_REPO}/{AGG_FILE} is missing. Upload your existing aggregate seed first.')
        existing = pd.read_parquet(_hf_download(api, AGG_REPO, AGG_FILE, revision))
        metadata = {}
        if AGG_META in files:
            metadata = json.loads(_hf_download(api, AGG_REPO, AGG_META, revision).read_text())
            if metadata.get('aggregation') != identity:
                fail('Saved aggregation settings differ. Rebuild the aggregate rather than mixing feature definitions.')
        else:
            print('[aggregate] Adopting existing non-broker seed; schema/labels checked. '
                  'Its original filters must match S in this script.', flush=True)
        merged, unchanged = merge_aggregate(existing, row)
        if unchanged and metadata:
            print(f'[AGGREGATE SUCCESS] {day} unchanged; no extra commit; {time.monotonic()-started:.1f}s', flush=True)
            return {'status':'UNCHANGED','date':day,'repo':AGG_REPO,'rows':len(merged),'revision':revision}
        metadata['aggregation'] = identity
        metadata.setdefault('daily_sources', {})[day] = {
            'floorsheets_repo':floorsheet_repo, 'floorsheets_revision':floor_revision,
            'index_repo':INDEX_REPO, 'index_revision':index_revision, 'previous_session':previous,
        }
        metadata['updated_at'] = datetime.now(UTC).isoformat()
        metadata['n_dates'] = len(merged)
        output = OUT_DIR / AGG_FILE
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output, index=False)
        try:
            commit = api.create_commit(
                repo_id=AGG_REPO, repo_type='dataset', parent_commit=revision,
                commit_message=f'Update aggregated NEPSE training row {day}',
                operations=[
                    CommitOperationAdd(path_in_repo=AGG_FILE, path_or_fileobj=str(output)),
                    CommitOperationAdd(path_in_repo=AGG_META, path_or_fileobj=json.dumps(metadata, indent=2).encode()),
                ],
            )
        except HfHubHTTPError as exc:
            if getattr(exc.response, 'status_code', None) in (409, 412) and attempt < 2:
                print('[aggregate] Concurrent update: reloading latest aggregate before retry', flush=True)
                continue
            raise
        url = str(getattr(commit, 'commit_url', ''))
        print(f'[AGGREGATE SUCCESS] {day}: {row["input_method"]}, class={row["cls"]}, '
              f'{len(merged)} dates; {time.monotonic()-started:.1f}s; {url}', flush=True)
        return {'status':'UPLOADED','date':day,'repo':AGG_REPO,'rows':len(merged),'commit_url':url,'revision':getattr(commit,'oid',None)}
    fail('Aggregate commit could not be completed')


# Training uses exactly the original non-broker Logistic Regression + Ridge pipeline.
import hashlib
import platform
import joblib
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, log_loss, confusion_matrix,
                             mean_absolute_error, mean_squared_error)
MODEL_REPO = os.getenv("HF_MODEL_REPO_ID", "ranjansummit/nepse-p33-models")
FEATURES = FEATS_TX + FEATS_PX
CLASSES = ["bear", "deer", "bull"]
SPLIT_SEED = 42
OUT = OUT_DIR / "trained_model"
TRAINER_VERSION = "combined_random70_15_15_logistic_ridge_v2"


def log(message):
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def usable_rows(df):
    return df[
        (df["n_elig"] >= S["prediction_min_transactions"])
        & df["w_ret"].notna()
    ].copy()

def combine_prepared_rows(prepared):
    """
    Convert the existing counter/ts prepared dataset into ONE row
    per date. Timestamp features take priority when usable.

    Also accepts an already-combined dataset.
    """
    data = prepared.copy()

    if "input_method" not in data.columns:
        data["input_method"] = data["mode"]

    if not data["input_method"].isin(["ts", "counter"]).all():
        raise ValueError("Expected ts/counter input methods")

    missing = [
        column
        for column in FEATURES + ["date", "cls", "chg_pct"]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Prepared dataset lacks: {missing}")

    usable = (
        (data["n_elig"] >= S["prediction_min_transactions"])
        & data["w_ret"].notna()
    )

    # Priority:
    # 0 = usable timestamp window
    # 1 = counter fallback
    # 2 = unusable timestamp row, only if no counter exists
    data["_priority"] = np.where(
        (data["input_method"] == "ts") & usable,
        0,
        np.where(data["input_method"] == "counter", 1, 2),
    )

    data = (
        data.sort_values(
            ["date", "_priority"], kind="stable"
        )
        .drop_duplicates("date", keep="first")
        .drop(columns="_priority")
        .sort_values("date")
        .reset_index(drop=True)
    )

    data["mode"] = "combined"

    log(
        f"COMBINED: {len(data)} dates; "
        f"input methods={data['input_method'].value_counts().to_dict()}"
    )

    return data

def fit_models(train):
    if train["cls"].nunique() != 3:
        raise ValueError(
            "Training data must contain bear, deer and bull"
        )

    classifier_train = train

    if S.get("isBalanced", False):
        count = int(train["cls"].value_counts().min())
        classifier_train = (
            train.sort_values("date")
            .groupby("cls", group_keys=False)
            .tail(count)
        )

    classifier = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    )

    regressor = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    classifier.fit(
        classifier_train[FEATURES],
        classifier_train["cls"],
    )

    regressor.fit(
        train[FEATURES],
        train["chg_pct"],
    )

    return classifier, regressor

def infer(models, data):
    classifier, regressor = models

    probabilities = classifier.predict_proba(
        data[FEATURES]
    )

    order = list(classifier.classes_)
    probabilities = probabilities[
        :, [order.index(c) for c in CLASSES]
    ]

    predicted_class = np.array(CLASSES)[
        probabilities.argmax(axis=1)
    ]

    predicted_pct = regressor.predict(
        data[FEATURES]
    )

    return predicted_class, probabilities, predicted_pct

def evaluate(models, data, majority):
    predicted_class, probabilities, predicted_pct = infer(
        models, data
    )

    predicted_points = (
        predicted_pct * data["prev_close"].to_numpy()
    )

    metrics = {
        "n_dates": len(data),
        "earliest_date": data["date"].min(),
        "latest_date": data["date"].max(),
        "class_counts": data["cls"].value_counts().to_dict(),
        "input_method_counts": (
            data["input_method"].value_counts().to_dict()
        ),
        "accuracy": accuracy_score(
            data["cls"], predicted_class
        ),
        "macro_f1": f1_score(
            data["cls"],
            predicted_class,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            data["cls"],
            probabilities[:, [CLASSES.index(c) for c in sorted(CLASSES)]],
            labels=sorted(CLASSES),
        ),
        "confusion_matrix": confusion_matrix(
            data["cls"],
            predicted_class,
            labels=CLASSES,
        ).tolist(),
        "confusion_order": CLASSES,
        "majority_baseline_accuracy": float(
            (data["cls"] == majority).mean()
        ),
        "regression_mae_points": mean_absolute_error(
            data["chg"], predicted_points
        ),
        "regression_rmse_points": float(
            np.sqrt(
                mean_squared_error(
                    data["chg"], predicted_points
                )
            )
        ),
        "previous_close_baseline_mae_points": float(
            data["chg"].abs().mean()
        ),
        "direction_hit": float(
            (
                np.sign(predicted_pct)
                == np.sign(data["chg_pct"].to_numpy())
            ).mean()
        ),
    }

    predictions = data[
        [
            "date", "input_method", "cls",
            "chg_pct", "prev_close", "close",
        ]
    ].copy()

    predictions["predicted_class"] = predicted_class
    predictions["predicted_change_pct"] = predicted_pct
    predictions["predicted_close"] = (
        data["prev_close"].to_numpy() * (1 + predicted_pct)
    )

    for i, cls in enumerate(CLASSES):
        predictions[f"p_{cls}"] = probabilities[:, i]

    return metrics, predictions

def train_and_report(dataset, revisions):
    dataset = combine_prepared_rows(dataset)

    dataset.to_parquet(
        OUT / "combined_dataset.parquet",
        index=False,
    )

    data = usable_rows(dataset).sort_values("date")

    # Keep the split based on all prepared dates, as before.
    dates = np.array(sorted(dataset["date"].unique()))

    if len(dates) < 30:
        raise RuntimeError("At least 30 prepared dates required")

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(dates).tolist()

    a = int(len(shuffled) * 0.70)
    b = int(len(shuffled) * 0.85)

    groups = {
        "train": sorted(shuffled[:a]),
        "validation": sorted(shuffled[a:b]),
        "test": sorted(shuffled[b:]),
    }

    pieces = {
        name: data[data["date"].isin(selected)].copy()
        for name, selected in groups.items()
    }

    counts = {
        name: len(frame)
        for name, frame in pieces.items()
    }

    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Empty usable split: {counts}")

    log(
        f"COMBINED RANDOM SPLIT | seed={SPLIT_SEED} | "
        f"usable counts={counts}"
    )

    # Fit on training dates only to obtain held-out scores.
    evaluation_models = fit_models(pieces["train"])
    majority = pieces["train"]["cls"].mode().iloc[0]

    report = {
        "target": "same_day_close_vs_previous_close",
        "input_selection": "usable_timestamp_window_else_counter",
        "neutral_percentage": S["neutral_percentage"],
        "source_revisions": revisions,
        "settings": S.copy(),
        "split_method": "random_by_date",
        "split_seed": SPLIT_SEED,
        "split_dates": groups,
        "usable_split_counts": counts,
        "classifier": "LogisticRegression(C=1.0)",
        "regressor": "Ridge(alpha=1.0)",
        "validation_used_for_tuning": False,
        "log_loss_probability_order": sorted(CLASSES),
    }

    for split in ("validation", "test"):
        metrics, predictions = evaluate(
            evaluation_models,
            pieces[split],
            majority,
        )

        report[split] = metrics

        predictions.to_csv(
            OUT / f"combined_{split}_predictions.csv",
            index=False,
        )

    log(
        "COMBINED RANDOM TEST: "
        + json.dumps(report["test"], default=str)
    )

    # Final fit on every usable day, saved in ONE file.
    production_models = fit_models(data)

    model_path = OUT / "combined_production.joblib"

    joblib.dump(
        {
            "models": production_models,
            "features": FEATURES,
            "classes": CLASSES,
            "settings": S.copy(),
            "input_selection": "usable_timestamp_window_else_counter",
            "trained_through": data["date"].max(),
            "training_dates": data["date"].tolist(),
            "n_training_dates": len(data),
            "source_revisions": revisions,
        },
        model_path,
    )

    report["production"] = {
        "file": model_path.name,
        "n_training_dates": len(data),
        "trained_through": data["date"].max(),
    }

    (OUT / "combined_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )

    log(
        f"SAVED: {model_path.name} | "
        f"trained on all {len(data)} usable dates"
    )

    return report

def train_and_publish(api, aggregate_revision=None):
    """Evaluate on held-out dates; refit on ALL usable dates; upload ONE dated bundle."""
    started = time.monotonic()
    revision = aggregate_revision or api.repo_info(AGG_REPO, repo_type="dataset").sha
    files = set(api.list_repo_files(AGG_REPO, repo_type="dataset", revision=revision))
    if AGG_FILE not in files:
        fail(f"Missing {AGG_REPO}/{AGG_FILE}")
    cached = _hf_download(api, AGG_REPO, AGG_FILE, revision)
    dataset = pd.read_parquet(cached)
    if dataset.empty:
        fail("Aggregate dataset is empty")
    # Validate the exact non-broker schema, dates and labels without altering rows.
    merge_aggregate(dataset, dataset.iloc[0].to_dict())
    for field in FEATURES + ['prev_close','close','chg','chg_pct']:
        dataset[field] = pd.to_numeric(dataset[field], errors='raise')
    if np.isinf(dataset[FEATURES].to_numpy(dtype=float)).any():
        fail('Infinite aggregate feature values; repair the aggregate before training')
    target_values = dataset[['prev_close','close','chg','chg_pct']].to_numpy(dtype=float)
    if not np.isfinite(target_values).all() or not (dataset[['prev_close','close']] > 0).all().all():
        fail('Missing/invalid index targets in aggregate')
    if not np.allclose(dataset['close'] - dataset['prev_close'], dataset['chg'], rtol=1e-9, atol=1e-9):
        fail('Aggregate change points do not match index closes')
    if not np.allclose(dataset['chg'] / dataset['prev_close'], dataset['chg_pct'], rtol=1e-9, atol=1e-9):
        fail('Aggregate change fractions do not match index closes')
    if dataset.date.max() > datetime.now(NEPAL).strftime('%Y-%m-%d'):
        fail('Aggregate contains future trading dates')
    sources = {'aggregate_repo':AGG_REPO, 'aggregate_revision':revision,
               'aggregate_file':AGG_FILE, 'aggregate_sha256':hashlib.sha256(cached.read_bytes()).hexdigest()}
    if AGG_META in files:
        meta = json.loads(_hf_download(api, AGG_REPO, AGG_META, revision).read_text())
        if meta.get('aggregation') != {'version':AGG_VERSION,'settings':S}:
            fail('Aggregate metadata/settings do not match this trainer')
        sources['daily_sources'] = meta.get('daily_sources', {})
    else:
        log('Training existing seed: original non-broker aggregation settings assumed; schema and targets validated.')
    # Read before fitting; reject a concurrent model-repo change rather than silently overwrite it.
    model_revision = api.repo_info(MODEL_REPO, repo_type='model').sha
    OUT.mkdir(parents=True, exist_ok=True)
    report = train_and_report(dataset, sources)
    bundle_path = OUT / 'combined_production.joblib'
    bundle = joblib.load(bundle_path)  # This file was just written by this process.
    trained_through = str(bundle['trained_through'])
    model_name = f'combined_production_{trained_through}.joblib'
    report_name = f'combined_report_{trained_through}.json'
    versions = {'python':platform.python_version(), 'scikit_learn':sklearn.__version__,
                'numpy':np.__version__, 'pandas':pd.__version__, 'joblib':joblib.__version__}
    bundle['library_versions'] = versions
    bundle['trainer_version'] = TRAINER_VERSION
    bundle['created_at'] = datetime.now(UTC).isoformat()
    bundle['training_start_date'] = min(bundle['training_dates'])
    dated_path = OUT / model_name
    joblib.dump(bundle, dated_path)
    report['production']['file'] = model_name
    report['production']['repo'] = MODEL_REPO
    report['library_versions'] = versions
    report['trainer_version'] = TRAINER_VERSION
    report['created_at'] = bundle['created_at']
    report['evaluation_note'] = ('Random-date held-out results belong to the evaluation fit. '
        'The published model is refitted on every usable date, including held-out dates; '
        'those scores are not an independent evaluation of the refitted model.')
    report_path = OUT / report_name
    report_path.write_text(json.dumps(report, indent=2, default=str))
    operations = [CommitOperationAdd(path_in_repo=model_name, path_or_fileobj=str(dated_path)),
                  CommitOperationAdd(path_in_repo=report_name, path_or_fileobj=str(report_path))]
    for split in ('validation','test'):
        operations.append(CommitOperationAdd(
            path_in_repo=f'combined_{split}_predictions_{trained_through}.csv',
            path_or_fileobj=str(OUT / f'combined_{split}_predictions.csv')))
    commit = api.create_commit(repo_id=MODEL_REPO, repo_type='model', parent_commit=model_revision,
        commit_message=f'Train combined NEPSE model through {trained_through}', operations=operations)
    url = str(getattr(commit,'commit_url',''))
    log(f'[MODEL UPLOAD SUCCESS] {MODEL_REPO}/{model_name}; '
        f'{bundle["n_training_dates"]} training dates; {time.monotonic()-started:.1f}s; {url}')
    return {'status':'UPLOADED','repo':MODEL_REPO,'file':model_name,
            'trained_through':trained_through, 'n_training_dates':bundle['n_training_dates'], 'commit_url':url}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(NEPAL).strftime("%Y-%m-%d"), help="Nepal trading date YYYY-MM-DD; default today")
    parser.add_argument("--expected-count", type=int, help="Known official unique-transaction count, if available")
    parser.add_argument("--allow-historical", action="store_true", help="Explicitly allow dates overlapping V2 master")
    parser.add_argument("--replace-existing", action="store_true", help="Explicitly overwrite an existing daily HF file")
    parser.add_argument("--yes", action="store_true", help="Upload without confirmation (use only for deliberate automation)")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild one aggregate row from already-published daily floorsheets; no API fetch")
    parser.add_argument("--train-only", action="store_true", help="Train/publish from HF aggregate; skip raw fetch and aggregation")
    args = parser.parse_args(argv)
    if args.train_only:
        if args.aggregate_only or args.replace_existing or args.expected_count is not None:
            parser.error("--train-only cannot be combined with raw/aggregation options")
        auto_confirm = args.yes or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        if not auto_confirm and input(f"Train from {AGG_REPO} and publish to {MODEL_REPO}? [y/N]: ").strip().lower() != "y":
            return 0
        try:
            train_and_publish(HfApi(token=os.getenv("HF_TOKEN") or None))
            return 0
        except Exception as exc:
            print(f"[TRAIN/PUBLISH FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    target_day = date.fromisoformat(args.date)
    now_nepal = datetime.now(NEPAL)
    if target_day > now_nepal.date():
        fail("Cannot fetch a future date")
    if target_day == now_nepal.date() and now_nepal.time() < dtime(16, 0):
        fail("Today is still within the post-market buffer (before 16:00 Nepal time). Run after 16:00.")
    if target_day <= MASTER_LAST_DATE and not args.allow_historical and not args.aggregate_only:
        fail(f"{args.date} may already exist in V2 master (through {MASTER_LAST_DATE}). "
             "Refusing duplicate historical publication; use --allow-historical only if intentional.")
    if args.expected_count is not None and args.expected_count < 0:
        fail("--expected-count must not be negative")
    repo = os.getenv("HF_REPO_ID", REPO_DEFAULT).strip()
    if not repo:
        fail("HF_REPO_ID is blank")
    print(f"[source] {BASE_URL}?date={args.date}&page=N&page_size={PAGE_SIZE}")
    print(f"[target] {repo} / {args.date}.parquet")
    path = None
    report = {"date": args.date, "status": "ERROR_NO_UPLOAD"}
    try:
        api = HfApi(token=os.getenv("HF_TOKEN") or None)
        remote = f"{args.date}.parquet"
        existing = set(api.list_repo_files(repo, repo_type="dataset"))
        raw_exists = remote in existing
        auto_confirm = args.yes or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        if args.aggregate_only or (raw_exists and not args.replace_existing):
            if not raw_exists:
                fail(f"{repo}/{remote} does not exist; run without --aggregate-only first")
            print(f"[raw retained] {repo}/{remote}; continuing to aggregate")
            report["status"] = "RAW_EXISTS_AGGREGATE_PENDING"
            if not auto_confirm and input(f"Refresh official index in {INDEX_REPO}, update {AGG_REPO}/{AGG_FILE}, train and publish to {MODEL_REPO}? [y/N]: ").strip().lower() != "y":
                report["status"] = "USER_DECLINED_AGGREGATE"
                save_report(args.date, report)
                return 0
            report["index"] = refresh_official_index(api, args.date)
            save_report(args.date, report)
            report["aggregate"] = publish_aggregate(api, repo, args.date)
            report["status"] = "AGGREGATE_SUCCESS_MODEL_PENDING"
            save_report(args.date, report)
            report["model"] = train_and_publish(api, report["aggregate"].get("revision"))
            report["status"] = "RAW_EXISTS_AGGREGATE_AND_MODEL_SUCCESS"
            save_report(args.date, report)
            return 0
        path, report = collect_day(args.date, args.expected_count)
        if path is None:
            save_report(args.date, report)
            print("[no upload] No transactions received (holiday / API may be delayed).")
            return 0
        print("[validated]", report["unique_rows"], "unique transactions")
        # GitHub Actions cannot provide interactive input. Running in CI is
        # an explicit automation context, so auto-confirm upload there.
        auto_confirm = args.yes or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        if auto_confirm:
            print("[upload] Automatic confirmation: GitHub Actions or --yes")
        else:
            prompt = f"Upload {remote}, refresh official index in {INDEX_REPO}, update {AGG_REPO}/{AGG_FILE}, train and publish to {MODEL_REPO}? [y/N]: "
            if input(prompt).strip().lower() != "y":
                print("[no upload] Local validated Parquet kept.")
                report["status"] = "USER_DECLINED_UPLOAD"
                save_report(args.date, report)
                return 0
        commit = api.upload_file(
            path_or_fileobj=str(path), path_in_repo=remote,
            repo_id=repo, repo_type="dataset",
            commit_message=f"Add validated NEPSE floorsheet {args.date} (V2 schema)",
        )
        report["status"] = "UPLOADED"
        report["commit_url"] = str(getattr(commit, "commit_url", ""))
        report["repo_path"] = remote
        save_report(args.date, report)
        print("[UPLOAD SUCCESS]", report["commit_url"])
        report["status"] = "RAW_UPLOADED_AGGREGATE_PENDING"
        save_report(args.date, report)
        report["index"] = refresh_official_index(api, args.date)
        save_report(args.date, report)
        report["aggregate"] = publish_aggregate(api, repo, args.date)
        report["status"] = "RAW_AND_AGGREGATE_SUCCESS_MODEL_PENDING"
        save_report(args.date, report)
        report["model"] = train_and_publish(api, report["aggregate"].get("revision"))
        report["status"] = "RAW_AGGREGATE_AND_MODEL_SUCCESS"
        save_report(args.date, report)
        return 0
    except Exception as exc:
        prior_status = report.get("status", "UNKNOWN")
        report["status"] = "FAILED"
        report["completed_stage"] = prior_status
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_report(args.date, report)
        print("[FAILED]", report["error"], file=sys.stderr)
        print("Raw, aggregate and model uploads are separate commits. Earlier steps may already be saved; rerun the same --date, or --train-only if only training/publishing failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
