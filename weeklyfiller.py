"""Weekly production floorsheet, official index and aggregate repair."""
from future import annotations

import hashlib
import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
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

The V2 master assembled through this date. Keep later daily updates separate.

MASTER_LAST_DATE = date(2026, 10, 30)
BASE_DIR = Path(file).resolve().parent
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
"""UTC-aware datetime or None. API traded_at is interpreted as UTC.

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
fail(f"page {page}: expected JSON object or list, got {type(payload).name}")
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



Daily aggregation: original non-broker training features, no model training.

import io
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError

AGG_REPO = "ranjansummit/aggregatedindices"
AGG_FILE = "combined_dataset.parquet"
AGG_META = "combined_dataset.metadata.json"
INDEX_REPO = "ranjansummit/dailyindices"
INDEX_FILE = "NEPSE_INDEX.csv"
AGG_VERSION = "same_day_original_10_guarded_v1"

Must match the notebook that created the initial combined_dataset.parquet.

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

Official website client: same WASM-token approach as the earlier client.py.

No third-party price/index fallback. The public website session needs no account key.

class OfficialNepseIndex:
def init(self, origin):
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
errors.append(f'{origin}: {type(exc).name}: {str(exc)[:200]}')
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

def official_sessions(start, end):
errors = []
for origin in ("https://www.nepalstock.com", "https://nepalstock.com.np"):
client = None
try:
client = OfficialNepseIndex(origin)
records = {}
for page in range(100):
data = client.history_page(page)
rows = data["content"]
if not rows:
break
dates = [_official_day(row["businessDate"]) for row in rows]
if page and max(dates) >= oldest:
raise ValueError("Unexpected index pagination order")
oldest = min(dates)
records.update(zip(dates, rows))
if oldest < start or data.get("last"):
break
else:
raise ValueError("Index pagination limit exceeded")
return {k: v for k, v in records.items() if k <= end}
except Exception as exc:
errors.append(str(exc))
finally:
if client is not None:
client.close()
raise RuntimeError("Official index unavailable: " + " | ".join(errors))

def main():
parser = argparse.ArgumentParser()
parser.add_argument("--week-of", type=date.fromisoformat)
args = parser.parse_args()
today = datetime.now(NEPAL).date()
anchor = args.week_of or today
monday = anchor - timedelta(days=anchor.weekday())
if monday > today:
parser.error("Choose the current week or an earlier week")
start = monday.isoformat()
end = min(today, monday + timedelta(days=6)).isoformat()
records = official_sessions(start, end)
days = sorted(k for k in records if start <= k <= end)
print(f"Week {start} through {end}. Official sessions: {days}", flush=True)
if not days:
return
api = HfApi(token=os.environ["HF_TOKEN"])
raw_repo = REPO_DEFAULT
rev = api.repo_info(INDEX_REPO, repo_type="dataset").sha
existing = pd.read_csv(_hf_download(api, INDEX_REPO, INDEX_FILE, rev))
merged = merge_official_index(existing, [records[k] for k in days], days[-1])
ops = []
if not existing.reset_index(drop=True).equals(merged.reset_index(drop=True)):
ops.append(CommitOperationAdd(path_in_repo=INDEX_FILE,
path_or_fileobj=merged.to_csv(index=False).encode()))
# Store the selected week's index parquet; the CSV remains the aggregate's input.
key = "published_date" if "published_date" in merged else "date"
week = merged.loc[merged[key].isin(days)].reset_index(drop=True)
name = f"weekly_indices/{start}.parquet"
payload = week.to_parquet(index=False)
files = api.list_repo_files(INDEX_REPO, repo_type="dataset", revision=rev)
if name not in files or not pd.read_parquet(_hf_download(api, INDEX_REPO, name, rev)).equals(week):
ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=payload))
if ops:
api.create_commit(repo_id=INDEX_REPO, repo_type="dataset", parent_commit=rev,
operations=ops, commit_message=f"Repair index week {start}")
raw = set(api.list_repo_files(raw_repo, repo_type="dataset"))
ag_rev = api.repo_info(AGG_REPO, repo_type="dataset").sha
path = _hf_download(api, AGG_REPO, AGG_FILE, ag_rev)
aggregate_dates = set(pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"]).dt.strftime("%Y-%m-%d"))
failures = []
for day in days:
try:
filename = f"{day}.parquet"
missing_raw = filename not in raw
if missing_raw:
path, _ = collect_day(day, None)
if path is None:
raise ValueError("No raw floorsheet available")
api.upload_file(repo_id=raw_repo, repo_type="dataset", path_in_repo=filename,
path_or_fileobj=str(path), commit_message=f"Fill missing floorsheet {day}")
raw.add(filename)
if day not in aggregate_dates or missing_raw:
previous = max((k for k in records if k < day), default=None)
if not previous or f"{previous}.parquet" not in raw:
raise ValueError(f"Previous session {previous} is required; repair its week first")
publish_aggregate(api, raw_repo, day)
print(f"[OK] {day}", flush=True)
except Exception as exc:
failures.append(f"{day}: {exc}")
if failures:
raise SystemExit("\n".join(failures))

if name == "main":
main()
