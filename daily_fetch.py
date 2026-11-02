"""Fetch one complete NepseAlerts trading day and publish YYYY-MM-DD.parquet to HF.

Never reads/overwrites the historical NEPSE_ALL_MERGED.parquet. Normalizes into
its 9-column V2 schema, validates a complete paginated fetch, and uploads only
one root-level daily file after confirmation (automatic in GitHub Actions).

Usage:
    python nepse_daily_v2.py
    python nepse_daily_v2.py --date 2026-11-02
    python nepse_daily_v2.py --date 2026-11-02 --expected-count 70000
    python nepse_daily_v2.py --date 2026-11-02 --yes

Environment: HF_TOKEN (or `hf auth login`), HF_REPO_ID (optional).
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
    pa.field("business_date", pa.string()),
    pa.field("contract_id", pa.string()),
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
        "business_date": business_date,
        "contract_id": contract_id,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(NEPAL).strftime("%Y-%m-%d"), help="Nepal trading date YYYY-MM-DD; default today")
    parser.add_argument("--expected-count", type=int, help="Known official unique-transaction count, if available")
    parser.add_argument("--allow-historical", action="store_true", help="Explicitly allow dates overlapping V2 master")
    parser.add_argument("--replace-existing", action="store_true", help="Explicitly overwrite an existing daily HF file")
    parser.add_argument("--yes", action="store_true", help="Upload without confirmation (use only for deliberate automation)")
    args = parser.parse_args(argv)
    target_day = date.fromisoformat(args.date)
    now_nepal = datetime.now(NEPAL)
    if target_day > now_nepal.date():
        fail("Cannot fetch a future date")
    if target_day == now_nepal.date() and now_nepal.time() < dtime(16, 0):
        fail("Today is still within the post-market buffer (before 16:00 Nepal time). Run after 16:00.")
    if target_day <= MASTER_LAST_DATE and not args.allow_historical:
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
        path, report = collect_day(args.date, args.expected_count)
        if path is None:
            save_report(args.date, report)
            print("[no upload] No transactions received (holiday / API may be delayed).")
            return 0
        print("\n[validated]", report["unique_rows"], "unique transactions;", report["raw_api_rows"], "API rows;")
        print("[schema]", ", ".join(COLUMNS))
        print("[time] UTC timestamp; null where missing")
        print("[note] No direct comparison to the exchange's full trade count unless --expected-count is provided.")
        print("[note] This adds a separate day file; the 70M-row master is NOT updated.")
        api = HfApi(token=os.getenv("HF_TOKEN") or None)
        # Read HF filenames before writing; fail closed on repo/auth errors.
        existing = set(api.list_repo_files(repo, repo_type="dataset"))
        remote = f"{args.date}.parquet"
        if remote in existing and not args.replace_existing:
            print(f"[no upload] {remote} already exists in HF. Use --replace-existing to overwrite intentionally.")
            report["status"] = "REMOTE_ALREADY_EXISTS_NO_UPLOAD"
            save_report(args.date, report)
            return 0
        # GitHub Actions cannot provide interactive input. Running in CI is
        # an explicit automation context, so auto-confirm upload there.
        auto_confirm = args.yes or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        if auto_confirm:
            print("[upload] Automatic confirmation: GitHub Actions or --yes")
        else:
            prompt = f"Upload ONE file '{remote}' to {repo} as a new commit? [y/N]: "
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
        return 0
    except Exception as exc:
        report["status"] = "ERROR_NO_UPLOAD_OR_UPLOAD_UNCONFIRMED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_report(args.date, report)
        print("[FAILED]", report["error"], file=sys.stderr)
        print("Check the HF repo before retrying if a network error occurred during upload.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
