"""Daily NEPSE floorsheet → parquet → Hugging Face.
Uses the exact URL pattern:
  https://nepsealerts.com/api/floorsheet?date=YYYY-MM-DD&page=N&page_size=200
Loops pages until an empty response or error is returned.
"""
import os
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# --- CONFIG ---
BASE_URL = "https://nepsealerts.com/api/floorsheet"
PAGE_SIZE = 200  # Hardcoded maximum allowed by the API
HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = os.environ["HF_REPO_ID"]
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))

# --- TARGET SCHEMA ---
TARGET_COLUMNS = [
    "business_date", "contract_id", "symbol", "buyer", "seller",
    "quantity", "rate", "amount", "trade_time", "id",
]
ARROW_SCHEMA = pa.schema([
    ("business_date", pa.string()),
    ("contract_id", pa.string()),
    ("symbol", pa.string()),
    ("buyer", pa.string()),
    ("seller", pa.string()),
    ("quantity", pa.float64()),
    ("rate", pa.float64()),
    ("amount", pa.float64()),
    ("trade_time", pa.string()),
    ("id", pa.string()),
])

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0"

def fetch_all_pages_for_date(date_str: str) -> pd.DataFrame:
    """
    Fetches all pages for a given date using the exact URL pattern.
    Stops when a page returns no results or an error.
    """
    all_rows = []
    page = 1
    while True:
        url = f"{BASE_URL}?date={date_str}&page={page}&page_size={PAGE_SIZE}"
        print(f"[info] Fetching: {url}")
        
        try:
            resp = _session.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"[warn] Page {page} returned status {resp.status_code}. Stopping.")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                print(f"[info] Page {page} is empty. End of data for {date_str}.")
                break

            all_rows.extend(results)
            print(f"[info] Page {page}: fetched {len(results)} rows (total so far: {len(all_rows)})")
            page += 1

            # Be polite to the API between page requests
            time.sleep(0.5)

        except requests.RequestException as e:
            print(f"[error] Request failed for page {page}: {e}. Stopping.")
            break
        except ValueError:
            print(f"[error] Page {page} did not return valid JSON. Stopping.")
            break

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)

def to_target(df: pd.DataFrame) -> pd.DataFrame:
    """Maps the raw API DataFrame to your exact target schema."""
    if df.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    out = pd.DataFrame(index=df.index)
    out["business_date"] = df["business_date"].astype(str)
    out["contract_id"]   = df["contract_id"].astype(str)
    out["symbol"]        = df["symbol"].astype(str)
    out["buyer"]         = df["buyer_broker"].astype(str)
    out["seller"]        = df["seller_broker"].astype(str)
    out["quantity"]      = pd.to_numeric(df["quantity"], errors="coerce").astype("float64")
    out["rate"]          = pd.to_numeric(df["rate"],     errors="coerce").astype("float64")
    out["amount"]        = pd.to_numeric(df["amount"],   errors="coerce").astype("float64")

    # trade_time: HH:MM:SS in Nepal time, derived from 'traded_at'
    t = pd.to_datetime(df["traded_at"], errors="coerce", utc=True).dt.tz_convert(NEPAL_TZ)
    out["trade_time"] = t.dt.strftime("%H:%M:%S").where(t.notna(), None)

    # id: The API does not provide an 'id' field, so it is set to None.
    out["id"] = None

    return out[TARGET_COLUMNS].reset_index(drop=True)

def write_parquet(df: pd.DataFrame, out_dir: str, date_str: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)
    path = os.path.join(out_dir, f"{date_str}.parquet")
    pq.write_table(table, path, compression="snappy")
    return path

def main():
    # Use today's date in Nepal time
    today = datetime.now(NEPAL_TZ).strftime("%Y-%m-%d")
    print(f"[info] Starting fetch for {today} (Nepal date)")

    raw_df = fetch_all_pages_for_date(today)
    if raw_df.empty:
        print(f"[info] No trades found for {today} (holiday?). Nothing to upload.")
        return

    # The API may return duplicate contract IDs; deduplicate by contract_id.
    raw_df = raw_df.drop_duplicates("contract_id")
    print(f"[info] Total unique trades for {today}: {len(raw_df)}")

    target_df = to_target(raw_df)
    parquet_path = write_parquet(target_df, "parquet_out", today)
    print(f"[ok] Wrote {len(target_df)} rows to {parquet_path}")

    # Push to Hugging Face
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path="parquet_out",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add NEPSE floorsheet for {today}",
    )
    print(f"[ok] Uploaded to https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
