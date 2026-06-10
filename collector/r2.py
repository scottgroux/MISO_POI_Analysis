"""
r2.py
-----
Cloudflare R2 client for durable Parquet storage.

R2 is S3-compatible; boto3 connects via a custom endpoint_url.
All credentials are loaded from environment variables (never hardcoded).

Required env vars (set in .env or platform secrets):
  R2_ACCOUNT_ID      — Cloudflare account ID (32-char hex)
  R2_ACCESS_KEY_ID   — R2 API token access key
  R2_SECRET_ACCESS_KEY — R2 API token secret
  R2_BUCKET          — bucket name (e.g. "miso-lmp")

Object key layout mirrors the local Hive partition layout:
  lmp/date=YYYY-MM-DD/part.parquet
"""

import io
import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Load .env from the project root (two levels up from this file).
# No-op if python-dotenv is not installed or if env vars are already set.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

log = logging.getLogger(__name__)

# Key prefix inside the bucket — keeps the bucket usable for other datasets.
KEY_PREFIX = "lmp"


def _client():
    """Return a boto3 S3 client pointed at Cloudflare R2."""
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _bucket() -> str:
    return os.environ.get("R2_BUCKET") or os.environ["R2_BUCKET_NAME"]


# ── Public API ────────────────────────────────────────────────────────────────

def upload(local_path: Path, date_str: str) -> None:
    """Upload a local Parquet file to R2 under lmp/date=<date_str>/part.parquet."""
    key = f"{KEY_PREFIX}/date={date_str}/part.parquet"
    try:
        _client().upload_file(str(local_path), _bucket(), key)
        log.info("R2 upload OK: %s", key)
    except ClientError as e:
        log.error("R2 upload failed for %s: %s", key, e)
        raise


def download(date_str: str, local_path: Path) -> bool:
    """Download lmp/date=<date_str>/part.parquet from R2 to local_path.
    Returns True on success, False if the object does not exist."""
    key = f"{KEY_PREFIX}/date={date_str}/part.parquet"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _client().download_file(_bucket(), key, str(local_path))
        log.info("R2 download OK: %s → %s", key, local_path)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        log.error("R2 download failed for %s: %s", key, e)
        raise


def list_dates() -> list[str]:
    """List all date partitions stored in R2 (under lmp/date=*/part.parquet).
    Returns a sorted list of 'YYYY-MM-DD' strings, or [] if R2 is not configured."""
    if not r2_enabled():
        return []
    client = _client()
    bucket = _bucket()
    prefix = f"{KEY_PREFIX}/date="

    dates = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            # cp["Prefix"] looks like: lmp/date=2024-03-01/
            segment = cp["Prefix"].rstrip("/").split("/")[-1]  # date=2024-03-01
            if segment.startswith("date="):
                dates.append(segment.replace("date=", ""))

    return sorted(dates)


def upload_json(data, key: str) -> None:
    """Upload a JSON-serializable object to R2 at the given key.
    Key should not have a leading slash, e.g. 'summaries/meta.json'."""
    import json
    body = json.dumps(data, separators=(",", ":")).encode()
    try:
        _client().put_object(
            Bucket=_bucket(),
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        log.info("R2 JSON upload OK: %s (%d bytes)", key, len(body))
    except ClientError as e:
        log.error("R2 JSON upload failed for %s: %s", key, e)
        raise


def download_json(key: str):
    """Download and parse a JSON object from R2. Returns None if the key does not exist."""
    import json
    try:
        resp = _client().get_object(Bucket=_bucket(), Key=key)
        return json.loads(resp["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        log.error("R2 JSON download failed for %s: %s", key, e)
        raise


def r2_enabled() -> bool:
    """Return True if all required R2 env vars are set."""
    has_bucket = "R2_BUCKET" in os.environ or "R2_BUCKET_NAME" in os.environ
    return (
        "R2_ACCOUNT_ID" in os.environ
        and "R2_ACCESS_KEY_ID" in os.environ
        and "R2_SECRET_ACCESS_KEY" in os.environ
        and has_bucket
    )
