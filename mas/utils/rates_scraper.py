"""Market rate collection — competitor bank packages + the MAS SORA benchmark.

Two sources, one schema, one cumulative CSV (``csv_tables/market_rates.csv``):

* ``dollarback`` — competitor home loan packages scraped from
  dollarbackmortgage.com (requests + BeautifulSoup; the rate tables are plain
  server-rendered ``<table>`` markup, so no JS rendering is needed).
* ``mas`` — the official SORA benchmark from the MAS APIMG API, via
  :mod:`utils.rates`. One MAS call returns the full daily series, so history is
  backfilled in a single request rather than accumulated run by run.

Rows carry two distinct timestamps and conflating them will skew any trend
chart: ``scraped_at_utc`` is when we fetched, ``rate_as_of`` is the date the
rate itself belongs to (MAS publication date; blank for scraped packages,
which are only ever "as of now").

There is deliberately no fallback rate. If a source is unreachable its rows are
simply not written and the failure is reported — a stale number that looks live
is worse than a visible gap.

    py -m utils.rates_scraper              # scrape + backfill 90d of SORA
    py -m utils.rates_scraper --sora-days 365
    py -m utils.rates_scraper --no-sora    # competitor packages only
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from utils import netguard
from utils.config import DATA_DIR
from utils.rates import SoraUnavailable, get_sora_history

URL = "https://dollarbackmortgage.com/new-private-property-loan/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-SG,en;q=0.9",
}

# "1.30 % Fixed" / "1.40% Floating" / "1.40 (EMI)% Fixed" / "+0.35%"
RATE_RE = re.compile(
    r"(?P<sign>[+-])?\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<note>\([^)]*\))?\s*%\s*(?P<kind>Fixed|Floating)?",
    re.IGNORECASE,
)

FLOATING_HINTS = ("sora", "fhr", "board", "fdr", "fdmr", "float")

# Whitelist of real banks. Anything not in here (broker promos like
# "Limited Promo*", "Exclusive Rate", footnote rows) is dropped.
# Keys are matched case-insensitively against the BANK cell; the value is the
# canonical name written to the CSV, so a bank never splits into two series
# just because the site relabels it ("BOC" vs "Bank of China").
BANK_WHITELIST = {
    "dbs": "DBS",
    "posb": "DBS",
    "ocbc": "OCBC",
    "uob": "UOB",
    "maybank": "Maybank",
    "hsbc": "HSBC",
    "standard chartered": "Standard Chartered",
    "stanchart": "Standard Chartered",
    "scb": "Standard Chartered",
    "citi": "Citibank",
    "citibank": "Citibank",
    "cimb": "CIMB",
    "rhb": "RHB",
    "boc": "Bank of China",
    "bank of china": "Bank of China",
    "sbi": "State Bank of India",
    "state bank of india": "State Bank of India",
    "hla": "HL Bank",
    "hl bank": "HL Bank",
    "hong leong": "HL Bank",
    "sing investments": "Sing Investments & Finance",
    "singapura finance": "Singapura Finance",
    "hong leong finance": "Hong Leong Finance",
}


def canonical_bank(name: str) -> str | None:
    """Map a BANK cell to its canonical name, or None if it isn't a real bank."""
    key = clean_text(name).lower().strip(" *")
    if key in BANK_WHITELIST:
        return BANK_WHITELIST[key]
    # tolerate suffixes like "DBS Bank" / "UOB Limited"
    for alias, canon in BANK_WHITELIST.items():
        if re.match(rf"^{re.escape(alias)}\b", key):
            return canon
    return None


def fetch(url: str, timeout: int = 30) -> str:
    # Already known unreachable: fail fast instead of stalling on a blocked
    # connection. collect() reports a source failure without inventing a rate, so
    # the only thing that changes is how long the caller waits to learn it.
    if netguard.is_offline():
        raise RuntimeError("no public internet from this host — rate scrape skipped")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as exc:
        netguard.note_failure(exc)
        raise
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


# Typographic chars the CMS emits as HTML entities (&#8217; etc). Normalising
# them keeps the CSV plain-ASCII-friendly for Excel and string matching.
PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "\xa0": " ",
}


def clean_text(s: str) -> str:
    for bad, good in PUNCT_MAP.items():
        s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s).strip()


def cell_texts(row) -> list[str]:
    return [clean_text(c.get_text(" ", strip=True)) for c in row.find_all(["th", "td"])]


def parse_rate(text: str) -> dict:
    """Split a rate cell into value / type / note. Returns empty dict if no rate."""
    m = RATE_RE.search(text or "")
    if not m:
        return {}
    value = float(m.group("value"))
    if m.group("sign") == "-":
        value = -value
    return {
        "value": value,
        "is_spread": m.group("sign") == "+",
        "kind": (m.group("kind") or "").capitalize(),
        "note": (m.group("note") or "").strip("()"),
    }


def is_rate_table(header: list[str]) -> bool:
    joined = " ".join(header).upper()
    return "BANK" in joined and "YEAR 1" in joined


def classify(loan_type: str, year_kinds: list[str]) -> str:
    lt = loan_type.lower()
    if any(h in lt for h in FLOATING_HINTS):
        return "floating"
    if "fixed" in lt:
        return "fixed"
    # fall back to whatever the year cells declared
    if any(k == "Fixed" for k in year_kinds):
        return "fixed"
    if any(k == "Floating" for k in year_kinds):
        return "floating"
    return "unknown"


def lock_in_years(loan_type: str) -> str:
    """'2 Year Fixed' -> 2. Blank when the label doesn't state a lock-in."""
    m = re.search(r"(\d+)\s*[- ]?\s*year", loan_type, re.I)
    return m.group(1) if m else ""


def benchmark(loan_type: str) -> str:
    """Floating peg as shown on the page, e.g. '1-month SORA', 'FHR6'."""
    return loan_type.strip()


def scrape(
    html: str, scraped_at: str, source_url: str
) -> tuple[list[dict], list[str]]:
    """Returns (records, skipped_bank_labels)."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []
    skipped: list[str] = []

    for table_idx, table in enumerate(soup.find_all("table")):
        rows = table.find_all("tr")
        if not rows:
            continue
        header = cell_texts(rows[0])
        if not is_rate_table(header):
            continue  # e.g. the LTV eligibility table

        # YEAR 1..YEAR N column positions
        year_cols = [
            (i, h) for i, h in enumerate(header) if re.fullmatch(r"YEAR\s*\d+", h.strip(), re.I)
        ]

        for row in rows[1:]:
            cells = cell_texts(row)
            if len(cells) < len(header) - 1:
                continue  # junk "Get rewards" filler rows
            raw_bank = cells[0].strip()
            loan_type = cells[1].strip() if len(cells) > 1 else ""
            if not raw_bank or not loan_type:
                continue

            bank = canonical_bank(raw_bank)
            if bank is None:
                skipped.append(raw_bank)
                continue  # broker promo / not a real bank

            parsed_years = []
            for col_idx, col_name in year_cols:
                raw = cells[col_idx] if col_idx < len(cells) else ""
                parsed_years.append((col_name, raw, parse_rate(raw)))

            kinds = [p["kind"] for _, _, p in parsed_years if p.get("kind")]
            rate_category = classify(loan_type, kinds)

            # One row per package; year rates go across as year_1..year_N.
            rec = {
                "scraped_at_utc": scraped_at,
                "source": "dollarback",
                "bank": bank,
                "rate_category": rate_category,
                "loan_type": loan_type,
                "lock_in_years": lock_in_years(loan_type),
                "benchmark": benchmark(loan_type) if rate_category == "floating" else "",
                "rate_as_of": "",
                "property_type": "private",
            }
            notes = []
            for col_name, raw, p in parsed_years:
                year = int(re.search(r"\d+", col_name).group())
                if not p:
                    rec[f"year_{year}"] = ""
                    continue
                # floating cells are spreads over the benchmark ("+0.35%")
                rec[f"year_{year}"] = f"+{p['value']}" if p["is_spread"] else p["value"]
                if p["note"]:
                    notes.append(f"Y{year}:{p['note']}")

            rec["note"] = "; ".join(notes)
            rec["source_url"] = source_url
            records.append(rec)

    return records, skipped


FIELDS = [
    "scraped_at_utc",
    "source",          # dollarback (bank package) | mas (benchmark)
    "bank",            # "MAS" for benchmark rows
    "rate_category",   # fixed | floating | benchmark
    "loan_type",       # as printed, e.g. "2 Year Fixed" / "1-month SORA"
    "lock_in_years",
    "benchmark",       # floating peg only
    "rate_as_of",      # MAS publication date (benchmark rows only)
    "year_1",
    "year_2",
    "year_3",
    "year_4",
    "year_5",
    "year_6",
    "note",
    "property_type",
    "source_url",
]



MAS_SOURCE_URL = (
    "https://eservices.mas.gov.sg/apimg-gw/server/"
    "monthly_statistical_bulletin_non610mssql/"
    "domestic_interest_rates_daily/views/domestic_interest_rates_daily"
)

# Human-readable landing pages for attribution in the UI. MAS_SOURCE_URL above
# is the raw API endpoint — correct provenance for the CSV, but useless to a
# person who clicks it, so the panel links to the catalogue page instead.
SOURCE_SITES = {
    "mas": {
        "label": "MAS",
        "detail": "Domestic Interest Rates (Daily)",
        "url": "https://eservices.mas.gov.sg/Statistics/dir/DomesticInterestRates.aspx",
    },
    "dollarback": {
        "label": "DollarBack Mortgage",
        "detail": "Private property loan rates",
        "url": URL,
    },
}

DEFAULT_OUT = Path(DATA_DIR) / "market_rates.csv"


def _sora_row(scraped_at: str, as_of: str, rate: float) -> dict:
    """One SORA observation in the shared schema."""
    return {
        "scraped_at_utc": scraped_at,
        "source": "mas",
        "bank": "MAS",
        "rate_category": "benchmark",
        "loan_type": "3M Compounded SORA",
        "lock_in_years": "",
        "benchmark": "3M Compounded SORA",
        "rate_as_of": as_of,
        # A benchmark has no year schedule; the level goes in year_1.
        "year_1": rate,
        "year_2": "",
        "year_3": "",
        "year_4": "",
        "year_5": "",
        "year_6": "",
        "note": "Add a package spread to get the all-in floating rate.",
        "property_type": "",
        "source_url": MAS_SOURCE_URL,
    }


def fetch_sora_rows(scraped_at: str, days: int = 90) -> list[dict]:
    """SORA benchmark rows: the full daily history in one MAS call.

    Raises ``SoraUnavailable`` if the live series cannot be fetched; callers
    must report that rather than substituting a rate.
    """
    return [
        _sora_row(scraped_at, obs["as_of"], obs["rate"])
        for obs in get_sora_history(days=days)
    ]


def existing_keys(path: Path) -> set[tuple[str, str]]:
    """(bank, rate_as_of) pairs already stored, so backfills stay idempotent.

    Only dated rows are de-duplicated. Scraped package rows carry no
    ``rate_as_of`` and are appended every run — that repetition is the point,
    since it is how a package's rate over time gets tracked.
    """
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {
            (r.get("bank", ""), r["rate_as_of"])
            for r in csv.DictReader(f)
            if r.get("rate_as_of")
        }


def write_csv(path: Path, records: list[dict], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not (append and exists):
            writer.writeheader()
        writer.writerows(records)


def collect(
    url: str = URL,
    out: Path | None = None,
    sora_days: int = 90,
    with_sora: bool = True,
) -> dict:
    """Scrape + backfill into the cumulative CSV. Returns a summary dict.

    Never raises for a source being down: each source is reported independently
    so one outage cannot discard the other's data.
    """
    out = Path(out) if out else DEFAULT_OUT
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary: dict = {
        "scraped_at": scraped_at,
        "out": str(out),
        "bank_rows": 0,
        "sora_rows": 0,
        "skipped": [],
        "errors": [],
    }

    try:
        records, skipped = scrape(fetch(url), scraped_at, url)
        summary["bank_rows"] = len(records)
        summary["skipped"] = sorted(set(skipped))
    except Exception as exc:  # noqa: BLE001 -- reported, never faked
        records = []
        summary["errors"].append(f"competitor scrape failed: {exc}")

    sora_rows: list[dict] = []
    if with_sora:
        try:
            seen = existing_keys(out)
            sora_rows = [
                r for r in fetch_sora_rows(scraped_at, days=sora_days)
                if ("MAS", r["rate_as_of"]) not in seen
            ]
            summary["sora_rows"] = len(sora_rows)
        except SoraUnavailable as exc:
            summary["errors"].append(str(exc))

    rows = records + sora_rows
    if rows:
        write_csv(out, rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="cumulative CSV")
    ap.add_argument(
        "--sora-days",
        type=int,
        default=90,
        help="days of SORA history to backfill (one MAS call covers all)",
    )
    ap.add_argument("--no-sora", action="store_true", help="skip the MAS benchmark")
    args = ap.parse_args(argv)

    s = collect(
        url=args.url,
        out=Path(args.out),
        sora_days=args.sora_days,
        with_sora=not args.no_sora,
    )

    total = s["bank_rows"] + s["sora_rows"]
    if total:
        print(f"[ok] appended {total} rows -> {s['out']}")
        print(f"     competitor packages: {s['bank_rows']} | SORA observations: {s['sora_rows']}")
    else:
        print("[warn] nothing written", file=sys.stderr)
    if s["skipped"]:
        print(f"     filtered out non-bank labels: {', '.join(s['skipped'])}")
    for e in s["errors"]:
        print(f"[warn] {e}", file=sys.stderr)
        print("       (no fallback value was invented)", file=sys.stderr)
    return 0 if total else 2


if __name__ == "__main__":
    raise SystemExit(main())
