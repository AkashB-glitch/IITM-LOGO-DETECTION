# =============================================================================
# module1_discovery.py — Discover Suspicious Websites Using Multiple Sources
# =============================================================================
# This module finds candidate websites that might be illegally using the
# IIT Madras logo.  It queries FOUR free data sources automatically:
#   1. crt.sh       — SSL certificate transparency logs (no API key needed)
#   2. DuckDuckGo   — Web search scraping (no API key needed, fully automatic)
#   3. Google CSE   — Custom Search API (optional, needs API key)
#   4. dnstwist     — generates typo-squatted domain variations
# All results are merged, de-duplicated, and cleaned.
# =============================================================================

import requests
import json
import time
import re
import subprocess
import os
import pandas as pd
from urllib.parse import urlparse

import config  # Our central config file


# ─────────────────────────────────────────────────────────────────────────────
# Helper — URL normalisation & whitelist check
# ─────────────────────────────────────────────────────────────────────────────

def is_whitelisted(url: str) -> bool:
    """Return True if the URL belongs to an official IITM domain."""
    try:
        host = urlparse(url).netloc.lower()
        # Strip port number if present
        host = host.split(":")[0]
        for domain in config.WHITELIST_DOMAINS:
            if host == domain or host.endswith("." + domain):
                return True
    except Exception:
        pass
    return False


def normalise_url(raw: str) -> str:
    """Ensure the string starts with https:// so it's a valid URL."""
    raw = raw.strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — crt.sh Certificate Transparency Search
# ─────────────────────────────────────────────────────────────────────────────

def query_crtsh(keyword: str) -> list[str]:
    """
    Query crt.sh for SSL certificates whose common name / SAN contains
    the given keyword.  Returns a list of domain names.
    crt.sh is completely free and does not need an API key.
    """
    print(f"  [crt.sh] Searching for keyword: '{keyword}' ...")
    url = f"https://crt.sh/?q=%25{keyword}%25&output=json"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        entries = resp.json()
    except requests.exceptions.RequestException as exc:
        print(f"    WARNING: crt.sh request failed: {exc}")
        return []
    except json.JSONDecodeError:
        print("    WARNING: crt.sh returned non-JSON response.")
        return []

    domains = set()
    for entry in entries:
        # Each entry may contain multiple names separated by newlines
        name_value = entry.get("name_value", "")
        for line in name_value.splitlines():
            line = line.strip().lstrip("*.")   # remove wildcard prefix
            if line and "." in line:           # basic sanity check
                domains.add(line.lower())

    print(f"    Found {len(domains)} unique domains from crt.sh.")
    return list(domains)


def run_crtsh_discovery() -> list[str]:
    """Run crt.sh queries for all configured keywords and merge results."""
    all_domains: list[str] = []
    for keyword in config.CRT_SH_QUERIES:
        domains = query_crtsh(keyword)
        all_domains.extend(domains)
        time.sleep(1)   # Be polite — don't hammer the free API

    # Convert domain names → full URLs
    urls = [normalise_url(d) for d in all_domains]
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Google Custom Search API
# ─────────────────────────────────────────────────────────────────────────────

def query_google_cse(query: str) -> list[str]:
    """
    Call Google's Custom Search JSON API.
    Free tier: 100 queries per day.
    Returns a list of result URLs.
    """
    print(f"  [Google CSE] Query: {query!r}")

    # Skip if the user hasn't provided real keys yet
    if "YOUR_" in config.GOOGLE_API_KEY or "YOUR_" in config.GOOGLE_CSE_ID:
        print("    SKIPPING — Google API key not configured (see config.py).")
        return []

    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": config.GOOGLE_API_KEY,
        "cx":  config.GOOGLE_CSE_ID,
        "q":   query,
        "num": config.MAX_GOOGLE_RESULTS_PER_QUERY,
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        print(f"    WARNING: Google CSE request failed: {exc}")
        return []

    items = data.get("items", [])
    urls = [item["link"] for item in items if "link" in item]
    print(f"    Found {len(urls)} results.")
    return urls


def run_google_discovery() -> list[str]:
    """Run all configured Google CSE queries and merge results."""
    all_urls: list[str] = []
    for q in config.GOOGLE_QUERIES:
        urls = query_google_cse(q)
        all_urls.extend(urls)
        time.sleep(1)
    return all_urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — dnstwist Typo-squatting
# ─────────────────────────────────────────────────────────────────────────────

def run_dnstwist(domain: str = "iitm.ac.in") -> list[str]:
    """
    Use the dnstwist command-line tool to generate typo-squatted domain
    variations of iitm.ac.in.  Requires `dnstwist` to be installed:
        pip install dnstwist
    Returns a list of domains that actually resolve (registered).
    """
    print(f"  [dnstwist] Generating typo-squats for: {domain}")

    try:
        result = subprocess.run(
            ["dnstwist", "--registered", "--format", "json", domain],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"    WARNING: dnstwist error: {result.stderr.strip()}")
            return []

        entries = json.loads(result.stdout)
    except FileNotFoundError:
        print("    WARNING: dnstwist not found. Install with: pip install dnstwist")
        return []
    except (json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"    WARNING: dnstwist failed: {exc}")
        return []

    domains = [e.get("domain", "") for e in entries if e.get("domain")]
    print(f"    Found {len(domains)} registered typo-squat domains.")
    return [normalise_url(d) for d in domains]


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — DuckDuckGo Web Search (FREE, no API key, fully automatic)
# ─────────────────────────────────────────────────────────────────────────────

# Queries to automatically search on DuckDuckGo
DDG_QUERIES = [
    '"IIT Madras" logo -site:iitm.ac.in',
    '"IITM" certificate -site:iitm.ac.in',
    '"IIT Madras" admission -site:iitm.ac.in',
    '"IIT Madras" online course -site:iitm.ac.in',
    'iitmadras logo site:.com OR site:.in OR site:.org',
    '"iit madras" placement portal -site:iitm.ac.in',
]


def search_duckduckgo(query: str, max_results: int = 20) -> list[str]:
    """
    Search DuckDuckGo using the duckduckgo-search library.
    No API key needed — works automatically and bypasses bot blocks.
    Returns a list of result page URLs.

    Install: pip install duckduckgo-search
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # fallback for older installs
        except ImportError:
            print("    WARNING: ddgs not installed. Run: pip install ddgs")
            return []

    urls: list[str] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                max_results=max_results,
                region="wt-wt",
                safesearch="off",
            )
            for r in results:
                link = r.get("href") or r.get("url", "")
                if link.startswith("http"):
                    urls.append(link)
    except Exception as exc:
        print(f"    WARNING: DuckDuckGo search failed: {exc}")

    return urls


def run_duckduckgo_discovery() -> list[str]:
    """
    Run all DuckDuckGo queries and collect URLs automatically.
    This is the primary free automatic discovery method.
    """
    print("  [DuckDuckGo] Automatically searching the web ...")
    all_urls: list[str] = []

    for query in DDG_QUERIES:
        print(f"    Query: {query!r}")
        urls = search_duckduckgo(query, max_results=20)
        print(f"    Found {len(urls)} results.")
        all_urls.extend(urls)
        time.sleep(2)   # polite delay between searches

    print(f"  [DuckDuckGo] Total: {len(all_urls)} URLs found.")
    return all_urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 5 — Hardcoded seed URLs (fallback / demo mode)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_SEED_URLS = [
    # These are example suspicious-looking URLs for demo/testing purposes.
    "https://iitm-courses.com",
    "https://iitmadras-online.com",
    "https://iit-madras-certificate.com",
    "https://iitmcertificate.in",
    "https://iitm-placement.com",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main discovery pipeline
# ─────────────────────────────────────────────────────────────────────────────

def discover_candidates() -> pd.DataFrame:
    """
    Orchestrate all discovery sources, merge results, remove whitelist
    entries, and return a clean DataFrame of candidate URLs.

    Returns
    -------
    pd.DataFrame with columns: url, source
    """
    print("\n" + "=" * 65)
    print("MODULE 1 — DISCOVERY: Finding Suspicious Websites")
    print("=" * 65)

    records: list[dict] = []

    # ── 1. crt.sh ─────────────────────────────────────────────────────────
    print("\n[1/5] Querying crt.sh certificate transparency logs ...")
    for url in run_crtsh_discovery():
        records.append({"url": url, "source": "crt.sh"})

    # ── 2. DuckDuckGo (FREE automatic web search, no API key) ─────────────
    print("\n[2/5] Searching DuckDuckGo automatically (no API key needed) ...")
    for url in run_duckduckgo_discovery():
        records.append({"url": url, "source": "duckduckgo"})

    # ── 3. Google CSE ──────────────────────────────────────────────────────
    print("\n[3/5] Querying Google Custom Search ...")
    for url in run_google_discovery():
        records.append({"url": url, "source": "google_cse"})

    # ── 4. dnstwist ───────────────────────────────────────────────────────
    print("\n[4/5] Running dnstwist typo-squat analysis ...")
    for url in run_dnstwist("iitm.ac.in"):
        records.append({"url": url, "source": "dnstwist"})

    # ── 5. Demo seeds ─────────────────────────────────────────────────────
    print("\n[5/5] Adding demo/seed URLs ...")
    for url in DEMO_SEED_URLS:
        records.append({"url": url, "source": "seed"})

    # ── Merge & clean ─────────────────────────────────────────────────────
    print("\n[→] Cleaning and de-duplicating results ...")
    df = pd.DataFrame(records)

    if df.empty:
        print("WARNING: No candidate URLs found. Check your API keys.")
        return df

    # Normalise all URLs
    df["url"] = df["url"].apply(normalise_url)

    # Remove whitelisted domains
    mask = df["url"].apply(lambda u: not is_whitelisted(u))
    before = len(df)
    df = df[mask].copy()
    print(f"  Removed {before - len(df)} whitelisted official IITM URLs.")

    # Remove duplicates (keep first source)
    df = df.drop_duplicates(subset="url", keep="first").reset_index(drop=True)

    # Cap to configured maximum
    df = df.head(config.MAX_CANDIDATE_URLS)

    # Save to CSV
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, "candidate_urls.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✓ Discovery complete. {len(df)} candidate URLs saved to: {out_path}")
    print(df["source"].value_counts().to_string())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = discover_candidates()
    print("\nSample of discovered URLs:")
    print(df.head(20).to_string(index=False))
