# =============================================================================
# module4_reverse_search.py — Reverse Image Search with TinEye + Google Lens
# =============================================================================
# This module performs reverse image search using the IITM logo image to
# find websites that are already known to host it.
#
# Methods supported:
#   1. TinEye API (free tier: 150 searches/month)
#      → https://services.tineye.com/TinEyeAPI
#   2. Google Vision / Lens (via scraping approach, no official API needed)
#   3. SauceNAO (free public API, good for general image search)
#
# These results are merged with Module 1 candidates for a complete picture.
# =============================================================================

import os
import time
import base64
import json
import requests
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path

import config   # Central config


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — TinEye API
# ─────────────────────────────────────────────────────────────────────────────

def search_tineye(logo_path: str) -> list[dict]:
    """
    Submit the IITM logo image to the TinEye reverse image search API.

    Parameters
    ----------
    logo_path : local file path to the IITM logo image (PNG or JPG)

    Returns
    -------
    list of dicts with keys: url, image_url, score
    """
    print("\n[TinEye] Submitting logo for reverse image search ...")

    # Skip if API key not configured
    if "YOUR_" in config.TINEYE_API_KEY:
        print("  SKIPPING — TinEye API key not configured (see config.py).")
        return []

    if not os.path.exists(logo_path):
        print(f"  ERROR: Logo file not found at {logo_path}")
        return []

    endpoint = "https://api.tineye.com/rest/search/"

    try:
        with open(logo_path, "rb") as f:
            files = {"image": (Path(logo_path).name, f, "image/png")}
            params = {"api_key": config.TINEYE_API_KEY, "limit": 100}
            resp = requests.post(endpoint, files=files, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
    except requests.exceptions.RequestException as exc:
        print(f"  ERROR: TinEye request failed: {exc}")
        return []

    results: list[dict] = []
    matches = data.get("results", {}).get("matches", [])

    for match in matches:
        for backlink in match.get("backlinks", []):
            url = backlink.get("url", "")
            if url:
                results.append({
                    "url":       url,
                    "image_url": backlink.get("backlink", ""),
                    "score":     match.get("score", 0),
                    "source":    "tineye",
                })

    print(f"  Found {len(results)} results from TinEye.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Google Custom Search (Image Search mode)
# ─────────────────────────────────────────────────────────────────────────────

def search_google_image(logo_url: str = None) -> list[dict]:
    """
    Use Google Custom Search API in image-search mode to find pages
    showing images similar to the IITM logo.

    If you have publicly hosted the logo, pass its URL.
    Otherwise this falls back to text-based Google CSE (Module 1).

    Parameters
    ----------
    logo_url : publicly accessible URL to the logo image (optional)

    Returns
    -------
    list of dicts with keys: url, source
    """
    print("\n[Google Image Search] Searching for logo matches ...")

    if "YOUR_" in config.GOOGLE_API_KEY:
        print("  SKIPPING — Google API key not configured (see config.py).")
        return []

    results: list[dict] = []

    # Google CSE supports image-focused queries using "IITM logo filetype:png"
    image_queries = [
        "IITM logo filetype:png -site:iitm.ac.in",
        "IIT Madras logo filetype:jpg -site:iitm.ac.in",
        "iitm.ac.in logo -site:iitm.ac.in",
    ]

    endpoint = "https://www.googleapis.com/customsearch/v1"

    for query in image_queries:
        params = {
            "key":        config.GOOGLE_API_KEY,
            "cx":         config.GOOGLE_CSE_ID,
            "q":          query,
            "searchType": "image",
            "num":        10,
        }
        try:
            resp = requests.get(endpoint, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"  WARNING: Google Image Search failed: {exc}")
            continue

        for item in data.get("items", []):
            page_url = item.get("image", {}).get("contextLink", "")
            img_url  = item.get("link", "")
            if page_url:
                results.append({
                    "url":       page_url,
                    "image_url": img_url,
                    "score":     1.0,
                    "source":    "google_image",
                })

        time.sleep(0.5)

    print(f"  Found {len(results)} results from Google Image Search.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — SauceNAO (free, no key required for basic use)
# ─────────────────────────────────────────────────────────────────────────────

def search_saucenao(logo_path: str) -> list[dict]:
    """
    Submit logo to SauceNAO's reverse image search.
    SauceNAO works best for illustrations/logos.
    Free tier: 6 searches/30 seconds, 200/day.

    Parameters
    ----------
    logo_path : local path to the IITM logo image

    Returns
    -------
    list of dicts with keys: url, score, source
    """
    print("\n[SauceNAO] Submitting logo for reverse image search ...")

    if not os.path.exists(logo_path):
        print(f"  ERROR: Logo not found at {logo_path}")
        return []

    endpoint = "https://saucenao.com/search.php"

    try:
        with open(logo_path, "rb") as f:
            files  = {"file": (Path(logo_path).name, f, "image/png")}
            params = {"output_type": 2, "numres": 20}  # output_type=2 → JSON
            resp = requests.post(endpoint, files=files, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
    except requests.exceptions.RequestException as exc:
        print(f"  WARNING: SauceNAO request failed: {exc}")
        return []
    except json.JSONDecodeError:
        print("  WARNING: SauceNAO returned non-JSON response.")
        return []

    results: list[dict] = []
    for result in data.get("results", []):
        header = result.get("header", {})
        data_  = result.get("data",   {})

        similarity = float(header.get("similarity", 0)) / 100.0   # normalise to 0–1
        ext_urls   = data_.get("ext_urls", [])

        for url in ext_urls:
            results.append({
                "url":    url,
                "score":  similarity,
                "source": "saucenao",
            })

    print(f"  Found {len(results)} results from SauceNAO.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Combine all reverse image search results
# ─────────────────────────────────────────────────────────────────────────────

def run_reverse_search(
    logo_path: str,
    existing_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Run all reverse image search methods and merge with existing candidates.

    Parameters
    ----------
    logo_path    : path to the official IITM logo file
    existing_df  : DataFrame of URLs already found in Module 1 (can be None)

    Returns
    -------
    Merged DataFrame of all candidate URLs including new reverse-search findings
    """
    print("\n" + "=" * 65)
    print("MODULE 4 — REVERSE IMAGE SEARCH")
    print("=" * 65)

    all_results: list[dict] = []

    # Run all three search sources
    all_results.extend(search_tineye(logo_path))
    all_results.extend(search_google_image())
    all_results.extend(search_saucenao(logo_path))

    if not all_results:
        print("\nNo new URLs found via reverse image search.")
        return existing_df if existing_df is not None else pd.DataFrame()

    # Build DataFrame from reverse search results
    rev_df = pd.DataFrame(all_results)
    rev_df = rev_df.rename(columns={"url": "url"})

    # Keep only relevant columns for consistency with Module 1 output
    rev_df = rev_df[["url", "source"]].copy()

    # Whitelist filter (reuse from module1)
    from module1_discovery import is_whitelisted, normalise_url
    rev_df["url"] = rev_df["url"].apply(normalise_url)
    rev_df = rev_df[~rev_df["url"].apply(is_whitelisted)].copy()

    # Merge with existing candidates
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, rev_df], ignore_index=True)
    else:
        combined = rev_df

    # Deduplicate
    combined = combined.drop_duplicates(subset="url", keep="first")
    combined = combined.reset_index(drop=True)

    # Save updated candidate list
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, "candidate_urls_with_reverse.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n✓ Reverse search complete.")
    print(f"  New URLs found: {len(rev_df)}")
    print(f"  Total candidates after merge: {len(combined)}")
    print(f"  Saved to: {out_path}")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Find the logo file to use for reverse search
    logo_file = None
    positive_dir = os.path.join(config.LOGO_SAMPLES_DIR, "positive")
    if os.path.isdir(positive_dir):
        for f in os.listdir(positive_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                logo_file = os.path.join(positive_dir, f)
                break

    if not logo_file:
        print("ERROR: No logo file found in logo_samples/positive/")
        print("Please add at least one IITM logo image there first.")
        exit(1)

    print(f"Using logo file: {logo_file}")

    # Load existing candidate URLs from Module 1 (if available)
    csv_path = os.path.join(config.OUTPUT_DIR, "candidate_urls.csv")
    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    merged_df = run_reverse_search(logo_file, existing)
    print(f"\nFinal candidate list: {len(merged_df)} URLs")
