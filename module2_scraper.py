# =============================================================================
# module2_scraper.py — Visit Each Website Like a Human with Playwright
# =============================================================================
# For every URL discovered in Module 1, this module:
#   1. Opens the page in a real Chromium browser (Playwright)
#   2. Scrolls down to trigger lazy-loaded images
#   3. Takes a full-page screenshot
#   4. Extracts all <img> src URLs and downloads the images
#   5. Extracts all visible text
#   6. Saves everything under output/screenshots/ and output/images/
# =============================================================================

import os
import re
import time
import hashlib
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Playwright async API (faster for scraping many URLs)
from playwright.sync_api import sync_playwright, Page, Browser

import config   # Central config


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_filename(url: str, suffix: str = "") -> str:
    """
    Convert a URL into a safe filename by hashing it.
    E.g. 'https://bad-site.com/page' → 'a3f9b2c1' + suffix
    """
    digest = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{digest}{suffix}"


def make_dirs() -> None:
    """Create all required output directories."""
    for d in [config.OUTPUT_DIR, config.SCREENSHOTS_DIR, config.IMAGES_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core scraping function for one URL
# ─────────────────────────────────────────────────────────────────────────────

def scrape_url(page: Page, url: str) -> dict:
    """
    Visit a single URL with Playwright and collect:
      - screenshot path
      - list of downloaded image file paths
      - extracted text content

    Parameters
    ----------
    page : playwright Page object (already opened)
    url  : the URL to visit

    Returns
    -------
    dict with keys: url, screenshot, images, text, error
    """
    result = {
        "url":        url,
        "screenshot": None,
        "images":     [],
        "text":       "",
        "error":      None,
    }

    try:
        # ── Navigate to the page ──────────────────────────────────────────
        print(f"  Visiting: {url}")
        page.goto(url, timeout=config.PAGE_TIMEOUT, wait_until="networkidle")

        # ── Scroll down to load lazy images ───────────────────────────────
        # Simulate a human scrolling through the page
        total_height = page.evaluate("document.body.scrollHeight")
        scrolled = 0
        step = 600  # pixels per scroll step
        while scrolled < total_height:
            page.evaluate(f"window.scrollTo(0, {scrolled})")
            time.sleep(config.SCROLL_PAUSE)
            scrolled += step
            # Re-check height in case content loaded dynamically
            total_height = page.evaluate("document.body.scrollHeight")

        # Scroll back to top so screenshot starts from the beginning
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)

        # ── Take full-page screenshot ─────────────────────────────────────
        shot_name = safe_filename(url, ".png")
        shot_path = os.path.join(config.SCREENSHOTS_DIR, shot_name)
        page.screenshot(path=shot_path, full_page=True)
        result["screenshot"] = shot_path
        print(f"    ✓ Screenshot saved: {shot_path}")

        # ── Extract all image src URLs ─────────────────────────────────────
        img_srcs = page.evaluate("""
            () => Array.from(document.images)
                       .map(img => img.src)
                       .filter(src => src.startsWith('http'))
        """)

        # Also look for CSS background images
        bg_images = page.evaluate("""
            () => {
                const urls = [];
                for (const el of document.querySelectorAll('*')) {
                    const style = window.getComputedStyle(el);
                    const bg = style.backgroundImage;
                    if (bg && bg !== 'none') {
                        const match = bg.match(/url\\(["']?([^"')]+)["']?\\)/);
                        if (match) urls.push(match[1]);
                    }
                }
                return urls.filter(u => u.startsWith('http'));
            }
        """)

        all_img_urls = list(set(img_srcs + bg_images))

        # ── Download each image ────────────────────────────────────────────
        saved_imgs: list[str] = []
        for img_url in all_img_urls[:50]:   # cap at 50 images per page
            img_path = download_image(img_url, url)
            if img_path:
                saved_imgs.append(img_path)

        result["images"] = saved_imgs
        print(f"    ✓ Downloaded {len(saved_imgs)} images.")

        # ── Extract visible text ──────────────────────────────────────────
        text = page.evaluate("""
            () => {
                // Remove scripts and style tags first
                for (const el of document.querySelectorAll('script, style')) {
                    el.remove();
                }
                return document.body ? document.body.innerText : '';
            }
        """)
        result["text"] = (text or "").strip()

    except Exception as exc:
        result["error"] = str(exc)
        print(f"    ✗ Error scraping {url}: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Image downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_image(img_url: str, page_url: str) -> str | None:
    """
    Download an image from img_url and save it locally.
    Returns the local file path on success, None on failure.
    """
    try:
        # Build a unique filename from the image URL
        ext = Path(urlparse(img_url).path).suffix or ".jpg"
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"):
            ext = ".jpg"

        fname = safe_filename(img_url, ext)
        save_path = os.path.join(config.IMAGES_DIR, fname)

        # Skip if already downloaded (useful when re-running)
        if os.path.exists(save_path):
            return save_path

        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        resp = requests.get(img_url, headers=headers, timeout=10, stream=True)
        resp.raise_for_status()

        # Only save if it looks like an image (check Content-Type)
        ct = resp.headers.get("Content-Type", "")
        if not any(t in ct for t in ("image", "octet-stream")):
            return None

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return save_path

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch scraping — process all candidate URLs
# ─────────────────────────────────────────────────────────────────────────────

def scrape_all(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scrape every URL in the candidate DataFrame.

    Parameters
    ----------
    candidate_df : DataFrame with at least a 'url' column (output of Module 1)

    Returns
    -------
    DataFrame with scraping results added as new columns
    """
    print("\n" + "=" * 65)
    print("MODULE 2 — SCRAPING: Visiting Each Website")
    print("=" * 65)

    make_dirs()

    if candidate_df.empty:
        print("WARNING: No URLs to scrape. Run Module 1 first.")
        return candidate_df

    results: list[dict] = []
    total = len(candidate_df)

    # ── Launch browser once and reuse it across all pages ─────────────────
    with sync_playwright() as pw:
        browser: Browser = pw.chromium.launch(
            headless=config.HEADLESS,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        # Use a single browser context with a realistic user-agent
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            # Accept all common content types
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

        for idx, row in candidate_df.iterrows():
            url = row["url"]
            print(f"\n[{idx + 1}/{total}] Processing: {url}")

            # Create a fresh page for each URL
            page = context.new_page()
            try:
                result = scrape_url(page, url)
            finally:
                page.close()   # Always close the page to free memory

            result["source"] = row.get("source", "unknown")
            results.append(result)

            # Small delay between requests to behave like a human
            time.sleep(1.0)

        browser.close()

    # ── Save results ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    # Merge with original candidate info (e.g. source column)
    out_path = os.path.join(config.OUTPUT_DIR, "scraped_results.csv")
    # Save a flattened version (images list → count)
    save_df = results_df.copy()
    save_df["image_count"] = save_df["images"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    save_df["text_length"] = save_df["text"].apply(len)
    save_df.drop(columns=["images", "text"], inplace=True)  # too large for CSV
    save_df.to_csv(out_path, index=False)

    print(f"\n✓ Scraping complete. Results saved to: {out_path}")
    successful = results_df["screenshot"].notna().sum()
    print(f"  Successfully scraped: {successful}/{total} URLs")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (can run standalone for testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load URLs from Module 1 output, or use a quick test set
    csv_path = os.path.join(config.OUTPUT_DIR, "candidate_urls.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Quick demo with a couple of URLs
        df = pd.DataFrame({
            "url":    ["https://example.com", "https://httpbin.org"],
            "source": ["seed", "seed"],
        })

    results_df = scrape_all(df)
    print("\nSample results:")
    print(results_df[["url", "screenshot", "error"]].head(10).to_string(index=False))
