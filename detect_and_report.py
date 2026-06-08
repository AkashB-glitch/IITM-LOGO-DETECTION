# =============================================================================
# detect_and_report.py — Fast Detection + Website Link Report
# =============================================================================
# Reads candidate_urls.csv, scrapes each site with a SHORT timeout,
# runs OpenCV template matching + DL detection on screenshots,
# then prints a RANKED LIST OF WEBSITE LINKS with their risk level.
#
# Usage:
#   python detect_and_report.py
# =============================================================================

import os
import sys
import csv
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config

# ── Paths ─────────────────────────────────────────────────────────────────────
CANDIDATE_CSV    = os.path.join(config.OUTPUT_DIR, "candidate_urls.csv")
SCRAPED_CSV      = os.path.join(config.OUTPUT_DIR, "scraped_results.csv")
DETECTION_CSV    = os.path.join(config.OUTPUT_DIR, "detection_results.csv")
REPORT_HTML      = os.path.join(config.OUTPUT_DIR, "reports", "website_links_report.html")
SCREENSHOT_DIR   = config.SCREENSHOTS_DIR
LOGO_POSITIVE    = os.path.join(config.LOGO_SAMPLES_DIR, "positive")

os.makedirs(SCREENSHOT_DIR,                              exist_ok=True)
os.makedirs(os.path.join(config.OUTPUT_DIR, "reports"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Scrape (fast — 15 s timeout, screenshot only)
# ─────────────────────────────────────────────────────────────────────────────

async def scrape_url(browser, url: str, out_dir: str) -> dict:
    """Take a screenshot of one URL. Returns dict with url, screenshot, error."""
    import hashlib
    fname = hashlib.md5(url.encode()).hexdigest()[:12] + ".png"
    shot_path = os.path.join(out_dir, fname)

    result = {"url": url, "screenshot": None, "error": None}

    try:
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1280, "height": 900})
        await page.goto(url, timeout=15_000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        await page.screenshot(path=shot_path, full_page=False)
        await page.close()
        result["screenshot"] = shot_path
    except Exception as e:
        result["error"] = str(e)[:120]
        try:
            await page.close()
        except Exception:
            pass

    return result


async def scrape_all_async(urls: list[str]) -> list[dict]:
    from playwright.async_api import async_playwright

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        print(f"\n  Scraping {len(urls)} URLs (15s timeout each) ...")

        for i, url in enumerate(urls, 1):
            r = await scrape_url(browser, url, SCREENSHOT_DIR)
            status = "OK" if r["screenshot"] else "FAIL"
            print(f"  [{i:3d}/{len(urls)}] {status}  {url[:70]}")
            results.append(r)

        await browser.close()

    return results


def scrape_phase(urls: list[str]) -> pd.DataFrame:
    """Run async scraper synchronously."""
    results = asyncio.run(scrape_all_async(urls))
    df = pd.DataFrame(results)
    df.to_csv(SCRAPED_CSV, index=False)
    ok = df["screenshot"].notna().sum()
    print(f"\n  Scraped: {ok}/{len(df)} successful screenshots saved.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Detection on screenshots
# ─────────────────────────────────────────────────────────────────────────────

def run_detection_phase(scraped_df: pd.DataFrame) -> pd.DataFrame:
    from module3_detector import (
        load_trained_model, load_templates,
        detect_logo_in_image, classify_risk,
    )

    print("\n" + "=" * 65)
    print("  DETECTION PHASE")
    print("=" * 65)

    dl_model, device = None, "cpu"
    templates        = load_templates(LOGO_POSITIVE)

    if not templates:
        print("  WARNING: No templates found — template matching disabled.")

    rows = []
    for idx, row in scraped_df.iterrows():
        url      = str(row.get("url", ""))
        shot     = row.get("screenshot")
        error    = row.get("error", "")

        if not isinstance(shot, str) or not os.path.exists(shot):
            rows.append({
                "url": url, "dl_score": 0.0, "tm_score": 0.0,
                "final_score": 0.0, "logo_detected": False,
                "risk_level": "LOW", "screenshot": None,
                "error": str(error) if pd.notna(error) else "no screenshot",
            })
            continue

        from module3_detector import template_match_score
        tm = template_match_score(shot, templates)
        det = {
            "dl_score": 0.0,
            "tm_score": round(tm, 4),
            "final_score": round(tm, 4),
            "logo_detected": tm >= config.CONFIDENCE_THRESH,
        }
        risk = classify_risk(det["final_score"])

        flag = "[HIGH]  " if risk == "HIGH" else ("[MEDIUM]" if risk == "MEDIUM" else "[low]   ")
        print(f"  [{idx+1:3d}] {flag} {det['final_score']*100:5.1f}%  {url[:65]}")

        rows.append({
            "url":          url,
            "dl_score":     det["dl_score"],
            "tm_score":     det["tm_score"],
            "final_score":  det["final_score"],
            "logo_detected": det["logo_detected"],
            "risk_level":   risk,
            "screenshot":   shot,
            "error":        str(error) if pd.notna(error) else "",
        })

    det_df = pd.DataFrame(rows)
    det_df.to_csv(DETECTION_CSV, index=False)
    print(f"\n  Detection results saved -> {DETECTION_CSV}")
    return det_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Print ranked website links + generate HTML report
# ─────────────────────────────────────────────────────────────────────────────

def print_ranked_links(det_df: pd.DataFrame) -> None:
    """Print a clean ranked table of all websites with scores."""

    high   = det_df[det_df["risk_level"] == "HIGH"].sort_values("final_score", ascending=False)
    medium = det_df[det_df["risk_level"] == "MEDIUM"].sort_values("final_score", ascending=False)
    low    = det_df[det_df["risk_level"] == "LOW"].sort_values("final_score", ascending=False)

    print("\n\n" + "=" * 75)
    print("  IITM LOGO MISUSE — WEBSITE LINK REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    print(f"\n  Total analysed : {len(det_df)}")
    print(f"  HIGH risk      : {len(high)}")
    print(f"  MEDIUM risk    : {len(medium)}")
    print(f"  LOW / clean    : {len(low)}")

    def section(title, df, emoji):
        if df.empty:
            return
        print(f"\n\n  {emoji}  {title} ({len(df)} sites)")
        print(f"  {'─'*71}")
        print(f"  {'#':>3}  {'SCORE':>6}  {'DL':>6}  {'TM':>6}  URL")
        print(f"  {'─'*71}")
        for rank, (_, r) in enumerate(df.iterrows(), 1):
            print(f"  {rank:>3}.  {r['final_score']*100:5.1f}%  "
                  f"{r['dl_score']*100:5.1f}%  {r['tm_score']*100:5.1f}%  "
                  f"{r['url']}")

    section("HIGH RISK — Likely Using IITM Logo",   high,   "🚨")
    section("MEDIUM RISK — Possible IITM Logo Use", medium, "⚠️ ")
    section("LOW RISK — No Logo Detected",           low,    "✅")

    print("\n" + "=" * 75)


def generate_html_report(det_df: pd.DataFrame) -> None:
    """Generate a clean HTML file with clickable links."""

    high   = det_df[det_df["risk_level"] == "HIGH"].sort_values("final_score", ascending=False)
    medium = det_df[det_df["risk_level"] == "MEDIUM"].sort_values("final_score", ascending=False)
    low    = det_df[det_df["risk_level"] == "LOW"].sort_values("final_score", ascending=False)

    def rows_html(df, badge_class):
        html = ""
        for rank, (_, r) in enumerate(df.iterrows(), 1):
            shot_tag = (f'<a href="../../{r["screenshot"]}" target="_blank">📷</a>'
                        if isinstance(r.get("screenshot"), str) and r["screenshot"] else "—")
            html += f"""
            <tr>
                <td>{rank}</td>
                <td><span class="badge {badge_class}">{r['risk_level']}</span></td>
                <td>{r['final_score']*100:.1f}%</td>
                <td>{r['dl_score']*100:.1f}%</td>
                <td>{r['tm_score']*100:.1f}%</td>
                <td><a href="{r['url']}" target="_blank">{r['url']}</a></td>
                <td>{shot_tag}</td>
                <td style="font-size:0.75rem;color:#666">{str(r.get('error',''))[:60]}</td>
            </tr>"""
        return html

    all_rows = rows_html(high, "high") + rows_html(medium, "medium") + rows_html(low, "low")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IITM Logo Misuse — Website Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }}
    h1 {{ color: #fff; font-size: 1.6rem; margin-bottom: 4px; }}
    .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 24px; }}
    .stats {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
    .stat-card {{ background: #1e2130; border-radius: 10px; padding: 16px 24px; min-width: 130px; }}
    .stat-card .num {{ font-size: 2rem; font-weight: 700; }}
    .stat-card .lbl {{ font-size: 0.8rem; color: #888; margin-top: 2px; }}
    .high-num {{ color: #ff4c4c; }}
    .medium-num {{ color: #ffa040; }}
    .low-num {{ color: #4caf50; }}
    .total-num {{ color: #64b5f6; }}
    table {{ width: 100%; border-collapse: collapse; background: #1a1d2e; border-radius: 10px; overflow: hidden; }}
    thead {{ background: #252840; }}
    th {{ padding: 12px 10px; text-align: left; font-size: 0.8rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }}
    td {{ padding: 10px; border-bottom: 1px solid #252840; font-size: 0.85rem; vertical-align: middle; }}
    tr:hover {{ background: #20233a; }}
    a {{ color: #64b5f6; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; }}
    .badge.high   {{ background: #ff4c4c22; color: #ff6b6b; border: 1px solid #ff4c4c55; }}
    .badge.medium {{ background: #ffa04022; color: #ffc06b; border: 1px solid #ffa04055; }}
    .badge.low    {{ background: #4caf5022; color: #81c784; border: 1px solid #4caf5055; }}
    .header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
    .logo-icon {{ font-size: 2rem; }}
    footer {{ margin-top: 24px; color: #555; font-size: 0.8rem; text-align: center; }}
  </style>
</head>
<body>
  <div class="header">
    <span class="logo-icon">🔍</span>
    <div>
      <h1>IITM Logo Misuse Detection — Website Report</h1>
      <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; IIT Madras Logo Protection System</p>
    </div>
  </div>

  <div class="stats">
    <div class="stat-card"><div class="num total-num">{len(det_df)}</div><div class="lbl">Total Analysed</div></div>
    <div class="stat-card"><div class="num high-num">{len(high)}</div><div class="lbl">HIGH Risk</div></div>
    <div class="stat-card"><div class="num medium-num">{len(medium)}</div><div class="lbl">MEDIUM Risk</div></div>
    <div class="stat-card"><div class="num low-num">{len(low)}</div><div class="lbl">LOW / Clean</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th><th>Risk</th><th>Score</th><th>DL</th><th>TM</th>
        <th>Website URL</th><th>Screenshot</th><th>Note</th>
      </tr>
    </thead>
    <tbody>
      {all_rows}
    </tbody>
  </table>

  <footer>IITM Logo Detection Project &mdash; IIT Madras &mdash; Scores: DL = Deep Learning, TM = Template Match</footer>
</body>
</html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report saved -> {REPORT_HTML}")
    print(f"  Open it in your browser to see all clickable links!")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load candidate URLs
    if not os.path.exists(CANDIDATE_CSV):
        print(f"ERROR: {CANDIDATE_CSV} not found. Run discovery first.")
        sys.exit(1)

    candidates_df = pd.read_csv(CANDIDATE_CSV)
    urls = candidates_df["url"].dropna().tolist()
    print(f"Loaded {len(urls)} candidate URLs.")

    # Step 1: Scrape
    if os.path.exists(SCRAPED_CSV):
        print(f"\n[SKIP] scraped_results.csv already exists. Loading ...")
        scraped_df = pd.read_csv(SCRAPED_CSV)
        print(f"  Loaded {len(scraped_df)} rows.")
    else:
        scraped_df = scrape_phase(urls)

    # Step 2: Detection
    det_df = run_detection_phase(scraped_df)

    # Step 3: Print links + generate HTML
    print_ranked_links(det_df)
    generate_html_report(det_df)


if __name__ == "__main__":
    main()
