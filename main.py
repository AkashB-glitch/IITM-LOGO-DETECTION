# =============================================================================
# main.py — Master Pipeline Orchestrator
# =============================================================================
# This is the single entry point to run the entire detection pipeline:
#
#   Step 1 — Module 1: Discover suspicious websites
#   Step 2 — Module 4: Enrich with reverse image search results
#   Step 3 — Module 2: Scrape each website (screenshot + images)
#   Step 4 — Module 3: Detect IITM logo with deep learning + OpenCV
#   Step 5 — Save final report (launch dashboard separately)
#
# Usage:
#   python main.py                  # Run the full pipeline
#   python main.py --skip-scrape    # Re-run detection on existing screenshots
#   python main.py --demo           # Run with 5 demo URLs (no API keys needed)
#
# To see the dashboard:
#   streamlit run module5_report.py
# =============================================================================

import os
import sys
import argparse
import pandas as pd
import datetime

import config
from module1_discovery import discover_candidates
from module2_scraper    import scrape_all
from module3_detector   import run_detection, train_model
from module4_reverse_search import run_reverse_search


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="IITM Logo Misuse Detection Pipeline",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip Module 1 — load existing candidate_urls.csv instead",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip Module 2 — load existing scraped_results.csv instead",
    )
    parser.add_argument(
        "--skip-reverse",
        action="store_true",
        help="Skip Module 4 reverse image search",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the deep learning model before running detection",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: use 5 hardcoded URLs, skip API calls",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Limit the number of URLs to process (useful for testing)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode
# ─────────────────────────────────────────────────────────────────────────────

DEMO_URLS = [
    "https://example.com",
    "https://httpbin.org",
    "https://python.org",
    "https://opencv.org",
    "https://pytorch.org",
]


def make_demo_candidates() -> pd.DataFrame:
    """Return a small DataFrame of demo URLs for testing without any API keys."""
    return pd.DataFrame({
        "url":    DEMO_URLS,
        "source": ["demo"] * len(DEMO_URLS),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step_discovery(args) -> pd.DataFrame:
    """Step 1 & 4: Discover candidate URLs (+ reverse image search enrichment)."""

    if args.demo:
        print("\n[DEMO MODE] Using hardcoded demo URLs.")
        return make_demo_candidates()

    # ── Module 1 ───────────────────────────────────────────────────────────
    candidate_path = os.path.join(config.OUTPUT_DIR, "candidate_urls.csv")

    if args.skip_discovery and os.path.exists(candidate_path):
        print(f"\n[→] Loading existing candidates from: {candidate_path}")
        candidates = pd.read_csv(candidate_path)
        print(f"    Loaded {len(candidates)} URLs.")
    else:
        candidates = discover_candidates()

    if candidates.empty:
        print("\nERROR: No candidate URLs found. Exiting.")
        sys.exit(1)

    # ── Module 4 — Reverse image search (optional enrichment) ──────────────
    if not args.skip_reverse:
        # Find logo file for reverse search
        logo_file = find_logo_file()
        if logo_file:
            candidates = run_reverse_search(logo_file, candidates)
        else:
            print("\n[Module 4] Skipping reverse search: no logo file found in logo_samples/positive/")

    # Apply URL cap
    if args.max_urls and len(candidates) > args.max_urls:
        print(f"\n[→] Capping to {args.max_urls} URLs (--max-urls flag).")
        candidates = candidates.head(args.max_urls)

    return candidates


def step_scrape(args, candidates: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Scrape each URL with Playwright."""

    scrape_path = os.path.join(config.OUTPUT_DIR, "scraped_results.csv")

    if args.skip_scrape and os.path.exists(scrape_path):
        print(f"\n[→] Loading existing scraped results from: {scrape_path}")
        scraped = pd.read_csv(scrape_path)
        # scraped_results.csv doesn't have the 'images' list — rebuild minimal structure
        if "images" not in scraped.columns:
            scraped["images"] = [[] for _ in range(len(scraped))]
        print(f"    Loaded {len(scraped)} scraped records.")
        return scraped
    else:
        return scrape_all(candidates)


def step_train(args) -> None:
    """Optional: Train the deep learning model before detection."""
    if args.train:
        print("\n[→] Training deep learning model ...")
        try:
            train_model()
        except RuntimeError as exc:
            print(f"  WARNING: Training failed: {exc}")
            print("  Detection will use OpenCV template matching only.")


def step_detect(scraped_df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Run logo detection on all scraped content."""
    return run_detection(scraped_df)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def find_logo_file() -> str | None:
    """Look for a logo image in logo_samples/positive/."""
    positive_dir = os.path.join(config.LOGO_SAMPLES_DIR, "positive")
    if os.path.isdir(positive_dir):
        for fname in sorted(os.listdir(positive_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(positive_dir, fname)
    return None


def print_final_summary(results_df: pd.DataFrame) -> None:
    """Print a nice summary table at the end of the pipeline."""
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE — FINAL SUMMARY")
    print("=" * 65)

    total   = len(results_df)
    flagged = int(results_df["logo_detected"].sum()) if "logo_detected" in results_df.columns else 0
    high    = int((results_df["risk_level"] == "HIGH").sum())   if "risk_level" in results_df.columns else 0
    medium  = int((results_df["risk_level"] == "MEDIUM").sum()) if "risk_level" in results_df.columns else 0

    print(f"  Total websites analysed : {total}")
    print(f"  Logo detected           : {flagged}")
    print(f"  High risk               : {high}")
    print(f"  Medium risk             : {medium}")
    print(f"  Low / no detection      : {total - high - medium}")

    if flagged > 0:
        print("\n  ⚠️  High-confidence detections:")
        top = results_df[results_df["logo_detected"] == True].sort_values(
            "final_score", ascending=False
        ).head(10)
        for _, r in top.iterrows():
            print(f"    {r['risk_level']:6} | {r['final_score']*100:5.1f}% | {r['url']}")

    out_path = os.path.join(config.OUTPUT_DIR, "detection_results.csv")
    print(f"\n  Full results saved to : {out_path}")
    print("\n  To launch the dashboard run:")
    print("    streamlit run module5_report.py")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  IITM LOGO MISUSE DETECTION SYSTEM")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Ensure output directories exist
    for d in [config.OUTPUT_DIR, config.SCREENSHOTS_DIR,
              config.IMAGES_DIR, config.MODEL_DIR,
              os.path.join(config.LOGO_SAMPLES_DIR, "positive"),
              os.path.join(config.LOGO_SAMPLES_DIR, "negative")]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1 + 4: Discovery ──────────────────────────────────────────────
    candidates = step_discovery(args)

    # ── Optional training ─────────────────────────────────────────────────
    step_train(args)

    # ── Step 2: Scraping ──────────────────────────────────────────────────
    scraped_df = step_scrape(args, candidates)

    # ── Step 3: Detection ─────────────────────────────────────────────────
    results_df = step_detect(scraped_df)

    # ── Final summary ─────────────────────────────────────────────────────
    print_final_summary(results_df)


if __name__ == "__main__":
    main()
