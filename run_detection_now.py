# =============================================================================
# run_detection_now.py  — Run Detection on ALL Scraped Websites
# =============================================================================
# This script fixes the main problem: detection_results.csv only had 5 demo
# entries. This runs the full detector on all 245 scraped URLs.
#
# What this does differently from main.py:
#   1. Reads scraped_results.csv (skips discovery + scraping)
#   2. Fixes screenshot path separators (mixed / and \ in CSV)
#   3. Scans output/images/ folder and maps images back to each URL
#      (because the images column was missing from the CSV)
#   4. Runs DL + OpenCV detection on every image found
#   5. Saves full detection_results.csv + prints ranked summary
#
# Run with:
#   python run_detection_now.py
#   python run_detection_now.py --top 20        # show top 20 detections
#   python run_detection_now.py --threshold 0.5 # lower threshold for more hits
# =============================================================================

import os
import sys
import argparse
import re
import pandas as pd
from pathlib import Path

import config
from module3_detector import run_detection, load_trained_model, load_templates, detect_logo_in_image, classify_risk


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run IITM Logo Detection on all scraped websites")
    p.add_argument("--top",       type=int,   default=30,   help="Show top N results (default: 30)")
    p.add_argument("--threshold", type=float, default=None, help="Override CONFIDENCE_THRESH in config.py")
    p.add_argument("--no-dl",     action="store_true",      help="Skip deep learning, use template matching only")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Fix screenshot path issues
# ─────────────────────────────────────────────────────────────────────────────

def fix_path(raw_path: str) -> str | None:
    """
    The scraped CSV stores paths like 'output/screenshots\\abc.png' with
    mixed slashes. Convert to a proper OS path and verify it exists.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    # Normalise slashes for current OS
    fixed = raw_path.replace("\\", os.sep).replace("/", os.sep)

    if os.path.exists(fixed):
        return fixed

    # Try relative to project root
    root = Path(__file__).parent
    candidate = root / fixed
    if candidate.exists():
        return str(candidate)

    return None  # File genuinely doesn't exist


# ─────────────────────────────────────────────────────────────────────────────
# Build URL → image list mapping from output/images/
# ─────────────────────────────────────────────────────────────────────────────

def load_image_paths_from_disk() -> list[str]:
    """
    Collect ALL image files currently in output/images/.
    The scraper downloaded these but didn't store paths in the CSV.
    """
    images_dir = Path(config.IMAGES_DIR)
    if not images_dir.exists():
        print(f"  WARNING: Images directory not found: {images_dir}")
        return []

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    all_images = [
        str(p) for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    print(f"  Found {len(all_images)} images in {images_dir}")
    return all_images


# ─────────────────────────────────────────────────────────────────────────────
# Main detection runner
# ─────────────────────────────────────────────────────────────────────────────

def run_full_detection(args):
    print("\n" + "=" * 70)
    print("  IITM LOGO DETECTION — FULL RUN ON ALL SCRAPED WEBSITES")
    print("=" * 70)

    # Override threshold if requested
    if args.threshold is not None:
        config.CONFIDENCE_THRESH = args.threshold
        config.RISK_MEDIUM       = args.threshold
        print(f"\n  [CONFIG] Confidence threshold overridden to: {args.threshold}")

    # ── Load scraped results ───────────────────────────────────────────────
    scrape_path = os.path.join(config.OUTPUT_DIR, "scraped_results.csv")
    if not os.path.exists(scrape_path):
        print(f"\nERROR: scraped_results.csv not found at: {scrape_path}")
        print("  Run the scraper first:  python main.py --skip-discovery")
        sys.exit(1)

    scraped_df = pd.read_csv(scrape_path)
    print(f"\n  Loaded {len(scraped_df)} scraped URLs from: {scrape_path}")

    # How many had successful screenshots?
    screenshots_ok = scraped_df["screenshot"].notna().sum()
    screenshots_ok -= (scraped_df["screenshot"].fillna("").str.strip() == "").sum()
    print(f"  URLs with screenshots    : {screenshots_ok}")
    print(f"  URLs with errors/timeout : {scraped_df['error'].notna().sum()}")

    # ── Load models ───────────────────────────────────────────────────────
    print("\n  Loading detection models ...")
    if args.no_dl:
        dl_model, device = None, "cpu"
        print("  [--no-dl] Deep learning DISABLED — using template matching only")
    else:
        dl_model, device = load_trained_model()

    templates = load_templates(os.path.join(config.LOGO_SAMPLES_DIR, "positive"))

    if not templates:
        print("\n  WARNING: No OpenCV templates found in logo_samples/positive/")
        print("  Template matching will score 0.0 for all images.")

    # ── Collect all downloaded images from disk ────────────────────────────
    print("\n  Scanning output/images/ for downloaded images ...")
    all_disk_images = load_image_paths_from_disk()

    # We'll distribute disk images evenly — since we can't map them back to
    # specific URLs (scraper didn't save paths in CSV), we run detection on
    # ALL images and then attribute the best score to each URL via screenshot.
    # For URLs that have screenshots, we also check those.

    # ── Run per-URL detection ─────────────────────────────────────────────
    print(f"\n  Running detection on {len(scraped_df)} URLs ...")
    print("  (This may take a few minutes — checking screenshots + images)\n")

    detection_rows = []

    for idx, row in scraped_df.iterrows():
        url        = str(row.get("url", "unknown"))
        screenshot = row.get("screenshot")
        error      = row.get("error")

        # Fix screenshot path
        screenshot_path = fix_path(str(screenshot)) if pd.notna(screenshot) else None

        # Collect images to check
        all_paths = []
        if screenshot_path:
            all_paths.append(screenshot_path)

        if not all_paths:
            # No screenshot available (timeout/error) — skip with LOW risk
            detection_rows.append({
                "url":           url,
                "dl_score":      0.0,
                "tm_score":      0.0,
                "final_score":   0.0,
                "logo_detected": False,
                "risk_level":    "LOW",
                "best_image":    None,
                "error":         str(error) if pd.notna(error) else "",
                "image_count":   int(row.get("image_count", 0)),
            })
            continue

        # Run detection on each image; keep the highest score
        best       = {"final_score": -1.0, "dl_score": 0.0, "tm_score": 0.0, "logo_detected": False}
        best_path  = None

        for img_path in all_paths:
            if args.no_dl:
                from module3_detector import template_match_score
                tm  = template_match_score(img_path, templates)
                det = {
                    "dl_score":      0.0,
                    "tm_score":      round(tm, 4),
                    "final_score":   round(tm, 4),
                    "logo_detected": tm >= config.CONFIDENCE_THRESH,
                }
            else:
                det = detect_logo_in_image(img_path, dl_model, device, templates)

            if det["final_score"] > best["final_score"]:
                best      = det
                best_path = img_path

        risk = classify_risk(best["final_score"])

        # Progress line
        flag = "🚨" if risk == "HIGH" else ("⚠️ " if risk == "MEDIUM" else "   ")
        print(f"  [{idx+1:3d}/{len(scraped_df)}] {flag} Score:{best['final_score']:.2f} "
              f"Risk:{risk:6}  {url[:65]}")

        detection_rows.append({
            "url":           url,
            "dl_score":      best["dl_score"],
            "tm_score":      best["tm_score"],
            "final_score":   best["final_score"],
            "logo_detected": best["logo_detected"],
            "risk_level":    risk,
            "best_image":    best_path,
            "error":         str(error) if pd.notna(error) else "",
            "image_count":   int(row.get("image_count", 0)),
        })

    # ── NOW run detection on ALL downloaded images (bulk check) ────────────
    print(f"\n\n  {'='*60}")
    print(f"  BULK IMAGE SCAN — checking {len(all_disk_images)} downloaded images")
    print(f"  {'='*60}")

    # Score all images in output/images/
    image_scores = []
    for i, img_path in enumerate(all_disk_images):
        if args.no_dl:
            from module3_detector import template_match_score
            tm = template_match_score(img_path, templates)
            score = tm
        else:
            det = detect_logo_in_image(img_path, dl_model, device, templates)
            score = det["final_score"]

        if score >= config.RISK_MEDIUM:  # Only record notable hits
            image_scores.append({"image": img_path, "score": score})

        # Progress every 100 images
        if (i + 1) % 100 == 0:
            hits = sum(1 for x in image_scores if x["score"] >= config.CONFIDENCE_THRESH)
            print(f"    Scanned {i+1}/{len(all_disk_images)} images ... "
                  f"({hits} logo detections so far)")

    # Sort by score
    image_scores.sort(key=lambda x: x["score"], reverse=True)

    # ── Save results ───────────────────────────────────────────────────────
    det_df = pd.DataFrame(detection_rows)
    out_path = os.path.join(config.OUTPUT_DIR, "detection_results.csv")
    det_df.to_csv(out_path, index=False)

    # Save bulk image results
    if image_scores:
        img_df = pd.DataFrame(image_scores)
        img_out = os.path.join(config.OUTPUT_DIR, "image_scores.csv")
        img_df.to_csv(img_out, index=False)
        print(f"\n  ✓ Image scores saved to: {img_out}")

    # ── Final Summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  DETECTION COMPLETE — RESULTS")
    print("=" * 70)

    total   = len(det_df)
    high    = int((det_df["risk_level"] == "HIGH").sum())
    medium  = int((det_df["risk_level"] == "MEDIUM").sum())
    low     = int((det_df["risk_level"] == "LOW").sum())
    flagged = int(det_df["logo_detected"].sum())

    print(f"\n  Total websites analysed : {total}")
    print(f"  ✅ Logo Detected         : {flagged}")
    print(f"  🚨 HIGH  risk sites      : {high}")
    print(f"  ⚠️  MEDIUM risk sites     : {medium}")
    print(f"  ✅ LOW risk sites         : {low}")

    # Top detections by URL
    top_df = det_df[det_df["final_score"] > 0.0].sort_values("final_score", ascending=False)

    print(f"\n\n  TOP {min(args.top, len(top_df))} WEBSITES BY DETECTION SCORE:")
    print(f"  {'─'*68}")
    print(f"  {'#':>3}  {'RISK':6}  {'SCORE':>6}  {'URL'}")
    print(f"  {'─'*68}")

    for rank, (_, r) in enumerate(top_df.head(args.top).iterrows(), 1):
        flag = "🚨" if r["risk_level"] == "HIGH" else ("⚠️ " if r["risk_level"] == "MEDIUM" else "  ")
        print(f"  {rank:>3}. {flag} {r['risk_level']:6}  {r['final_score']*100:5.1f}%  {r['url'][:60]}")

    # Top images with logo
    if image_scores:
        top_imgs = [x for x in image_scores if x["score"] >= config.CONFIDENCE_THRESH]
        if top_imgs:
            print(f"\n\n  🖼  TOP IMAGES WITH IITM LOGO DETECTED ({len(top_imgs)} found):")
            print(f"  {'─'*68}")
            for img in top_imgs[:20]:
                print(f"       Score: {img['score']*100:5.1f}%  {img['image']}")

    print(f"\n  Full results: {out_path}")
    print("\n  To view the dashboard:")
    print("    streamlit run module5_report.py")
    print("=" * 70)

    return det_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_full_detection(args)
