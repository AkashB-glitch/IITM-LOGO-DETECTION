# =============================================================================
# generate_sample_data.py — Create Demo Data for Testing Without Live Websites
# =============================================================================
# This script creates:
#   1. Synthetic "positive" and "negative" training images for the DL model
#   2. A fake detection_results.csv so the dashboard can display something
#      before the full pipeline has been run
#
# Run this FIRST to make sure the dashboard works immediately:
#   python generate_sample_data.py
# =============================================================================

import os
import csv
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import config


# ─────────────────────────────────────────────────────────────────────────────
# Generate synthetic logo images (positive samples)
# ─────────────────────────────────────────────────────────────────────────────

def draw_iitm_logo_placeholder(draw: ImageDraw.ImageDraw, x: int, y: int, size: int):
    """
    Draw a simplified placeholder that approximates the IITM logo colours/shape.
    (Replace these with real IITM logo images for actual use.)
    """
    # Outer circle — dark blue
    draw.ellipse([x, y, x + size, y + size], fill=(10, 45, 110), outline=(255, 200, 0), width=4)

    # Inner circle — white
    margin = size // 6
    draw.ellipse(
        [x + margin, y + margin, x + size - margin, y + size - margin],
        fill=(255, 255, 255),
    )

    # IIT text in the centre
    cx, cy = x + size // 2, y + size // 2
    try:
        font = ImageFont.truetype("arial.ttf", size // 6)
    except OSError:
        font = ImageFont.load_default()

    draw.text((cx, cy - size // 10), "IITM", fill=(10, 45, 110), font=font, anchor="mm")
    draw.text((cx, cy + size // 8), "Madras", fill=(10, 45, 110), font=font, anchor="mm")


def generate_positive_images(n: int = 50, out_dir: str = None):
    """Generate n synthetic images WITH the IITM logo placeholder."""
    if out_dir is None:
        out_dir = os.path.join(config.LOGO_SAMPLES_DIR, "positive")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {n} positive (logo) images → {out_dir}")
    for i in range(n):
        w, h = 400, 300
        # Random background colour (simulate a web page)
        bg = tuple(random.randint(200, 255) for _ in range(3))
        img  = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)

        # Add some random text lines (simulate page content)
        for _ in range(5):
            tx = random.randint(0, w - 100)
            ty = random.randint(0, h - 20)
            draw.text((tx, ty), "Sample web content", fill=(100, 100, 100))

        # Draw logo at a random position and size
        logo_size = random.randint(60, 120)
        lx = random.randint(0, w - logo_size)
        ly = random.randint(0, h - logo_size)
        draw_iitm_logo_placeholder(draw, lx, ly, logo_size)

        # Apply slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        img.save(os.path.join(out_dir, f"positive_{i:04d}.png"))

    print(f"  ✓ {n} positive images saved.")


def generate_negative_images(n: int = 50, out_dir: str = None):
    """Generate n synthetic images WITHOUT any logo (negative samples)."""
    if out_dir is None:
        out_dir = os.path.join(config.LOGO_SAMPLES_DIR, "negative")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {n} negative (no logo) images → {out_dir}")
    shapes = ["rect", "circle", "text_only"]

    for i in range(n):
        w, h = 400, 300
        bg   = tuple(random.randint(200, 255) for _ in range(3))
        img  = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)

        shape = random.choice(shapes)
        color = tuple(random.randint(0, 200) for _ in range(3))

        if shape == "rect":
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            draw.rectangle([x1, y1, x1 + 100, y1 + 80], fill=color)
        elif shape == "circle":
            cx = random.randint(50, w - 50)
            cy = random.randint(50, h - 50)
            r  = random.randint(20, 70)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:
            for _ in range(8):
                draw.text(
                    (random.randint(0, w - 80), random.randint(0, h - 15)),
                    "Generic content here",
                    fill=color,
                )

        img.save(os.path.join(out_dir, f"negative_{i:04d}.png"))

    print(f"  ✓ {n} negative images saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Generate fake detection results (for dashboard demo)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_SITES = [
    ("https://iitm-coaching-center.com",       0.93, "HIGH",   True),
    ("https://fake-iitm-certificate.in",       0.88, "HIGH",   True),
    ("https://iitmadras-online-degree.com",    0.82, "HIGH",   True),
    ("https://iit-madras-courses.net",         0.76, "MEDIUM", True),
    ("https://iitmstudentclub.org",            0.71, "MEDIUM", True),
    ("https://iitm-alumni.info",               0.65, "MEDIUM", True),
    ("https://random-edtech-site.com",         0.42, "LOW",    False),
    ("https://some-university-portal.edu",     0.31, "LOW",    False),
    ("https://legituniversity.ac",             0.22, "LOW",    False),
    ("https://iitm-scholarship-apply.com",     0.91, "HIGH",   True),
    ("https://iitmadras-fee-payment.com",      0.85, "HIGH",   True),
    ("https://iitm-fake-alumni-portal.net",    0.79, "MEDIUM", True),
    ("https://study-at-iitm.com",             0.68, "MEDIUM", True),
    ("https://innocentsite.org",               0.18, "LOW",    False),
    ("https://another-random-blog.com",        0.09, "LOW",    False),
]


def generate_fake_screenshots(sites: list, screenshots_dir: str):
    """Generate small synthetic screenshot images for the demo."""
    os.makedirs(screenshots_dir, exist_ok=True)
    screenshot_paths: list[str | None] = []

    for url, score, risk, detected in sites:
        # Create a simple page image
        w, h = 800, 400
        bg_color = (240, 245, 255) if detected else (245, 245, 245)
        img  = Image.new("RGB", (w, h), bg_color)
        draw = ImageDraw.Draw(img)

        # Navigation bar
        draw.rectangle([0, 0, w, 50], fill=(10, 45, 110))
        draw.text((20, 15), url[:60], fill=(255, 255, 255))

        # Simulated content
        for row in range(4):
            y = 70 + row * 60
            draw.rectangle([20, y, random.randint(200, 750), y + 20], fill=(220, 220, 220))

        if detected:
            # Draw logo placeholder
            draw_iitm_logo_placeholder(draw, w - 160, 70, 120)
            draw.text((w - 160, 210), "IIT Madras", fill=(10, 45, 110))

        # Save
        import hashlib
        fname = hashlib.md5(url.encode()).hexdigest()[:12] + ".png"
        path  = os.path.join(screenshots_dir, fname)
        img.save(path)
        screenshot_paths.append(path)

    return screenshot_paths


def generate_fake_results_csv():
    """Create a sample detection_results.csv so the dashboard has data to show."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    screenshots_dir = config.SCREENSHOTS_DIR
    os.makedirs(screenshots_dir, exist_ok=True)

    print(f"\nGenerating fake detection results → {config.OUTPUT_DIR}/detection_results.csv")

    shot_paths = generate_fake_screenshots(DEMO_SITES, screenshots_dir)

    rows = []
    for (url, score, risk, detected), shot in zip(DEMO_SITES, shot_paths):
        noise = random.uniform(-0.05, 0.05)
        dl_s  = max(0, min(1, score + noise))
        tm_s  = max(0, min(1, score - 0.05 + noise))
        rows.append({
            "url":           url,
            "dl_score":      round(dl_s, 4),
            "tm_score":      round(tm_s, 4),
            "final_score":   round(score, 4),
            "logo_detected": detected,
            "risk_level":    risk,
            "best_image":    shot,
            "screenshot":    shot,
            "error":         None,
        })

    out = os.path.join(config.OUTPUT_DIR, "detection_results.csv")
    df_rows = rows
    import pandas as pd
    pd.DataFrame(df_rows).to_csv(out, index=False)
    print(f"  ✓ {len(rows)} demo results saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Generating sample data for IITM Logo Misuse Detector ...")
    print("=" * 60)

    print("\n[1/3] Generating positive training images ...")
    generate_positive_images(n=30)

    print("\n[2/3] Generating negative training images ...")
    generate_negative_images(n=30)

    print("\n[3/3] Generating demo detection results for dashboard ...")
    generate_fake_results_csv()

    print("\n✓ All sample data generated successfully!")
    print("\nNext steps:")
    print("  1. streamlit run module5_report.py   ← view the demo dashboard")
    print("  2. python main.py --train            ← train the DL model")
    print("  3. python main.py --demo             ← run full pipeline on demo URLs")
