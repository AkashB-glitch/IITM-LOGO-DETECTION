# =============================================================================
# config.py — Central Configuration for IITM Logo Detection Project
# =============================================================================
# Fill in your API keys here before running the project.
# All free-tier keys work for a student project.
# =============================================================================

# ── Google Custom Search API ──────────────────────────────────────────────────
# Get yours at: https://developers.google.com/custom-search/v1/introduction
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
GOOGLE_CSE_ID  = "YOUR_CUSTOM_SEARCH_ENGINE_ID_HERE"

# ── TinEye Reverse Image Search (optional) ────────────────────────────────────
# Free tier: 150 searches/month — https://services.tineye.com/TinEyeAPI
TINEYE_API_KEY = "YOUR_TINEYE_API_KEY_HERE"

# ── Paths & Directories ───────────────────────────────────────────────────────
OUTPUT_DIR       = "output"            # Root folder for all saved results
SCREENSHOTS_DIR  = "output/screenshots"
IMAGES_DIR       = "output/images"
REPORTS_DIR      = "output/reports"
MODEL_DIR        = "models"            # Saved PyTorch model weights
LOGO_SAMPLES_DIR = "logo_samples"      # Positive IITM logo images for training

# ── IITM Whitelist ────────────────────────────────────────────────────────────
# Any URL containing these strings is considered OFFICIAL and is skipped.
WHITELIST_DOMAINS = [
    "iitm.ac.in",
    "iit-madras.ac.in",
    "iitmadras.ac.in",
    "smail.iitm.ac.in",
    "research.iitm.ac.in",
    "alumni.iitm.ac.in",
    "placement.iitm.ac.in",
]

# ── Discovery Settings ────────────────────────────────────────────────────────
CRT_SH_QUERIES    = ["iitm", "iit-madras", "iitmadras"]   # crt.sh certificate search
GOOGLE_QUERIES    = [
    '"IIT Madras" -site:iitm.ac.in',
    '"IITM" logo -site:iitm.ac.in',
    '"IIT Madras logo" -site:iitm.ac.in',
]
MAX_GOOGLE_RESULTS_PER_QUERY = 10   # Free tier: 100/day total
MAX_CANDIDATE_URLS           = 1000  # Cap total URLs to analyse

# ── Playwright Settings ───────────────────────────────────────────────────────
PAGE_TIMEOUT      = 20_000   # ms (20 seconds) before giving up on a page
SCROLL_PAUSE      = 1.0      # seconds to pause while scrolling
HEADLESS          = True     # Set False to watch the browser during debugging

# ── Deep Learning Settings ────────────────────────────────────────────────────
IMG_SIZE          = 224      # Input size for ResNet
BATCH_SIZE        = 16
EPOCHS            = 10
LEARNING_RATE     = 1e-4
TRAIN_SPLIT       = 0.8      # 80% training, 20% validation
CONFIDENCE_THRESH = 0.70     # Above this → "logo found"

# ── OpenCV Template Matching ──────────────────────────────────────────────────
TM_THRESHOLD      = 0.75     # cv2.matchTemplate score threshold (0–1)

# ── Risk Level Thresholds ─────────────────────────────────────────────────────
RISK_HIGH   = 0.85  # Confidence ≥ 85% → HIGH risk
RISK_MEDIUM = 0.60  # Confidence 60–84% → MEDIUM risk
                    # Below 60% → LOW risk
