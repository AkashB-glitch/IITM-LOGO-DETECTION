# =============================================================================
# Project Report: Automated Detection of Unauthorized IITM Logo Usage on the Web
# =============================================================================
# Course Project | IIT Madras
# =============================================================================

---

# Automated Detection of Unauthorized IIT Madras Logo Usage on the Web

**Project Type:** Student Research Project  
**Technology Stack:** Python, PyTorch, OpenCV, Playwright, Streamlit  
**Date:** June 2026

---

## 1. Introduction

### 1.1 Background
The IIT Madras (IITM) logo is a registered trademark and intellectual property of the Indian Institute of Technology Madras. With the rapid growth of the internet, unauthorized usage of the IITM logo has become a significant concern. Fraudulent websites use the official logo to:
- Sell fake course certificates
- Conduct phishing attacks targeting students and alumni
- Impersonate official IITM services to collect fees
- Mislead prospective students with fake admission portals

Manual detection of such misuse is impractical given the scale of the internet.

### 1.2 Objective
This project builds a **fully automated pipeline** to:
1. Discover suspicious websites that might be using the IITM logo
2. Automatically visit and capture visual evidence from each website
3. Detect logo presence using deep learning and computer vision
4. Generate actionable reports for legal follow-up

### 1.3 Scope
- Analyses **~500–1000 candidate URLs** (not the entire internet)
- Runs on a **standard laptop or Google Colab**
- Designed as a **student-level educational project**

---

## 2. Methodology

### 2.1 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Module 1      │───▶│   Module 4      │───▶│   Module 2      │
│   Discovery     │    │ Reverse Search  │    │   Scraping      │
│ (crt.sh, Google │    │ (TinEye,        │    │ (Playwright     │
│  dnstwist)      │    │  SauceNAO)      │    │  Screenshots)   │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                        │
                                               ┌────────▼────────┐
                                               │   Module 3      │
                                               │   Detection     │
                                               │ (ResNet + OpenCV│
                                               └────────┬────────┘
                                                        │
                                               ┌────────▼────────┐
                                               │   Module 5      │
                                               │   Dashboard     │
                                               │ (Streamlit +    │
                                               │  CSV/PDF)       │
                                               └─────────────────┘
```

### 2.2 Module 1 — Website Discovery

Three independent data sources are queried and merged:

**a) crt.sh Certificate Transparency Logs**
- SSL/TLS certificates are publicly logged for transparency
- Any domain containing "iitm", "iit-madras", or "iitmadras" is flagged
- No API key required; completely free
- Typical yield: 50–200 domains

**b) Google Custom Search API**
- Searches for pages mentioning "IIT Madras" while excluding the official site
- Queries: `"IIT Madras" -site:iitm.ac.in`, `"IITM logo" -site:iitm.ac.in`
- Free tier: 100 queries/day
- Typical yield: 30–100 URLs

**c) dnstwist Typo-squatting**
- Generates variations of `iitm.ac.in` (e.g., `iittm.ac.in`, `iltm.ac.in`)
- Checks which variations are actually registered domains
- Identifies deliberate impersonation attempts
- Typical yield: 10–50 domains

**Whitelist Filtering:**  
All official IITM domains (`*.iitm.ac.in`) are automatically removed to avoid false positives.

### 2.3 Module 2 — Automated Web Scraping

**Tool:** Playwright (Chromium browser automation)

**Process:**
1. Open a real Chromium browser (not just HTTP requests)
2. Navigate to the URL with a realistic user-agent
3. Scroll down the page to trigger lazy-loaded images
4. Capture a full-page screenshot
5. Extract all `<img>` src URLs and CSS background-image URLs
6. Download up to 50 images per page
7. Extract all visible text content

**Why Playwright instead of requests/BeautifulSoup?**
- Many modern websites use JavaScript to load content dynamically
- Simple HTTP requests would miss JavaScript-rendered logos
- Playwright executes JavaScript just like a real browser

### 2.4 Module 3 — Deep Learning Logo Detection

#### Method A: ResNet-50 Transfer Learning

**Architecture:**
- Base: ResNet-50 pretrained on ImageNet (1.28M images, 1000 classes)
- Modification: Replace final fully-connected layer with:
  - Linear(2048 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2)
- Output: Binary classification (logo / no logo)

**Transfer Learning Rationale:**
- ResNet-50 already learned powerful feature extractors (edges, textures, shapes)
- We only need to teach it the specific appearance of the IITM logo
- This works well even with a small dataset (~50–100 samples per class)

**Training:**
- All pretrained layers are frozen
- Only the new FC head is trained
- Optimizer: Adam (lr=1e-4)
- Scheduler: StepLR (step=5, γ=0.5)
- Data augmentation: random crops, horizontal flips, color jitter

#### Method B: OpenCV Template Matching

**Algorithm:** `cv2.TM_CCOEFF_NORMED`
- Slides the logo template across the image
- Returns a normalized correlation score (0 to 1)
- Applied at multiple scales (0.5×, 0.75×, 1.0×, 1.25×, 1.5×) to handle resized logos

**Advantages:**
- Fast and deterministic
- No training required
- Works well for exact or near-exact logo copies

**Limitations:**
- Fails if the logo is significantly modified (color change, rotation)
- This is why DL is the primary method

**Final Score:** `max(DL_score, OpenCV_score)`

### 2.5 Module 4 — Reverse Image Search

Three services are queried with the official IITM logo:

| Service | Free Tier | API Key |
|---------|-----------|---------|
| TinEye | 150/month | Required |
| Google Image Search | 100/day | Required |
| SauceNAO | 200/day | Not required |

Results are merged with Module 1 candidates.

### 2.6 Module 5 — Report Generation

**Streamlit Dashboard** displays:
- Summary statistics (total, detected, risk counts)
- Bar charts (risk distribution, score histogram)
- Per-site cards with screenshot, score, risk badge
- WHOIS domain ownership information
- CSV and PDF export

**Risk Classification:**
| Risk Level | Confidence Score | Recommended Action |
|------------|-----------------|-------------------|
| HIGH (🔴) | ≥ 85% | Immediate legal notice |
| MEDIUM (🟠) | 60–84% | Manual review required |
| LOW (🟢) | < 60% | Monitor / ignore |

---

## 3. Results

### 3.1 Expected Performance

| Metric | Expected Value |
|--------|---------------|
| Discovery yield | 300–800 candidate URLs |
| Scraping success rate | 60–80% (some sites block bots) |
| DL model accuracy (val) | 85–95% (with sufficient training data) |
| Template matching precision | ~90% (for exact logo copies) |
| Overall false positive rate | ~10–20% |
| Overall false negative rate | ~5–15% |

### 3.2 Sample Detections (Demo Data)

The demo data (`generate_sample_data.py`) simulates typical findings:

| Website | Confidence | Risk | Finding |
|---------|-----------|------|---------|
| fake-iitm-certificate.in | 93% | HIGH | Logo on certificate page |
| iitmadras-online-degree.com | 88% | HIGH | Logo in header |
| iit-madras-courses.net | 76% | MEDIUM | Logo in footer |
| iitmstudentclub.org | 71% | MEDIUM | Logo on about page |

### 3.3 Computational Requirements

| Hardware | Training Time | Inference (per URL) |
|----------|--------------|---------------------|
| CPU only | ~30 min (10 epochs) | ~2–5 sec |
| GPU (T4) | ~3 min (10 epochs) | ~0.2 sec |
| Google Colab (free) | ~5 min | ~0.5 sec |

---

## 4. Limitations

1. **Dynamic Content:** Some sites use anti-bot measures (Cloudflare, reCAPTCHA) that block Playwright
2. **Logo Modifications:** Heavily modified logos (color-inverted, distorted) may not be detected
3. **Training Data:** Model accuracy depends on quality and diversity of training images
4. **API Limits:** Free tiers limit discovery to ~100 Google results/day
5. **Legal Disclaimer:** Detection is probabilistic — human review is required before legal action

---

## 5. Future Improvements

1. **Perceptual Hashing (pHash):** Add image hashing for near-duplicate detection
2. **CLIP Model:** Use OpenAI's CLIP for zero-shot logo detection without training data
3. **Continuous Monitoring:** Schedule weekly runs with new URL discovery
4. **Browser Fingerprint Evasion:** Use playwright-stealth to bypass bot detection
5. **Automated DMCA:** Generate pre-filled DMCA takedown notices for HIGH risk sites

---

## 6. Conclusion

This project demonstrates a practical, end-to-end pipeline for automated intellectual property monitoring. By combining:
- **Certificate transparency logs** for systematic domain discovery
- **Playwright** for reliable JavaScript-rendered page capture
- **Transfer learning** for accurate logo detection with limited training data
- **Template matching** as a fast, interpretable backup

...we achieve a scalable system that can scan hundreds of websites in a few hours on consumer hardware.

The most significant contribution is the **dual-method detection** approach (DL + OpenCV), which achieves higher recall than either method alone, reducing the risk of missing genuine infringements.

---

## 7. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016.*
2. Playwright Documentation: https://playwright.dev/python/
3. Certificate Transparency: https://certificate.transparency.dev/
4. dnstwist: https://github.com/elceef/dnstwist
5. OpenCV Template Matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
6. TinEye API: https://services.tineye.com/TinEyeAPI
7. Google Custom Search API: https://developers.google.com/custom-search/v1/introduction

---

*Submitted as a student research project. For educational purposes only.*
