# Automated Detection of Unauthorized IITM Logo Usage on the Web

> **Student Project** — IIT Madras | Python + Deep Learning  
> Detects websites that illegally use the IITM logo without authorization.

---

## 📁 Project Structure

```
iitm-logo-detector/
├── config.py                  ← All API keys and settings (edit this first)
├── main.py                    ← Master pipeline orchestrator
├── module1_discovery.py       ← Find suspicious websites (crt.sh, Google, dnstwist)
├── module2_scraper.py         ← Visit sites with Playwright, take screenshots
├── module3_detector.py        ← ResNet-50 + OpenCV template matching
├── module4_reverse_search.py  ← TinEye / SauceNAO reverse image search
├── module5_report.py          ← Streamlit dashboard + CSV/PDF export
├── generate_sample_data.py    ← Create demo data (run this first!)
├── requirements.txt           ← Python dependencies
│
├── logo_samples/
│   ├── positive/              ← Images CONTAINING the IITM logo (add your own)
│   └── negative/              ← Images WITHOUT the logo
│
├── output/
│   ├── candidate_urls.csv     ← Discovered suspicious URLs
│   ├── scraped_results.csv    ← Scraping results
│   ├── detection_results.csv  ← Final detection scores
│   ├── screenshots/           ← Full-page screenshots of each site
│   └── images/                ← Individual images extracted from each site
│
└── models/
    └── logo_detector.pth      ← Trained PyTorch model weights
```

---

## 🚀 Quick Start (5 Steps)

### Step 1 — Clone & Install

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install all dependencies
pip install -r requirements.txt

# Install the Playwright browser
playwright install chromium
```

### Step 2 — Configure API Keys

Open `config.py` and fill in your API keys:

```python
GOOGLE_API_KEY = "your-key-here"   # https://console.cloud.google.com
GOOGLE_CSE_ID  = "your-cse-id"     # https://programmablesearchengine.google.com
TINEYE_API_KEY = "your-key-here"   # https://services.tineye.com (optional)
```

> **Note:** crt.sh and SauceNAO work without any API key.  
> Google CSE gives 100 free queries/day.  
> TinEye gives 150 free searches/month.

### Step 3 — Add IITM Logo Images (Training Data)

Place real IITM logo images in:
- `logo_samples/positive/` — images **with** the IITM logo (PNG/JPG)
- `logo_samples/negative/` — images **without** the logo (any other images)

You need **at least 20 images in each folder** for a meaningful model.  
Or run `generate_sample_data.py` for synthetic placeholders:

```bash
python generate_sample_data.py
```

### Step 4 — Run the Full Pipeline

```bash
# Full pipeline (all 5 modules)
python main.py

# With training the DL model first
python main.py --train

# Demo mode (no API keys needed, uses 5 test URLs)
python main.py --demo

# Skip discovery (reuse existing candidate_urls.csv)
python main.py --skip-discovery

# Skip scraping (reuse existing screenshots)
python main.py --skip-scrape

# Limit to first 50 URLs (for quick testing)
python main.py --max-urls 50
```

### Step 5 — View the Dashboard

```bash
streamlit run module5_report.py
```

Open your browser at **http://localhost:8501**

---

## 🔬 How It Works

### Module 1 — Discovery
Finds ~500–1000 suspicious websites from three sources:

| Source | Method | Requires Key? |
|--------|--------|---------------|
| **crt.sh** | SSL certificate transparency logs | ❌ No |
| **Google CSE** | Custom Search API | ✅ Yes (free) |
| **dnstwist** | Typo-squat domain generation | ❌ No |

All official IITM domains (`iitm.ac.in` and subdomains) are automatically removed.

### Module 2 — Scraping
Uses **Playwright** (real Chromium browser) to:
- Visit each URL like a human
- Scroll to load lazy images
- Take a full-page screenshot
- Download all images from the page
- Extract visible text

### Module 3 — Detection (Two Methods)

**Method A — Deep Learning (ResNet-50)**
- Pretrained on ImageNet (1.2M images)
- Fine-tuned on IITM logo images vs. non-logo images
- Outputs a confidence score (0–100%)

**Method B — OpenCV Template Matching**
- Uses official IITM logo as a template
- Slides it across each image at multiple scales
- Fast and interpretable

Final score = **MAX(DL score, Template score)**

### Module 4 — Reverse Image Search
Submits the IITM logo to:
- **TinEye** — finds exact copies online
- **Google Image Search** — finds visually similar images
- **SauceNAO** — free, no key needed

### Module 5 — Dashboard
Interactive Streamlit UI with:
- Risk level badges (🔴 HIGH / 🟠 MEDIUM / 🟢 LOW)
- Confidence score bars
- Screenshot viewer
- WHOIS domain info
- CSV + PDF export

---

## ⚙️ Configuration Reference

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CONFIDENCE_THRESH` | 0.70 | Logo detected if score ≥ this |
| `RISK_HIGH` | 0.85 | HIGH risk threshold |
| `RISK_MEDIUM` | 0.60 | MEDIUM risk threshold |
| `MAX_CANDIDATE_URLS` | 1000 | Max URLs to analyse |
| `PAGE_TIMEOUT` | 20000ms | Time before giving up on a page |
| `HEADLESS` | True | Set False to watch the browser |
| `EPOCHS` | 10 | DL training epochs |
| `IMG_SIZE` | 224 | ResNet input image size |

---

## 🐛 Troubleshooting

**`playwright install chromium` fails**
```bash
pip install playwright --upgrade
playwright install --with-deps chromium
```

**`dnstwist` not found**
```bash
pip install dnstwist
```

**CUDA out of memory**
- Reduce `BATCH_SIZE` in `config.py` (try 8 or 4)

**Google CSE returns no results**
- Check your API key and CSE ID in `config.py`
- Ensure your Custom Search Engine is set to search the entire web

**Empty training dataset**
```bash
python generate_sample_data.py   # creates synthetic placeholder images
```

---

## 📊 Sample Output

After running the full pipeline, `output/detection_results.csv` looks like:

| url | final_score | risk_level | logo_detected |
|-----|-------------|------------|---------------|
| https://fake-iitm-cert.com | 0.93 | HIGH | True |
| https://iitmadras-online.com | 0.76 | MEDIUM | True |
| https://random-site.org | 0.21 | LOW | False |

---

## 📝 Project Report Summary

**Introduction:**  
This project automates the detection of unauthorized IIT Madras logo usage across the web, combining web scraping, deep learning, and reverse image search techniques.

**Methodology:**  
1. Discovery using certificate transparency, search APIs, and typo-squat analysis
2. Automated browser-based scraping with Playwright
3. Dual-method detection: ResNet-50 CNN (transfer learning) + OpenCV template matching
4. Reverse image search cross-validation

**Expected Results:**  
~5–15% of discovered domains may show suspicious logo usage (varies with data source quality).

**Conclusion:**  
This pipeline provides a scalable, automated approach to intellectual property monitoring, demonstrating practical applications of deep learning in web security research.

---

## 🔗 References

- [crt.sh Certificate Search](https://crt.sh)
- [Google Custom Search API](https://developers.google.com/custom-search/v1)
- [dnstwist](https://github.com/elceef/dnstwist)
- [Playwright for Python](https://playwright.dev/python/)
- [ResNet Paper (He et al., 2016)](https://arxiv.org/abs/1512.03385)
- [TinEye API](https://services.tineye.com)
- [python-whois](https://pypi.org/project/python-whois/)

---

*This project is for educational and research purposes only.*
