# =============================================================================
# module5_report.py — Streamlit Dashboard + CSV/PDF Report Generation
# =============================================================================
# This module builds a rich interactive dashboard using Streamlit.
# It shows:
#   • Summary statistics and charts
#   • Per-website cards with screenshot, confidence score, risk level
#   • WHOIS domain ownership information
#   • Export buttons for CSV and PDF
#
# Run with:  streamlit run module5_report.py
# =============================================================================

import os
import io
import json
import datetime
import base64
import pandas as pd
import streamlit as st
from pathlib import Path

# WHOIS library for domain ownership lookup
try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    print("WARNING: python-whois not installed. WHOIS info will be unavailable.")

# PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage,
    )
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("WARNING: reportlab not installed. PDF export will be unavailable.")

import config   # Central config


# ─────────────────────────────────────────────────────────────────────────────
# WHOIS helper
# ─────────────────────────────────────────────────────────────────────────────

def get_whois_info(url: str) -> dict:
    """
    Query WHOIS for the given URL's domain.

    Returns a dict with: registrar, creation_date, expiration_date, name_servers
    Returns empty dict on failure.
    """
    if not WHOIS_AVAILABLE:
        return {}

    from urllib.parse import urlparse
    domain = urlparse(url).netloc.split(":")[0]

    try:
        w = whois.whois(domain)
        return {
            "registrar":       str(w.registrar or "N/A"),
            "creation_date":   str(w.creation_date or "N/A"),
            "expiration_date": str(w.expiration_date or "N/A"),
            "name_servers":    ", ".join(w.name_servers or []),
            "emails":          str(w.emails or "N/A"),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# PDF report builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(df: pd.DataFrame) -> bytes:
    """
    Generate a PDF report from the detection results DataFrame.

    Returns
    -------
    bytes — the raw PDF file content
    """
    if not REPORTLAB_AVAILABLE:
        return b""

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ──────────────────────────────────────────────────────────────
    title_style = styles["Title"]
    story.append(Paragraph("IITM Logo Misuse Detection Report", title_style))
    story.append(Paragraph(
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 20))

    # ── Summary stats ──────────────────────────────────────────────────────
    total   = len(df)
    flagged = df["logo_detected"].sum() if "logo_detected" in df.columns else 0
    high    = (df["risk_level"] == "HIGH").sum()   if "risk_level" in df.columns else 0
    medium  = (df["risk_level"] == "MEDIUM").sum() if "risk_level" in df.columns else 0

    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    summary_data = [
        ["Metric", "Value"],
        ["Total Websites Analysed",    str(total)],
        ["Websites with Logo Detected", str(flagged)],
        ["High Risk",                  str(high)],
        ["Medium Risk",                str(medium)],
        ["Detection Rate",             f"{(flagged/total*100):.1f}%" if total else "0%"],
    ]
    summary_table = Table(summary_data, colWidths=[250, 150])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f0f4ff")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # ── Detailed results table ─────────────────────────────────────────────
    story.append(Paragraph("Detailed Results", styles["Heading1"]))

    cols      = ["url", "final_score", "risk_level", "logo_detected"]
    available = [c for c in cols if c in df.columns]
    sub_df    = df[available].copy()

    if "final_score" in sub_df.columns:
        sub_df["final_score"] = sub_df["final_score"].apply(
            lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "N/A"
        )
    if "logo_detected" in sub_df.columns:
        sub_df["logo_detected"] = sub_df["logo_detected"].apply(
            lambda x: "YES" if x else "no"
        )

    # Truncate URLs so they fit in the PDF
    if "url" in sub_df.columns:
        sub_df["url"] = sub_df["url"].apply(lambda u: str(u)[:60] + "…" if len(str(u)) > 60 else str(u))

    headers  = [c.replace("_", " ").title() for c in available]
    table_data = [headers] + sub_df.values.tolist()

    detail_table = Table(table_data, repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
    ]))
    story.append(detail_table)

    # ── Methodology section ────────────────────────────────────────────────
    story.append(Spacer(1, 30))
    story.append(Paragraph("Methodology", styles["Heading1"]))
    story.append(Paragraph(
        "This report was generated by the IITM Logo Misuse Detection System, "
        "a student research project. The detection pipeline uses:\n"
        "1. Website discovery via crt.sh, Google CSE, and dnstwist\n"
        "2. Automated web scraping with Playwright\n"
        "3. Deep learning (ResNet-50) + OpenCV template matching\n"
        "4. Reverse image search via TinEye and SauceNAO\n\n"
        "Confidence scores above 85% are classified as HIGH risk, "
        "60–85% as MEDIUM risk, and below 60% as LOW risk.",
        styles["Normal"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    """Load the detection results CSV. Returns an empty DataFrame if not found."""
    det_path = os.path.join(config.OUTPUT_DIR, "detection_results.csv")
    scrp_path = os.path.join(config.OUTPUT_DIR, "scraped_results.csv")

    if os.path.exists(det_path):
        return pd.read_csv(det_path)
    elif os.path.exists(scrp_path):
        df = pd.read_csv(scrp_path)
        # Add placeholder detection columns so the dashboard still works
        df["final_score"]   = 0.0
        df["dl_score"]      = 0.0
        df["tm_score"]      = 0.0
        df["logo_detected"] = False
        df["risk_level"]    = "LOW"
        return df
    else:
        return pd.DataFrame()


def risk_color(risk: str) -> str:
    """Return a hex colour for a risk level string."""
    return {"HIGH": "#ff4444", "MEDIUM": "#ff8c00", "LOW": "#28a745"}.get(risk, "#888")


def image_to_base64(path: str) -> str | None:
    """Read an image file and return its base64-encoded string for HTML embedding."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    if ext in ("jpg", "jpeg"):
        ext = "jpeg"
    return f"data:image/{ext};base64,{data}"


def run_dashboard():
    """Main Streamlit dashboard application."""

    # ── Page config ────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="IITM Logo Misuse Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .main-header {
            background: linear-gradient(135deg, #1a3a6b 0%, #0e2040 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
        .main-header p  { margin: 0.5rem 0 0; opacity: 0.8; font-size: 1rem; }

        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border-left: 4px solid #1a3a6b;
        }
        .metric-card .value { font-size: 2.5rem; font-weight: 700; color: #1a3a6b; }
        .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.3rem; }

        .site-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 16px rgba(0,0,0,0.07);
            border-top: 3px solid #1a3a6b;
        }

        .risk-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            color: white;
        }

        .confidence-bar {
            background: #eee;
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
            margin: 8px 0;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 0.4s ease;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1a3a6b, #2a5298);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🔍 IITM Logo Misuse Detector</h1>
        <p>Automated Detection of Unauthorized IIT Madras Logo Usage on the Web</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_results()

    # ── Sidebar filters ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Filters")

        if not df.empty and "risk_level" in df.columns:
            risk_filter = st.multiselect(
                "Risk Level",
                options=["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"],
            )
            detected_only = st.checkbox("Show detected only", value=False)
            min_score = st.slider(
                "Minimum confidence score",
                min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            )
        else:
            risk_filter   = ["HIGH", "MEDIUM", "LOW"]
            detected_only = False
            min_score     = 0.0

        st.markdown("---")
        st.header("📁 Data Source")
        det_path = os.path.join(config.OUTPUT_DIR, "detection_results.csv")
        if os.path.exists(det_path):
            st.success(f"Loaded from:\n`{det_path}`")
        else:
            st.warning("No results found.\nRun the pipeline first.")

    # ── Empty-state guard ──────────────────────────────────────────────────
    if df.empty:
        st.info(
            "🚀 No results to display yet.\n\n"
            "Run the full pipeline with:\n```\npython main.py\n```"
        )
        return

    # ── Apply filters ──────────────────────────────────────────────────────
    view = df.copy()
    if "risk_level" in view.columns:
        view = view[view["risk_level"].isin(risk_filter)]
    if detected_only and "logo_detected" in view.columns:
        view = view[view["logo_detected"] == True]
    if "final_score" in view.columns:
        view = view[view["final_score"] >= min_score]

    # ── Summary metrics ────────────────────────────────────────────────────
    st.markdown("### 📊 Summary")
    col1, col2, col3, col4 = st.columns(4)

    total   = len(df)
    flagged = int(df["logo_detected"].sum()) if "logo_detected" in df.columns else 0
    high    = int((df["risk_level"] == "HIGH").sum())   if "risk_level" in df.columns else 0
    medium  = int((df["risk_level"] == "MEDIUM").sum()) if "risk_level" in df.columns else 0

    for col, val, label in [
        (col1, total,   "Total Websites Analysed"),
        (col2, flagged, "Logo Detected"),
        (col3, high,    "🔴 High Risk"),
        (col4, medium,  "🟠 Medium Risk"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────
    if "risk_level" in df.columns and "final_score" in df.columns:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Risk Distribution")
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            st.bar_chart(risk_counts.set_index("risk_level"))

        with chart_col2:
            st.markdown("#### Confidence Score Distribution")
            st.bar_chart(
                df["final_score"].dropna().apply(
                    lambda x: round(float(x), 1)
                ).value_counts().sort_index()
            )

    st.markdown("---")

    # ── Per-website cards ──────────────────────────────────────────────────
    st.markdown(f"### 🌐 Website Results ({len(view)} shown)")

    # Sort by score descending
    if "final_score" in view.columns:
        view = view.sort_values("final_score", ascending=False)

    for _, row in view.iterrows():
        url         = row.get("url", "N/A")
        score       = float(row.get("final_score", 0.0))
        risk        = row.get("risk_level", "LOW")
        detected    = bool(row.get("logo_detected", False))
        screenshot  = row.get("screenshot")
        dl_score    = float(row.get("dl_score", 0.0))
        tm_score    = float(row.get("tm_score", 0.0))
        error       = row.get("error")

        risk_col   = risk_color(risk)
        score_pct  = f"{score*100:.1f}%"
        fill_color = risk_col

        # Card HTML
        st.markdown(f"""
        <div class="site-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <a href="{url}" target="_blank" style="font-size:1.05rem; font-weight:600; color:#1a3a6b; text-decoration:none;">
                        {url}
                    </a>
                </div>
                <span class="risk-badge" style="background:{risk_col};">{risk}</span>
            </div>

            <div style="margin-bottom:8px;">
                <span style="font-size:0.9rem; color:#444;">
                    {"✅ IITM Logo <b>DETECTED</b>" if detected else "❌ Logo not detected"}
                    &nbsp;|&nbsp; Confidence: <b>{score_pct}</b>
                </span>
            </div>

            <div class="confidence-bar">
                <div class="confidence-fill"
                     style="width:{score_pct}; background:linear-gradient(90deg, {fill_color}99, {fill_color});"></div>
            </div>

            <div style="font-size:0.8rem; color:#888; margin-top:4px;">
                DL Score: {dl_score*100:.1f}% &nbsp;|&nbsp; Template Match: {tm_score*100:.1f}%
                {"&nbsp;|&nbsp; <span style='color:#cc0000;'>Error: " + str(error)[:60] + "</span>" if error and pd.notna(error) else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Screenshot thumbnail
        if screenshot and os.path.exists(str(screenshot)):
            with st.expander(f"📸 Screenshot — {url[:60]}"):
                b64 = image_to_base64(str(screenshot))
                if b64:
                    st.image(str(screenshot), caption=url, width="stretch")

        # WHOIS info (on demand)
        with st.expander(f"🔎 WHOIS Info — {url[:60]}"):
            if st.button(f"Fetch WHOIS for {url[:40]}…", key=f"whois_{url}"):
                with st.spinner("Querying WHOIS ..."):
                    info = get_whois_info(url)
                if info:
                    for k, v in info.items():
                        st.write(f"**{k.replace('_',' ').title()}:** {v}")
                else:
                    st.write("WHOIS data unavailable.")

    # ── Export buttons ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Reports")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_data,
            file_name=f"iitm_logo_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    with export_col2:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_pdf_report(view)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"iitm_logo_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("Install `reportlab` to enable PDF export:\n```pip install reportlab```")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_dashboard()
