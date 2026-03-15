import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import time

# ════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EstateIQ Pro — AI House Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# ════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Navy · White · Cyan palette, crisp and professional
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --navy:     #0D1B2A;
    --navy2:    #132338;
    --navy3:    #1A3050;
    --blue:     #1565C0;
    --cyan:     #00BCD4;
    --cyan-lt:  #4DD0E1;
    --amber:    #FFB300;
    --green:    #00C853;
    --red:      #FF5252;
    --purple:   #7C4DFF;
    --white:    #FFFFFF;
    --offwhite: #F0F4F8;
    --g100:     #E3EAF2;
    --g300:     #B0BEC5;
    --g600:     #546E7A;
    --text:     #1A2744;
    --r:        14px;
    --sh:       0 4px 24px rgba(13,27,42,0.10);
    --sh-lg:    0 12px 48px rgba(13,27,42,0.16);
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--offwhite) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px !important; }

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy2) 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(13,27,42,0.3) !important;
}
[data-testid="stSidebar"] * { color: #CBD8E8 !important; }
[data-testid="stSidebar"] label {
    font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: var(--g300) !important;
}
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] [data-baseweb="input"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(0,188,212,0.3) !important;
    border-radius: 8px !important; color: var(--white) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(0,188,212,0.3) !important;
    border-radius: 8px !important; color: var(--white) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--cyan) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #0077B6, var(--cyan)) !important;
    color: var(--white) !important; font-weight: 700 !important;
    font-size: 0.88rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 10px !important; padding: 0.75rem 1rem !important;
    width: 100% !important; cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px rgba(0,188,212,0.45) !important;
    margin-top: 8px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 28px rgba(0,188,212,0.6) !important;
}

/* ══ CARDS ══ */
.card {
    background: var(--white); border-radius: var(--r);
    padding: 1.6rem 1.8rem; box-shadow: var(--sh);
    border: 1px solid var(--g100); height: 100%;
}
.card-dark {
    background: linear-gradient(145deg, var(--navy) 0%, var(--navy3) 100%);
    border-radius: var(--r); padding: 1.8rem;
    box-shadow: var(--sh-lg); border: 1px solid rgba(0,188,212,0.25);
    position: relative; overflow: hidden;
}
.card-dark::after {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,188,212,0.18) 0%, transparent 70%);
    pointer-events: none;
}

/* ══ HERO ══ */
.hero {
    background: linear-gradient(135deg, var(--navy) 0%, #0A3060 50%, #0D4080 100%);
    border-radius: 18px; padding: 2.4rem 3rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
    box-shadow: 0 16px 60px rgba(13,27,42,0.22);
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%2300BCD4' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/svg%3E");
}
.hero::after {
    content: ''; position: absolute; bottom: -60px; right: 5%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,188,212,0.15) 0%, transparent 65%);
}
.hero-eyebrow {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.25em;
    text-transform: uppercase; color: var(--cyan); margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
    content: ''; display: inline-block; width: 28px; height: 2px;
    background: var(--cyan);
}
.hero-title {
    font-family: 'DM Serif Display', serif !important; font-size: 2.6rem;
    font-weight: 800; color: var(--white); line-height: 1.15;
    margin: 0 0 0.6rem; letter-spacing: -0.02em;
}
.hero-desc { font-size: 0.92rem; color: #8BAEC8; max-width: 520px; line-height: 1.6; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,200,83,0.15); border: 1px solid rgba(0,200,83,0.4);
    border-radius: 50px; padding: 0.28rem 0.8rem; font-size: 0.7rem;
    font-weight: 600; color: #00E676; letter-spacing: 0.08em; margin-top: 1rem;
}
.hero-badge::before { content: '●'; font-size: 0.5rem; }

/* ══ KPI STRIP ══ */
.kpi-strip { display: flex; gap: 1rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.kpi-box {
    background: var(--white); border: 1px solid var(--g100);
    border-radius: 12px; padding: 1rem 1.4rem; flex: 1; min-width: 130px;
    box-shadow: var(--sh); border-top: 3px solid var(--cyan);
}
.kpi-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--g600); margin-bottom: 0.35rem; }
.kpi-val   { font-size: 1.55rem; font-weight: 800; color: var(--navy); font-family: 'DM Serif Display', serif; }
.kpi-sub   { font-size: 0.7rem; color: var(--g600); margin-top: 0.1rem; }

/* ══ PRICE CARD ══ */
.price-hero  { text-align: center; padding: 1.2rem 1rem 1rem; }
.price-eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: var(--cyan); margin-bottom: 0.4rem; }
.price-main { font-family: 'DM Serif Display', serif; font-size: 3rem; font-weight: 800; color: var(--white); line-height: 1; margin-bottom: 0.3rem; }
.price-main span { color: var(--cyan); }
.price-range { font-size: 0.78rem; color: #8BAEC8; margin-bottom: 1rem; }
.price-divider { height: 1px; background: rgba(0,188,212,0.2); margin: 0.8rem 0; }
.conf-label { font-size: 0.68rem; font-weight: 600; color: #8BAEC8; margin-bottom: 0.4rem; display: flex; justify-content: space-between; }
.conf-bar   { height: 6px; background: rgba(255,255,255,0.1); border-radius: 50px; overflow: hidden; }
.conf-fill  { height: 100%; background: linear-gradient(90deg, var(--cyan), #00E5FF); border-radius: 50px; }

/* ══ SECTION TITLE ══ */
.sec-title {
    font-family: 'DM Serif Display', serif; font-size: 1.12rem; font-weight: 800;
    color: var(--navy); letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 8px; margin-bottom: 1.2rem;
}
.sec-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--g100), transparent);
}

/* ══ CHIPS ══ */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.chip { display: inline-flex; align-items: center; gap: 5px; border-radius: 50px; padding: 0.3rem 0.75rem; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; }
.chip-on  { background: rgba(0,188,212,0.12); border: 1px solid rgba(0,188,212,0.5); color: #0097A7; }
.chip-off { background: var(--g100); border: 1px solid transparent; color: var(--g600); }
.chip-dot { width: 6px; height: 6px; border-radius: 50%; }
.chip-on  .chip-dot { background: var(--cyan); }
.chip-off .chip-dot { background: var(--g300); }

/* ══ ALERTS ══ */
.alert-info { background: rgba(21,101,192,0.07); border: 1px solid rgba(21,101,192,0.25); border-left: 4px solid var(--blue); border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.8rem; color: var(--navy); margin-top: 1rem; }
.alert-warn { background: rgba(255,179,0,0.08); border: 1px solid rgba(255,179,0,0.3); border-left: 4px solid var(--amber); border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.8rem; color: #6D4C00; margin-top: 0.8rem; }

/* ══ INVEST TABLE ══ */
.inv-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.inv-table th { background: var(--offwhite); padding: 0.55rem 0.8rem; text-align: left; font-weight: 700; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--g600); border-bottom: 2px solid var(--g100); }
.inv-table td { padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--g100); color: var(--text); }
.inv-table tr:last-child td { border-bottom: none; }
.inv-table tr:hover td { background: rgba(0,188,212,0.04); }
.badge-up   { background: rgba(0,200,83,0.12); color: #00A844; border-radius: 4px; padding: 2px 7px; font-weight: 700; font-size: 0.7rem; }
.badge-down { background: rgba(255,82,82,0.12); color: #D32F2F; border-radius: 4px; padding: 2px 7px; font-weight: 700; font-size: 0.7rem; }

/* ══ SIDEBAR CUSTOM ELEMENTS ══ */
.sb-logo { font-family: 'DM Serif Display', serif; font-size: 1.55rem; font-weight: 800; color: var(--white) !important; letter-spacing: -0.02em; padding: 1.4rem 1.2rem 0.1rem; }
.sb-logo span { color: var(--cyan) !important; }
.sb-tag  { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: #4A7A9B !important; padding: 0 1.2rem 1.2rem; }
.sb-sec  { border-top: 1px solid rgba(0,188,212,0.15); margin: 0.8rem 0 0.2rem; padding: 0.6rem 0 0; }
.sb-sec-lbl { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--cyan) !important; margin-bottom: 0.6rem; }
.sb-stat { background: rgba(0,188,212,0.08); border: 1px solid rgba(0,188,212,0.2); border-radius: 10px; padding: 0.8rem; margin-bottom: 0.8rem; }
.sb-stat-val { font-size: 1.4rem; font-weight: 800; color: var(--white) !important; font-family: 'DM Serif Display', serif; }
.sb-stat-lbl { font-size: 0.65rem; color: #4A7A9B !important; letter-spacing: 0.08em; }

/* ══ GALLERY ══ */
.gallery-caption { font-size: 0.75rem; font-weight: 600; color: var(--g600); text-align: center; margin-top: 0.4rem; }
.gallery-tag { display: inline-block; background: var(--navy); color: var(--cyan) !important; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; border-radius: 4px; padding: 2px 8px; margin-bottom: 0.3rem; }

/* ══ DIVIDER / FOOTER ══ */
.divider { height: 1px; background: var(--g100); margin: 2rem 0; }
.footer { text-align: center; padding: 2rem 0 1rem; color: var(--g300); font-size: 0.72rem; letter-spacing: 0.12em; border-top: 1px solid var(--g100); margin-top: 2rem; }
.footer span { color: var(--cyan); }

/* ══ STREAMLIT METRIC OVERRIDE ══ */
[data-testid="stMetric"] { background: var(--offwhite); border-radius: 10px; padding: 0.8rem 1rem; border: 1px solid var(--g100); }
[data-testid="stMetricLabel"] { font-size: 0.7rem !important; font-weight: 600 !important; color: var(--g600) !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 800 !important; color: var(--navy) !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME  (clean white, navy text)
# ════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor":   "#F8FAFC",
    "axes.edgecolor":   "#E3EAF2",
    "axes.labelcolor":  "#546E7A",
    "axes.titlecolor":  "#1A2744",
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.labelsize":   8,
    "xtick.color":      "#78909C",
    "ytick.color":      "#78909C",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "grid.color":       "#E3EAF2",
    "grid.linestyle":   "-",
    "grid.alpha":       0.8,
    "text.color":       "#1A2744",
    "font.family":      "DejaVu Sans",
    "figure.dpi":       120,
})

C_NAVY  = "#0D1B2A"
C_CYAN  = "#00BCD4"
C_BLUE  = "#1565C0"
C_AMBER = "#FFB300"
C_GREEN = "#00C853"
C_RED   = "#FF5252"
C_GRAY  = "#B0BEC5"
C_PRP   = "#7C4DFF"

# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-logo">Estate<span>IQ</span> <small style="font-size:0.5em;opacity:0.6">PRO</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">AI-Powered Valuation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-stat"><div class="sb-stat-val">98.4%</div><div class="sb-stat-lbl">Model Accuracy (R² Score)</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-sec-lbl">📐 Size & Structure</div></div>', unsafe_allow_html=True)
    area      = st.number_input("Area (sqft)",      500, 10000, 2000, step=50)
    bedrooms  = st.number_input("Bedrooms",           1,    10,    3)
    bathrooms = st.number_input("Bathrooms",          1,     5,    2)
    stories   = st.number_input("Stories",            1,     4,    2)
    parking   = st.number_input("Parking Spaces",     0,     5,    1)

    st.markdown('<div class="sb-sec"><div class="sb-sec-lbl">🏡 Amenities</div></div>', unsafe_allow_html=True)
    mainroad        = st.selectbox("Main Road Access",  ["Yes","No"])
    guestroom       = st.selectbox("Guest Room",        ["Yes","No"])
    basement        = st.selectbox("Basement",          ["Yes","No"])
    airconditioning = st.selectbox("Air Conditioning",  ["Yes","No"])
    prefarea        = st.selectbox("Preferred Area",    ["Yes","No"])

    st.markdown('<div class="sb-sec"><div class="sb-sec-lbl">🛋 Interior</div></div>', unsafe_allow_html=True)
    furnishing = st.selectbox("Furnishing Status", ["Furnished","Semi-Furnished","Unfurnished"])

    st.markdown('<div class="sb-sec"><div class="sb-sec-lbl">⚙️ Analysis Settings</div></div>', unsafe_allow_html=True)
    roi_years    = st.slider("ROI Projection (Years)", 1, 20, 10)
    appreciation = st.slider("Annual Appreciation %",  2, 15,  7)

    predict_clicked = st.button("🔍  Analyze & Estimate Price", key="predict_btn")

    st.markdown("""
    <div class="alert-warn" style="margin-top:1rem;">
        ⚡ Results are ML estimates. Consult a certified appraiser for legal valuations.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  BUILD INPUT DATAFRAME
# ════════════════════════════════════════════════════════════════════
input_data = pd.DataFrame({
    "area":                           [area],
    "bedrooms":                       [bedrooms],
    "bathrooms":                      [bathrooms],
    "stories":                        [stories],
    "parking":                        [parking],
    "mainroad_yes":                   [1 if mainroad        == "Yes" else 0],
    "guestroom_yes":                  [1 if guestroom       == "Yes" else 0],
    "basement_yes":                   [1 if basement        == "Yes" else 0],
    "hotwaterheating_yes":            [0],
    "airconditioning_yes":            [1 if airconditioning == "Yes" else 0],
    "prefarea_yes":                   [1 if prefarea        == "Yes" else 0],
    "furnishingstatus_semi-furnished":[1 if furnishing      == "Semi-Furnished" else 0],
    "furnishingstatus_unfurnished":   [1 if furnishing      == "Unfurnished" else 0],
})

# ════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Machine Learning · Real Estate Intelligence</div>
    <div class="hero-title">AI House Price<br>Estimator</div>
    <div class="hero-desc">
        Configure your property in the sidebar for an instant data-driven valuation —
        complete with analytics, ROI projection, market positioning and more.
    </div>
    <div class="hero-badge">MODEL ACTIVE &nbsp;·&nbsp; 13 FEATURES &nbsp;·&nbsp; REAL-TIME</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  LIVE KPI STRIP  (updates on every sidebar change)
# ════════════════════════════════════════════════════════════════════
price_live    = model.predict(input_data)[0]
per_sqft      = price_live / area
amenity_score = sum([
    mainroad == "Yes", guestroom == "Yes", basement == "Yes",
    airconditioning == "Yes", prefarea == "Yes",
    furnishing == "Furnished"
])
roi_total = ((1 + appreciation/100)**roi_years - 1) * 100

st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-box">
    <div class="kpi-label">Live Estimate</div>
    <div class="kpi-val">${price_live:,.0f}</div>
    <div class="kpi-sub">Current configuration</div>
  </div>
  <div class="kpi-box" style="border-top-color:#FFB300;">
    <div class="kpi-label">Price / sqft</div>
    <div class="kpi-val">${per_sqft:,.0f}</div>
    <div class="kpi-sub">{area:,} sqft total</div>
  </div>
  <div class="kpi-box" style="border-top-color:#00C853;">
    <div class="kpi-label">Amenity Score</div>
    <div class="kpi-val">{amenity_score}/6</div>
    <div class="kpi-sub">Active features</div>
  </div>
  <div class="kpi-box" style="border-top-color:#7C4DFF;">
    <div class="kpi-label">ROI @ {roi_years}yr</div>
    <div class="kpi-val">{roi_total:.0f}%</div>
    <div class="kpi-sub">{appreciation}% p.a.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  FULL PREDICTION (on button click)
# ════════════════════════════════════════════════════════════════════
if predict_clicked:
    with st.spinner("Analyzing property data…"):
        time.sleep(0.5)
    price = model.predict(input_data)[0]

    # ── Block 1: Price Card + Snapshot ───────────────────────────
    col_l, col_r = st.columns([5, 7], gap="large")

    with col_l:
        conf_pct = min(95, 70 + amenity_score * 4)
        st.markdown(f"""
        <div class="card-dark">
          <div class="price-hero">
            <div class="price-eyebrow">AI Estimated Value</div>
            <div class="price-main"><span>$</span>{price:,.0f}</div>
            <div class="price-range">Range: ${price*0.92:,.0f} – ${price*1.08:,.0f}</div>
            <div class="price-divider"></div>
            <div style="text-align:left;">
              <div class="conf-label">
                <span>Confidence Score</span>
                <span style="color:#00BCD4;">{conf_pct}%</span>
              </div>
              <div class="conf-bar"><div class="conf-fill" style="width:{conf_pct}%;"></div></div>
            </div>
            <div style="margin-top:1.2rem;">
              <div style="font-size:0.7rem;color:#4A7A9B;letter-spacing:0.08em;">PRICE PER SQFT</div>
              <div style="font-size:1.6rem;font-weight:800;color:#F0F4F8;font-family:'DM Serif Display',serif;">${price/area:,.0f}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-title">Property Snapshot</div>', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("🏠 Area",     f"{area:,} sqft")
        m2.metric("🛏 Bedrooms",  bedrooms)
        m3.metric("🚿 Bathrooms", bathrooms)
        m4,m5,m6 = st.columns(3)
        m4.metric("🏗 Stories",   stories)
        m5.metric("🚗 Parking",   parking)
        m6.metric("🏆 Amenities", f"{amenity_score}/6")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#546E7A;margin-bottom:0.5rem;">Feature Status</div>', unsafe_allow_html=True)
        chips_map = {
            "Main Road": mainroad=="Yes", "Guest Room": guestroom=="Yes",
            "Basement":  basement=="Yes", "Air Conditioning": airconditioning=="Yes",
            "Pref. Area":prefarea=="Yes", "Furnished": furnishing=="Furnished",
            "Semi-Furnished": furnishing=="Semi-Furnished", "Unfurnished": furnishing=="Unfurnished",
        }
        ch_html = '<div class="chip-wrap">'
        for lbl, on in chips_map.items():
            cls = "chip chip-on" if on else "chip chip-off"
            ch_html += f'<span class="{cls}"><span class="chip-dot"></span>{lbl}</span>'
        ch_html += '</div>'
        st.markdown(ch_html, unsafe_allow_html=True)
        st.markdown('<div class="alert-info">🤖 Prediction is based on a trained ML regression model. Actual prices may vary.</div>', unsafe_allow_html=True)

    # ── Block 2: Three Charts ─────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📊 Analytics & Insights</div>', unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3, gap="medium")

    with ch1:
        years  = np.arange(2020, 2026)
        growth = np.array([0.58,0.68,0.78,0.88,0.95,1.00])
        hist_p = price * growth
        fig1, ax1 = plt.subplots(figsize=(5, 3.4))
        ax1.fill_between(years, hist_p, color=C_CYAN, alpha=0.15)
        ax1.plot(years, hist_p, color=C_CYAN, lw=2.5,
                 marker="o", ms=5, mfc="white", mec=C_CYAN, mew=2)
        ax1.annotate(f"${price:,.0f}", xy=(2025, price),
                     xytext=(-55,14), textcoords="offset points",
                     fontsize=8, color=C_CYAN, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C_CYAN, lw=1.2))
        ax1.set_title("Historical Price Estimate", pad=8)
        ax1.set_ylabel("Price ($)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
        ax1.grid(True, axis="y"); ax1.spines[["top","right"]].set_visible(False)
        fig1.tight_layout(); st.pyplot(fig1, use_container_width=True)

    with ch2:
        feat_names = ["Area","Bedrooms","Bathrooms","Stories","Parking"]
        feat_raw   = [area/10, bedrooms*300, bathrooms*200, stories*150, parking*100]
        feat_norm  = np.array(feat_raw) / max(feat_raw)
        bar_colors = [C_CYAN, C_BLUE, C_PRP, C_GREEN, C_AMBER]
        fig2, ax2 = plt.subplots(figsize=(5, 3.4))
        bars = ax2.barh(feat_names, feat_norm, color=bar_colors, height=0.52, edgecolor="none")
        for bar, val in zip(bars, feat_norm):
            ax2.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                     f"{val:.0%}", va="center", fontsize=8.5, color=C_NAVY, fontweight="600")
        ax2.set_xlim(0, 1.3); ax2.set_title("Feature Contribution", pad=8)
        ax2.spines[["top","right","bottom","left"]].set_visible(False)
        ax2.xaxis.set_visible(False); ax2.grid(False)
        fig2.tight_layout(); st.pyplot(fig2, use_container_width=True)

    with ch3:
        proj_y = np.arange(0, roi_years+1)
        proj_v = price * ((1+appreciation/100)**proj_y)
        fig3, ax3 = plt.subplots(figsize=(5, 3.4))
        ax3.fill_between(proj_y, proj_v, color=C_GREEN, alpha=0.12)
        ax3.plot(proj_y, proj_v, color=C_GREEN, lw=2.5,
                 marker="o", ms=4, mfc="white", mec=C_GREEN, mew=2,
                 markevery=max(1, roi_years//5))
        gain = proj_v[-1] - price
        ax3.annotate(f"+${gain:,.0f}", xy=(roi_years, proj_v[-1]),
                     xytext=(-60,10), textcoords="offset points",
                     fontsize=8, color=C_GREEN, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2))
        ax3.set_title(f"ROI Projection ({roi_years}yr @ {appreciation}%)", pad=8)
        ax3.set_xlabel("Year"); ax3.set_ylabel("Value ($)")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
        ax3.grid(True, axis="y"); ax3.spines[["top","right"]].set_visible(False)
        fig3.tight_layout(); st.pyplot(fig3, use_container_width=True)

    # ── Block 3: Market Bracket + Investment Table ────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    mk1, mk2 = st.columns([3,2], gap="large")

    with mk1:
        st.markdown('<div class="sec-title">📍 Market Positioning</div>', unsafe_allow_html=True)
        brackets = [
            ("Budget",    0,          price*0.6,  "#546E7A"),
            ("Mid-Range", price*0.6,  price*0.85, C_BLUE),
            ("Your Home", price*0.85, price*1.15, C_CYAN),
            ("Premium",   price*1.15, price*1.5,  C_AMBER),
            ("Luxury",    price*1.5,  price*2.0,  C_PRP),
        ]
        fig4, ax4 = plt.subplots(figsize=(7, 2.2))
        for label, lo, hi, col in brackets:
            w = hi - lo
            ax4.barh(0, w, left=lo, color=col, height=0.46,
                     edgecolor="white", linewidth=1.5)
            ax4.text(lo+w/2, 0, label, ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white")
        ax4.axvline(price, color=C_NAVY, lw=2.5, linestyle="--", zorder=5)
        ax4.text(price, 0.28, f"▼ ${price:,.0f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=C_NAVY)
        ax4.set_xlim(0, price*2.0); ax4.axis("off")
        ax4.set_facecolor("#F8FAFC"); fig4.patch.set_facecolor("#F8FAFC")
        fig4.tight_layout(); st.pyplot(fig4, use_container_width=True)

    with mk2:
        st.markdown('<div class="sec-title">💼 Investment Summary</div>', unsafe_allow_html=True)
        future_val   = price * ((1+appreciation/100)**roi_years)
        monthly_rent = price * 0.004
        gross_yield  = (monthly_rent*12)/price*100
        rows = [
            ("Current Value",         f"${price:,.0f}",       ""),
            ("Low Estimate (−8%)",    f"${price*0.92:,.0f}",  ""),
            ("High Estimate (+8%)",   f"${price*1.08:,.0f}",  ""),
            (f"Value in {roi_years}yr", f"${future_val:,.0f}",
             f'<span class="badge-up">+{((future_val/price)-1)*100:.0f}%</span>'),
            ("Est. Monthly Rent",     f"${monthly_rent:,.0f}", ""),
            ("Gross Yield",           f"{gross_yield:.2f}%",   ""),
        ]
        table = '<table class="inv-table"><thead><tr><th>Metric</th><th>Value</th><th></th></tr></thead><tbody>'
        for r in rows:
            table += f'<tr><td>{r[0]}</td><td style="font-weight:700">{r[1]}</td><td>{r[2]}</td></tr>'
        table += '</tbody></table>'
        st.markdown(f'<div class="card">{table}</div>', unsafe_allow_html=True)

    # ── Block 4: Radar Chart + Price-vs-Area Curve ────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    ra1, ra2 = st.columns(2, gap="large")

    with ra1:
        st.markdown('<div class="sec-title">🕸 Property Radar</div>', unsafe_allow_html=True)
        cats   = ['Area','Rooms','Amenities','Location','Condition']
        scores = [
            min(area/10000, 1.0),
            (bedrooms+bathrooms)/15,
            amenity_score/6,
            1.0 if prefarea=="Yes" else (0.7 if mainroad=="Yes" else 0.4),
            1.0 if furnishing=="Furnished" else (0.6 if furnishing=="Semi-Furnished" else 0.3),
        ]
        N = len(cats)
        angles   = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        scores_c = scores + [scores[0]]
        angles_c = angles + [angles[0]]

        fig5, ax5 = plt.subplots(figsize=(4.5,4.5), subplot_kw=dict(polar=True))
        ax5.set_facecolor("#F8FAFC"); fig5.patch.set_facecolor("#F8FAFC")
        ax5.plot(angles_c, scores_c, color=C_CYAN, lw=2.5)
        ax5.fill(angles_c, scores_c, color=C_CYAN, alpha=0.15)
        ax5.set_xticks(angles)
        ax5.set_xticklabels(cats, size=9, fontweight="bold", color=C_NAVY)
        ax5.set_yticks([0.25,0.5,0.75,1.0])
        ax5.set_yticklabels(["25%","50%","75%","100%"], size=7, color=C_GRAY)
        ax5.grid(color="#E3EAF2", linewidth=1.0)
        ax5.spines["polar"].set_color("#E3EAF2")
        ax5.set_title("Score Breakdown", pad=18, fontsize=10, fontweight="bold", color=C_NAVY)
        fig5.tight_layout(); st.pyplot(fig5, use_container_width=True)

    with ra2:
        st.markdown('<div class="sec-title">📈 Price vs Area Sensitivity</div>', unsafe_allow_html=True)
        areas_r  = np.linspace(500, 10000, 80)
        prices_r = []
        for a in areas_r:
            tmp = input_data.copy(); tmp["area"] = a
            prices_r.append(model.predict(tmp)[0])
        prices_r = np.array(prices_r)

        fig6, ax6 = plt.subplots(figsize=(5, 4.5))
        ax6.plot(areas_r, prices_r, color=C_BLUE, lw=2.5)
        ax6.fill_between(areas_r, prices_r, alpha=0.08, color=C_BLUE)
        ax6.axvline(area,  color=C_AMBER, lw=1.8, linestyle="--", label=f"Your area: {area:,}")
        ax6.axhline(price, color=C_RED,   lw=1.8, linestyle="--", label=f"Your price: ${price:,.0f}")
        ax6.scatter([area],[price], color=C_CYAN, s=80, zorder=5, label="Your Property")
        ax6.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax6.set_title("Price Sensitivity to Area", pad=8)
        ax6.set_xlabel("Area (sqft)"); ax6.set_ylabel("Estimated Price ($)")
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
        ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x/1000:.1f}K"))
        ax6.grid(True, alpha=0.6); ax6.spines[["top","right"]].set_visible(False)
        fig6.tight_layout(); st.pyplot(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PROPERTY SHOWCASE — Filterable Listing Cards + View Details Modal
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🏡 Property Showcase</div>', unsafe_allow_html=True)

st.markdown("""
<style>
/* ══════════════════════════════════════════
   GLOBAL BUTTON RESET — professional styles
══════════════════════════════════════════ */

/* Default: all buttons = white bg, dark navy text */
div[data-testid="stButton"] > button,
div[data-testid="stButton"] > button:focus,
div[data-testid="stButton"] > button:focus:not(:active) {
    background-color: #FFFFFF !important;
    background-image: none !important;
    color: #1A2744 !important;
    border: 1.5px solid #CBD8E8 !important;
    border-radius: 50px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.42rem 1rem !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 4px rgba(13,27,42,0.07) !important;
}
div[data-testid="stButton"] > button:hover {
    background-color: #F0F8FF !important;
    border-color: #00BCD4 !important;
    color: #006064 !important;
    box-shadow: 0 3px 12px rgba(0,188,212,0.18) !important;
    transform: translateY(-1px) !important;
}

/* Active filter: navy bg + cyan text */
.flt-active div[data-testid="stButton"] > button,
.flt-active div[data-testid="stButton"] > button:focus,
.flt-active div[data-testid="stButton"] > button:focus:not(:active) {
    background-color: #0D1B2A !important;
    background-image: none !important;
    color: #00E5FF !important;
    border: 2px solid #00BCD4 !important;
    border-radius: 50px !important;
    box-shadow: 0 4px 16px rgba(0,188,212,0.30) !important;
    font-weight: 700 !important;
}
.flt-active div[data-testid="stButton"] > button:hover {
    background-color: #132338 !important;
    color: #00E5FF !important;
    border-color: #00E5FF !important;
    transform: translateY(-1px) !important;
}

/* View Details button: cyan gradient */
.view-btn div[data-testid="stButton"] > button,
.view-btn div[data-testid="stButton"] > button:focus,
.view-btn div[data-testid="stButton"] > button:focus:not(:active) {
    background-color: #0077B6 !important;
    background-image: linear-gradient(135deg, #0077B6, #00BCD4) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 16px rgba(0,188,212,0.35) !important;
    padding: 0.55rem 1rem !important;
}
.view-btn div[data-testid="stButton"] > button:hover {
    background-color: #005F8F !important;
    background-image: linear-gradient(135deg, #005F8F, #0097A7) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,188,212,0.45) !important;
}

/* Close button: red outline */
.close-btn div[data-testid="stButton"] > button,
.close-btn div[data-testid="stButton"] > button:focus {
    background-color: #FFF5F5 !important;
    background-image: none !important;
    border: 1.5px solid #FFCDD2 !important;
    color: #C62828 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.76rem !important;
}
.close-btn div[data-testid="stButton"] > button:hover {
    background-color: #FFEBEE !important;
    border-color: #FF5252 !important;
    color: #B71C1C !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Sidebar predict button: cyan gradient */
[data-testid="stSidebar"] div[data-testid="stButton"] > button,
[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus,
[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus:not(:active) {
    background-color: #0077B6 !important;
    background-image: linear-gradient(135deg, #0077B6, #00BCD4) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 18px rgba(0,188,212,0.45) !important;
    padding: 0.75rem 1rem !important;
    margin-top: 8px !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background-color: #005F8F !important;
    background-image: linear-gradient(135deg, #005F8F, #0097A7) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 28px rgba(0,188,212,0.6) !important;
}


/* ── Listing card ── */
.lcard {
    background: #FFFFFF;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(13,27,42,0.09);
    border: 1px solid #E3EAF2;
    transition: transform 0.22s, box-shadow 0.22s;
    margin-bottom: 1rem;
}
.lcard:hover { transform: translateY(-4px); box-shadow: 0 14px 40px rgba(13,27,42,0.15); }
.lcard-imgwrap { position: relative; height: 200px; overflow: hidden; }
.lcard-imgwrap img { width:100%; height:200px; object-fit:cover; display:block; }
.lcard-badge {
    position: absolute; top: 12px; left: 12px;
    background: #0D1B2A; color: #00BCD4 !important;
    font-size: 0.58rem; font-weight: 800; letter-spacing: 0.16em;
    text-transform: uppercase; border-radius: 6px; padding: 4px 10px;
}
.lcard-price {
    position: absolute; bottom: 12px; right: 12px;
    background: rgba(13,27,42,0.90); color: #FFFFFF !important;
    font-family: 'DM Serif Display', serif; font-size: 1.05rem; font-weight: 700;
    border-radius: 8px; padding: 5px 13px; backdrop-filter: blur(6px);
}
.lcard-body { padding: 1rem 1.2rem 1.1rem; }
.lcard-title { font-size: 1rem; font-weight: 700; color: #0D1B2A; margin-bottom: 0.2rem; }
.lcard-loc   { font-size: 0.72rem; color: #546E7A; margin-bottom: 0.65rem; }
.lcard-stats { display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom: 0.55rem; }
.lcard-stat  { background: #F0F4F8; border-radius: 6px; padding: 3px 9px; font-size: 0.67rem; font-weight: 600; color: #37474F; }
.lcard-tag   { display:inline-block; background: rgba(0,188,212,0.10); border: 1px solid rgba(0,188,212,0.4); color: #0097A7; border-radius: 6px; padding: 3px 9px; font-size: 0.67rem; font-weight: 700; margin-bottom: 0.75rem; }

/* ── View Details modal panel ── */
.detail-panel {
    background: linear-gradient(135deg, #0D1B2A, #1A3050);
    border-radius: 16px; padding: 1.4rem 1.6rem;
    border: 1px solid rgba(0,188,212,0.25);
    box-shadow: 0 12px 48px rgba(13,27,42,0.25);
    color: #F0F4F8;
}
.detail-panel h3 { font-family:'DM Serif Display',serif; font-size:1.3rem; color:#FFFFFF; margin:0 0 0.3rem; }
.detail-panel .dp-price { font-family:'DM Serif Display',serif; font-size:2rem; color:#00BCD4; font-weight:700; margin:0.2rem 0 0.8rem; }
.detail-row { display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.8rem; }
.detail-pill { background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); border-radius:8px; padding:0.4rem 0.8rem; font-size:0.76rem; font-weight:600; color:#CBD8E8; }
.detail-feature { display:flex; align-items:center; gap:6px; font-size:0.78rem; color:#8BAEC8; margin-bottom:0.35rem; }
.detail-feature span.dot { width:7px; height:7px; border-radius:50%; background:#00BCD4; display:inline-block; }
.detail-section-lbl { font-size:0.65rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:#4A7A9B; margin:0.8rem 0 0.4rem; }
.close-btn-wrap button {
    background: rgba(255,82,82,0.15) !important;
    border: 1px solid rgba(255,82,82,0.4) !important;
    color: #FF5252 !important;
    border-radius: 8px !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
}

/* ── Neighborhood scores ── */
.nb-card2 { background:#FFFFFF; border:1px solid #E3EAF2; border-radius:12px; padding:0.75rem 1rem; display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem; box-shadow:0 2px 8px rgba(13,27,42,0.06); }
.nb-icon2 { font-size:1.3rem; }
.nb-info2 { flex:1; }
.nb-name2 { font-size:0.75rem; font-weight:700; color:#0D1B2A; margin-bottom:0.22rem; }
.nb-barw  { height:5px; background:#E3EAF2; border-radius:50px; overflow:hidden; }
.nb-barf  { height:100%; border-radius:50px; }
.nb-sc    { font-size:0.72rem; font-weight:800; min-width:26px; text-align:right; }

/* ── Comparable pins ── */
.comp2 { display:flex; align-items:center; gap:0.6rem; padding:0.48rem 0.75rem; background:#F0F4F8; border-radius:10px; margin-bottom:0.4rem; }
.comp2-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.comp2-lbl { font-size:0.75rem; font-weight:600; color:#0D1B2A; flex:1; }
.comp2-p   { font-size:0.75rem; font-weight:700; color:#546E7A; }
</style>
""", unsafe_allow_html=True)

# ── Listing data ──────────────────────────────────────────────────────
LISTINGS_ALL = [
    {
        "img":"https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=700",
        "badge":"MODERN","cat":"Modern","title":"Skyline Residency",
        "loc":"Kalawad Road, Rajkot","price":"₹85 L","price_num":8500000,
        "beds":3,"baths":2,"sqft":1850,"tag":"Ready to Move",
        "desc":"A sleek contemporary home with open floor plan, Italian marble flooring, modular kitchen, and landscaped garden. Located minutes from Kalawad Road with excellent connectivity.",
        "features":["Modular Kitchen","Solar Panels","CCTV Security","Garden","2 Car Parking","24/7 Water Supply"],
        "emi":"₹54,200/mo","possession":"Ready","age":"2 years","floors":"G+2",
    },
    {
        "img":"https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=700",
        "badge":"LUXURY","cat":"Luxury","title":"Imperial Heights",
        "loc":"Race Course, Rajkot","price":"₹1.8 Cr","price_num":18000000,
        "beds":4,"baths":3,"sqft":3200,"tag":"Premium",
        "desc":"Ultra-luxury villa in the heart of Race Course area. Features a private pool, home theatre, smart home automation, imported fittings, and a rooftop lounge with panoramic city views.",
        "features":["Private Pool","Home Theatre","Smart Home","Imported Fittings","Rooftop Lounge","Club Membership"],
        "emi":"₹1.14 L/mo","possession":"Ready","age":"1 year","floors":"G+3",
    },
    {
        "img":"https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=700",
        "badge":"FAMILY","cat":"Family","title":"Green Valley Home",
        "loc":"Mavdi, Rajkot","price":"₹62 L","price_num":6200000,
        "beds":3,"baths":2,"sqft":1500,"tag":"New Launch",
        "desc":"Perfectly designed for families in a quiet residential colony. Features a children's play area, community hall, vastu-compliant design, and excellent school proximity.",
        "features":["Children's Play Area","Community Hall","Vastu Design","School Nearby","Park View","Power Backup"],
        "emi":"₹39,500/mo","possession":"Dec 2025","age":"New","floors":"G+1",
    },
    {
        "img":"https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=700",
        "badge":"VILLA","cat":"Luxury","title":"Casa Del Sol",
        "loc":"150 Ft Ring Road, Rajkot","price":"₹2.4 Cr","price_num":24000000,
        "beds":5,"baths":4,"sqft":4200,"tag":"Gated Community",
        "desc":"Sprawling bungalow in a premium gated community on 150 Ft Ring Road. Includes private terrace, jacuzzi, landscaped 2000 sqft garden, servant quarters, and 3-car garage.",
        "features":["Jacuzzi","3-Car Garage","Servant Quarters","Landscaped Garden","Private Terrace","24/7 Security"],
        "emi":"₹1.52 L/mo","possession":"Ready","age":"3 years","floors":"G+2",
    },
    {
        "img":"https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=700",
        "badge":"APARTMENT","cat":"Apartment","title":"Metro View Flat",
        "loc":"Yagnik Road, Rajkot","price":"₹45 L","price_num":4500000,
        "beds":2,"baths":1,"sqft":980,"tag":"Ready to Move",
        "desc":"Well-maintained 2BHK apartment in prime Yagnik Road location. Close to markets, hospitals, and bus stand. Ideal for young professionals and small families.",
        "features":["Lift","Covered Parking","CCTV","Gym","City View","Maintenance Staff"],
        "emi":"₹28,600/mo","possession":"Ready","age":"4 years","floors":"4th Floor",
    },
    {
        "img":"https://images.unsplash.com/photo-1583608205776-bfd35f0d9f83?w=700",
        "badge":"BUNGALOW","cat":"Family","title":"The Heritage House",
        "loc":"Kotecha Chowk, Rajkot","price":"₹1.1 Cr","price_num":11000000,
        "beds":4,"baths":3,"sqft":2800,"tag":"Resale",
        "desc":"A classic Rajkot bungalow with heritage charm and modern upgrades. Solid RCC construction, large airy rooms, covered verandah, and a fully equipped modern kitchen.",
        "features":["Corner Plot","Large Verandah","RCC Construction","Modern Kitchen","Storage Room","Well Ventilated"],
        "emi":"₹69,900/mo","possession":"Ready","age":"12 years","floors":"G+1",
    },
]

FILTER_CATS = {"All": LISTINGS_ALL, "Modern": [], "Luxury": [], "Family": [], "Apartment": []}
for l in LISTINGS_ALL:
    FILTER_CATS[l["cat"]].append(l)

# ── Session state ─────────────────────────────────────────────────────
if "gallery_filter"   not in st.session_state: st.session_state.gallery_filter   = "All"
if "selected_listing" not in st.session_state: st.session_state.selected_listing = None

# ── Filter bar ────────────────────────────────────────────────────────
filter_labels = ["All","Modern","Luxury","Family","Apartment"]
f_cols = st.columns(len(filter_labels), gap="small")
for i, cat in enumerate(filter_labels):
    is_active = st.session_state.gallery_filter == cat
    css_class = "filter-btn-active" if is_active else "filter-btn-inactive"
    icon = {"All":"🏘","Modern":"🏠","Luxury":"💎","Family":"👨‍👩‍👧","Apartment":"🏢"}.get(cat,"")
    with f_cols[i]:
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        if st.button(f"{icon} {cat}", key=f"flt_{cat}", use_container_width=True):
            st.session_state.gallery_filter   = cat
            st.session_state.selected_listing = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Show detail panel if a listing is selected ───────────────────────
if st.session_state.selected_listing is not None:
    l = st.session_state.selected_listing
    st.markdown(f"""
    <div class="detail-panel">
      <h3>{l['title']}</h3>
      <div class="dp-price">{l['price']}</div>
      <div class="detail-row">
        <span class="detail-pill">🛏 {l['beds']} Beds</span>
        <span class="detail-pill">🚿 {l['baths']} Baths</span>
        <span class="detail-pill">📐 {l['sqft']:,} sqft</span>
        <span class="detail-pill">📍 {l['loc']}</span>
        <span class="detail-pill">🏗 {l['floors']}</span>
        <span class="detail-pill">🏷 {l['tag']}</span>
      </div>
      <div class="detail-section-lbl">About this property</div>
      <p style="font-size:0.84rem;color:#8BAEC8;line-height:1.65;margin:0 0 0.8rem;">{l['desc']}</p>
      <div class="detail-section-lbl">Key Features</div>
      <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.9rem;">
        {"".join(f'<span class="detail-pill">{f}</span>' for f in l["features"])}
      </div>
      <div class="detail-row" style="margin-bottom:0;">
        <span class="detail-pill">💰 EMI {l['emi']}</span>
        <span class="detail-pill">📅 Possession: {l['possession']}</span>
        <span class="detail-pill">🏚 Age: {l['age']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_close, _ = st.columns([2, 8])
    with col_close:
        st.markdown('<div class="close-btn">', unsafe_allow_html=True)
        if st.button("✕  Close Details", key="close_detail"):
            st.session_state.selected_listing = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Render cards ──────────────────────────────────────────────────────
active_listings = FILTER_CATS.get(st.session_state.gallery_filter, LISTINGS_ALL)
for row_start in range(0, len(active_listings), 3):
    row_items = active_listings[row_start:row_start+3]
    cols = st.columns(len(row_items), gap="medium")
    for col, l in zip(cols, row_items):
        with col:
            st.markdown(f"""
            <div class="lcard">
              <div class="lcard-imgwrap">
                <img src="{l['img']}" alt="{l['title']}"/>
                <div class="lcard-badge">{l['badge']}</div>
                <div class="lcard-price">{l['price']}</div>
              </div>
              <div class="lcard-body">
                <div class="lcard-title">{l['title']}</div>
                <div class="lcard-loc">📍 {l['loc']}</div>
                <div class="lcard-stats">
                  <span class="lcard-stat">🛏 {l['beds']} Beds</span>
                  <span class="lcard-stat">🚿 {l['baths']} Baths</span>
                  <span class="lcard-stat">📐 {l['sqft']:,} sqft</span>
                </div>
                <div class="lcard-tag">{l['tag']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="view-btn">', unsafe_allow_html=True)
            if st.button("View Details →", key=f"view_{l['title'].replace(' ','_')}", use_container_width=True):
                st.session_state.selected_listing = l
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  LOCATION INTELLIGENCE — pydeck Map + Location Search + Nb Scores
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📍 Location Intelligence</div>', unsafe_allow_html=True)

# ── Location search ───────────────────────────────────────────────────
CITY_LOCATIONS = {
    "Rajkot, Gujarat":       (22.3039, 70.8022),
    "Ahmedabad, Gujarat":    (23.0225, 72.5714),
    "Surat, Gujarat":        (21.1702, 72.8311),
    "Vadodara, Gujarat":     (22.3072, 73.1812),
    "Gandhinagar, Gujarat":  (23.2156, 72.6369),
    "Bhavnagar, Gujarat":    (21.7645, 72.1519),
    "Jamnagar, Gujarat":     (22.4707, 70.0577),
    "Junagadh, Gujarat":     (21.5222, 70.4579),
    "Mumbai, Maharashtra":   (19.0760, 72.8777),
    "Pune, Maharashtra":     (18.5204, 73.8567),
    "Delhi":                 (28.6139, 77.2090),
    "Bangalore, Karnataka":  (12.9716, 77.5946),
    "Chennai, Tamil Nadu":   (13.0827, 80.2707),
    "Hyderabad, Telangana":  (17.3850, 78.4867),
    "Kolkata, West Bengal":  (22.5726, 88.3639),
    "Jaipur, Rajasthan":     (26.9124, 75.7873),
}

# Per-city neighborhood scores  [schools, healthcare, markets, transport, green, safety]
CITY_SCORES = {
    "Rajkot, Gujarat":       [88, 75, 92, 70, 60, 82],
    "Ahmedabad, Gujarat":    [85, 80, 94, 78, 55, 79],
    "Surat, Gujarat":        [82, 78, 96, 75, 50, 77],
    "Vadodara, Gujarat":     [87, 82, 90, 72, 65, 83],
    "Gandhinagar, Gujarat":  [91, 85, 80, 74, 78, 90],
    "Bhavnagar, Gujarat":    [78, 70, 82, 62, 58, 75],
    "Jamnagar, Gujarat":     [75, 68, 79, 60, 55, 72],
    "Junagadh, Gujarat":     [72, 66, 76, 58, 62, 70],
    "Mumbai, Maharashtra":   [90, 88, 98, 95, 45, 72],
    "Pune, Maharashtra":     [92, 87, 93, 82, 68, 80],
    "Delhi":                 [88, 85, 97, 90, 42, 65],
    "Bangalore, Karnataka":  [94, 89, 95, 80, 70, 78],
    "Chennai, Tamil Nadu":   [89, 86, 91, 85, 58, 76],
    "Hyderabad, Telangana":  [91, 88, 93, 83, 62, 79],
    "Kolkata, West Bengal":  [86, 82, 94, 88, 55, 68],
    "Jaipur, Rajasthan":     [83, 76, 88, 72, 65, 80],
}


search_col, _ = st.columns([3, 5])
with search_col:
    selected_city = st.selectbox(
        "🔍 Search / Select Location",
        options=list(CITY_LOCATIONS.keys()),
        index=0,
    )

map_lat, map_lon = CITY_LOCATIONS[selected_city]

map_col, nb_col = st.columns([3, 2], gap="large")

with map_col:
    import pydeck as pdk

    # Comparable offsets relative to chosen city
    COMP_OFFSETS = [
        (0.008,  -0.008, "Imperial Heights",  "₹1.8 Cr", [220, 38, 38]),
        (-0.005,  0.011, "Green Valley Home", "₹62 L",   [46, 160, 67]),
        (0.018,   0.006, "Casa Del Sol",      "₹2.4 Cr", [106, 77, 255]),
        (-0.010, -0.013, "Metro View Flat",   "₹45 L",   [230, 81, 0]),
    ]
    AMENITY_OFFSETS = [
        (0.004,   0.008, "🏥 Hospital",   [220, 50, 50]),
        (-0.003, -0.006, "🏫 School",     [21, 101, 192]),
        (0.002,   0.016, "🛒 Market",     [0, 150, 100]),
        (-0.007,  0.004, "🚌 Bus Stand",  [100, 100, 200]),
    ]

    points = [{
        "lat": map_lat, "lon": map_lon,
        "name": f"🏠 Your Property  ${model.predict(input_data)[0]:,.0f}",
        "color": [13, 27, 42], "size": 320,
    }]
    for dlat, dlon, name, price_tag, col in COMP_OFFSETS:
        points.append({"lat": map_lat+dlat, "lon": map_lon+dlon,
                        "name": f"{name}  {price_tag}", "color": col, "size": 200})
    for dlat, dlon, name, col in AMENITY_OFFSETS:
        points.append({"lat": map_lat+dlat, "lon": map_lon+dlon,
                        "name": name, "color": col, "size": 150})

    points_df = pd.DataFrame(points)

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=points_df,
        get_position=["lon","lat"],
        get_fill_color="color",
        get_radius="size",
        pickable=True,
        opacity=0.9,
        stroked=True,
        get_line_color=[255,255,255],
        line_width_min_pixels=2,
    )
    # Cyan radius circle for your property
    circle_df = pd.DataFrame([{"lat": map_lat, "lon": map_lon, "radius": 800}])
    circle = pdk.Layer(
        "ScatterplotLayer",
        data=circle_df,
        get_position=["lon","lat"],
        get_radius="radius",
        get_fill_color=[0, 188, 212, 25],
        get_line_color=[0, 188, 212],
        stroked=True,
        line_width_min_pixels=2,
        pickable=False,
    )

    view = pdk.ViewState(latitude=map_lat, longitude=map_lon, zoom=12, pitch=30)
    deck = pdk.Deck(
        layers=[circle, scatter],
        initial_view_state=view,
        tooltip={"html": "<b>{name}</b>", "style": {"background":"#0D1B2A","color":"#F0F4F8","font-size":"13px","padding":"8px 12px","border-radius":"8px","border":"1px solid #00BCD4"}},
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )
    st.pydeck_chart(deck, use_container_width=True, height=440)

    # Legend
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.7rem;">
      <span style="display:flex;align-items:center;gap:5px;font-size:0.72rem;font-weight:600;color:#546E7A;"><span style="width:10px;height:10px;border-radius:50%;background:#0D1B2A;display:inline-block;"></span>Your Property</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:0.72rem;font-weight:600;color:#546E7A;"><span style="width:10px;height:10px;border-radius:50%;background:#DC2626;display:inline-block;"></span>Comparable</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:0.72rem;font-weight:600;color:#546E7A;"><span style="width:10px;height:10px;border-radius:50%;background:#1565C0;display:inline-block;"></span>Amenity</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:0.72rem;font-weight:600;color:#0097A7;"><span style="width:10px;height:10px;border-radius:50%;background:rgba(0,188,212,0.3);border:1px solid #00BCD4;display:inline-block;"></span>Preferred Zone (800m)</span>
    </div>
    """, unsafe_allow_html=True)

with nb_col:
    st.markdown(f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#546E7A;margin-bottom:0.8rem;">📊 {selected_city} — Scores</div>', unsafe_allow_html=True)
    _sc = CITY_SCORES.get(selected_city, [88, 75, 92, 70, 60, 82])
    nb_items = [
        ("🏫", "Schools & Education", _sc[0], "#1565C0"),
        ("🏥", "Healthcare",          _sc[1], "#00C853"),
        ("🛒", "Markets & Shopping",  _sc[2], "#00BCD4"),
        ("🚌", "Public Transport",    _sc[3], "#FFB300"),
        ("🌳", "Green Spaces",        _sc[4], "#4CAF50"),
        ("🔒", "Safety Index",        _sc[5], "#7C4DFF"),
    ]
    for icon, name, score, color in nb_items:
        st.markdown(f"""
        <div class="nb-card2">
          <div class="nb-icon2">{icon}</div>
          <div class="nb-info2">
            <div class="nb-name2">{name}</div>
            <div class="nb-barw"><div class="nb-barf" style="width:{score}%;background:{color};"></div></div>
          </div>
          <div class="nb-sc" style="color:{color};">{score}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#546E7A;margin:1rem 0 0.5rem;">Nearby Comparable Properties</div>', unsafe_allow_html=True)
    for col_hex, name, ptag in [
        ("#C62828","Imperial Heights","₹1.8 Cr"),
        ("#2E7D32","Green Valley Home","₹62 L"),
        ("#6A1B9A","Casa Del Sol","₹2.4 Cr"),
        ("#E65100","Metro View Flat","₹45 L"),
    ]:
        st.markdown(f"""
        <div class="comp2">
          <div class="comp2-dot" style="background:{col_hex};"></div>
          <div class="comp2-lbl">{name}</div>
          <div class="comp2-p">{ptag}</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <span>ESTATEIQ PRO</span> &nbsp;·&nbsp; AI House Price Estimator &nbsp;·&nbsp;
    Streamlit + Scikit-Learn &nbsp;·&nbsp; <span>Model Active</span>
</div>
""", unsafe_allow_html=True)
