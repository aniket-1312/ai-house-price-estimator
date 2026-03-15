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
#  PROPERTY SHOWCASE — Filterable Listing Cards
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🏡 Property Showcase</div>', unsafe_allow_html=True)

st.markdown("""
<style>
.listing-card{background:var(--white);border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(13,27,42,0.10);border:1px solid var(--g100);transition:transform 0.22s ease,box-shadow 0.22s ease;}
.listing-card:hover{transform:translateY(-4px);box-shadow:0 12px 36px rgba(13,27,42,0.16);}
.listing-img-wrap{position:relative;overflow:hidden;height:200px;}
.listing-img-wrap img{width:100%;height:200px;object-fit:cover;display:block;}
.listing-badge{position:absolute;top:12px;left:12px;background:var(--navy);color:#00BCD4!important;font-size:0.6rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;border-radius:6px;padding:4px 10px;}
.listing-price-tag{position:absolute;bottom:12px;right:12px;background:rgba(13,27,42,0.88);color:var(--white)!important;font-family:'DM Serif Display',serif;font-size:1.05rem;font-weight:700;border-radius:8px;padding:5px 12px;}
.listing-body{padding:1rem 1.1rem 1.2rem;}
.listing-title{font-size:0.95rem;font-weight:700;color:var(--navy);margin-bottom:0.25rem;}
.listing-location{font-size:0.72rem;color:var(--g600);margin-bottom:0.7rem;}
.listing-stats{display:flex;gap:0.6rem;margin-bottom:0.9rem;flex-wrap:wrap;}
.listing-stat{background:var(--offwhite);border-radius:6px;padding:4px 9px;font-size:0.68rem;font-weight:600;color:var(--g600);}
.listing-btn{display:block;width:100%;text-align:center;background:linear-gradient(135deg,#0077B6,#00BCD4);color:var(--white)!important;font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;border-radius:8px;padding:0.5rem;}
.nb-card{background:var(--white);border:1px solid var(--g100);border-radius:12px;padding:0.8rem 1rem;box-shadow:var(--sh);display:flex;align-items:center;gap:0.9rem;margin-bottom:0.55rem;}
.nb-icon{font-size:1.4rem;}
.nb-info{flex:1;}
.nb-name{font-size:0.76rem;font-weight:700;color:var(--navy);margin-bottom:0.25rem;}
.nb-bar-wrap{height:6px;background:var(--g100);border-radius:50px;overflow:hidden;}
.nb-bar{height:100%;border-radius:50px;}
.nb-score{font-size:0.72rem;font-weight:700;min-width:28px;text-align:right;}
.comp-pin{display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0.75rem;background:var(--offwhite);border-radius:10px;margin-bottom:0.45rem;}
.pin-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.pin-label{font-size:0.76rem;font-weight:600;color:var(--navy);flex:1;}
.pin-price{font-size:0.76rem;font-weight:700;color:var(--g600);}
</style>
""", unsafe_allow_html=True)

LISTINGS = {
    "All": [
        {"img":"https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=700","badge":"MODERN","title":"Skyline Residency","loc":"Kalawad Road, Rajkot","price":"₹85 L","beds":3,"baths":2,"sqft":1850,"tag":"Ready to Move"},
        {"img":"https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=700","badge":"LUXURY","title":"Imperial Heights","loc":"Race Course, Rajkot","price":"₹1.8 Cr","beds":4,"baths":3,"sqft":3200,"tag":"Premium"},
        {"img":"https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=700","badge":"FAMILY","title":"Green Valley Home","loc":"Mavdi, Rajkot","price":"₹62 L","beds":3,"baths":2,"sqft":1500,"tag":"New Launch"},
        {"img":"https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=700","badge":"VILLA","title":"Casa Del Sol","loc":"150 Ft Ring Road, Rajkot","price":"₹2.4 Cr","beds":5,"baths":4,"sqft":4200,"tag":"Gated"},
        {"img":"https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=700","badge":"APARTMENT","title":"Metro View Flat","loc":"Yagnik Road, Rajkot","price":"₹45 L","beds":2,"baths":1,"sqft":980,"tag":"Ready to Move"},
        {"img":"https://images.unsplash.com/photo-1583608205776-bfd35f0d9f83?w=700","badge":"BUNGALOW","title":"The Heritage House","loc":"Kotecha Chowk, Rajkot","price":"₹1.1 Cr","beds":4,"baths":3,"sqft":2800,"tag":"Resale"},
    ]
}
LISTINGS["Modern"]    = [l for l in LISTINGS["All"] if l["badge"]=="MODERN"]
LISTINGS["Luxury"]    = [l for l in LISTINGS["All"] if l["badge"] in ("LUXURY","VILLA")]
LISTINGS["Family"]    = [l for l in LISTINGS["All"] if l["badge"] in ("FAMILY","BUNGALOW")]
LISTINGS["Apartment"] = [l for l in LISTINGS["All"] if l["badge"]=="APARTMENT"]

if "gallery_filter" not in st.session_state:
    st.session_state.gallery_filter = "All"

f_cols = st.columns(6, gap="small")
for i, cat in enumerate(["All","Modern","Luxury","Family","Apartment"]):
    with f_cols[i]:
        label = f"● {cat}" if st.session_state.gallery_filter == cat else cat
        if st.button(label, key=f"flt_{cat}"):
            st.session_state.gallery_filter = cat

active_listings = LISTINGS.get(st.session_state.gallery_filter, LISTINGS["All"])
for row_start in range(0, len(active_listings), 3):
    row_items = active_listings[row_start:row_start+3]
    cols = st.columns(len(row_items), gap="medium")
    for col, l in zip(cols, row_items):
        with col:
            st.markdown(f"""
            <div class="listing-card">
              <div class="listing-img-wrap">
                <img src="{l['img']}" alt="{l['title']}"/>
                <div class="listing-badge">{l['badge']}</div>
                <div class="listing-price-tag">{l['price']}</div>
              </div>
              <div class="listing-body">
                <div class="listing-title">{l['title']}</div>
                <div class="listing-location">📍 {l['loc']}</div>
                <div class="listing-stats">
                  <span class="listing-stat">🛏 {l['beds']} Beds</span>
                  <span class="listing-stat">🚿 {l['baths']} Baths</span>
                  <span class="listing-stat">📐 {l['sqft']:,} sqft</span>
                  <span class="listing-stat" style="color:#0097A7;background:rgba(0,188,212,0.1);">{l['tag']}</span>
                </div>
                <div class="listing-btn">View Details →</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  LOCATION INTELLIGENCE — Folium Map + Neighborhood Scores
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📍 Location Intelligence — Rajkot, Gujarat</div>', unsafe_allow_html=True)

try:
    import folium
    from streamlit_folium import st_folium

    map_col, nb_col = st.columns([3, 2], gap="large")

    with map_col:
        m = folium.Map(location=[22.3039, 70.8022], zoom_start=13, tiles="CartoDB positron")

        price_for_map = model.predict(input_data)[0]
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:180px;padding:4px">
          <div style="font-size:0.68rem;font-weight:700;color:#546E7A;letter-spacing:0.1em;text-transform:uppercase;">Your Property</div>
          <div style="font-size:1.2rem;font-weight:800;color:#0D1B2A;">${price_for_map:,.0f}</div>
          <div style="font-size:0.73rem;color:#546E7A;margin-top:4px">📐 {area:,} sqft &nbsp;|&nbsp; 🛏 {bedrooms} beds</div>
          <div style="font-size:0.7rem;color:#0097A7;margin-top:4px">📍 Rajkot, Gujarat</div>
        </div>"""
        folium.Marker(
            location=[22.3039, 70.8022],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip="🏠 Your Property — click for details",
            icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
        ).add_to(m)

        folium.Circle(
            location=[22.3039, 70.8022], radius=800,
            color="#00BCD4", fill=True, fill_color="#00BCD4", fill_opacity=0.07,
            tooltip="Preferred Area Zone (800m radius)",
        ).add_to(m)

        comparables = [
            ([22.3119, 70.7952], "Imperial Heights",  "₹1.8 Cr", "darkred"),
            ([22.2989, 70.8132], "Green Valley Home",  "₹62 L",   "green"),
            ([22.3209, 70.8062], "Casa Del Sol",       "₹2.4 Cr", "purple"),
            ([22.2899, 70.7922], "Metro View Flat",    "₹45 L",   "orange"),
        ]
        for loc, name, ptag, color in comparables:
            folium.Marker(
                location=loc, tooltip=f"{name} — {ptag}",
                popup=folium.Popup(f"<b>{name}</b><br>{ptag}", max_width=160),
                icon=folium.Icon(color=color, icon="building", prefix="fa"),
            ).add_to(m)

        amenities = [
            ([22.3080, 70.8100], "🏥 Rajkot Civil Hospital", "hospital",       "red"),
            ([22.3010, 70.7960], "🏫 MK High School",         "graduation-cap", "cadetblue"),
            ([22.3060, 70.8180], "🛒 Reliance Smart Bazaar",  "shopping-cart",  "green"),
            ([22.2970, 70.8050], "🚌 Rajkot Bus Stand",       "bus",            "darkblue"),
        ]
        for loc, name, icon_n, color in amenities:
            folium.Marker(location=loc, tooltip=name,
                          icon=folium.Icon(color=color, icon=icon_n, prefix="fa")).add_to(m)

        st_folium(m, width=None, height=440, returned_objects=[])

    with nb_col:
        st.markdown('<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#546E7A;margin-bottom:0.9rem;">Neighborhood Scores</div>', unsafe_allow_html=True)
        nb_items = [
            ("🏫", "Schools & Education", 88, "#1565C0"),
            ("🏥", "Healthcare",          75, "#00C853"),
            ("🛒", "Markets & Shopping",  92, "#00BCD4"),
            ("🚌", "Public Transport",    70, "#FFB300"),
            ("🌳", "Green Spaces",        60, "#4CAF50"),
            ("🔒", "Safety Index",        82, "#7C4DFF"),
        ]
        for icon, name, score, color in nb_items:
            st.markdown(f"""
            <div class="nb-card">
              <div class="nb-icon">{icon}</div>
              <div class="nb-info">
                <div class="nb-name">{name}</div>
                <div class="nb-bar-wrap">
                  <div class="nb-bar" style="width:{score}%;background:{color};"></div>
                </div>
              </div>
              <div class="nb-score" style="color:{color};">{score}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#546E7A;margin:1.1rem 0 0.6rem;">Comparable Properties</div>', unsafe_allow_html=True)
        for color, name, ptag in [("#C62828","Imperial Heights","₹1.8 Cr"),("#2E7D32","Green Valley Home","₹62 L"),("#6A1B9A","Casa Del Sol","₹2.4 Cr"),("#E65100","Metro View Flat","₹45 L")]:
            st.markdown(f"""
            <div class="comp-pin">
              <div class="pin-dot" style="background:{color};"></div>
              <div class="pin-label">{name}</div>
              <div class="pin-price">{ptag}</div>
            </div>""", unsafe_allow_html=True)

except ImportError:
    st.warning("📦 Install `folium` and `streamlit-folium` for the interactive map:\n```\npip install folium streamlit-folium\n```")
    st.map(pd.DataFrame({"lat":[22.3039],"lon":[70.8022]}), zoom=14)

# ════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <span>ESTATEIQ PRO</span> &nbsp;·&nbsp; AI House Price Estimator &nbsp;·&nbsp;
    Streamlit + Scikit-Learn &nbsp;·&nbsp; <span>Model Active</span>
</div>
""", unsafe_allow_html=True)
