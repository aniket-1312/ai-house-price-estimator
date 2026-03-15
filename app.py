import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EstateIQ – AI House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --cream:    #F5F0E8;
    --warm:     #E8DFD0;
    --espresso: #1C1208;
    --walnut:   #3D2B1F;
    --caramel:  #C4882A;
    --gold:     #E8A827;
    --sage:     #5C7A5A;
    --blush:    #D4847A;
    --sand:     #A89070;
    --ink:      #2C2416;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background: linear-gradient(135deg, #F5F0E8 0%, #EDE5D8 50%, #E5DDD0 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--espresso) !important;
    border-right: 3px solid var(--caramel);
}

[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--sand) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(196,136,42,0.4) !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--caramel), var(--gold)) !important;
    color: var(--espresso) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(196,136,42,0.4) !important;
    margin-top: 12px !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(196,136,42,0.6) !important;
}

/* ── Main Content ── */
.main-header {
    background: linear-gradient(135deg, var(--espresso) 0%, var(--walnut) 60%, #5a3a25 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(28,18,8,0.25);
}

.main-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(196,136,42,0.25) 0%, transparent 70%);
}

.main-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,168,39,0.15) 0%, transparent 70%);
}

.main-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    color: var(--cream) !important;
    line-height: 1.1 !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.02em;
}

.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    color: var(--sand);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0;
}

.accent-dot {
    color: var(--gold);
}

/* ── Price Card ── */
.price-card {
    background: linear-gradient(135deg, var(--espresso), #2d1f10);
    border: 2px solid var(--caramel);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(28,18,8,0.3), inset 0 1px 0 rgba(196,136,42,0.3);
    position: relative;
    overflow: hidden;
}

.price-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--caramel), var(--gold), var(--caramel));
}

.price-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--sand);
    margin-bottom: 0.5rem;
}

.price-value {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--gold), var(--caramel));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.price-sub {
    font-size: 0.8rem;
    color: var(--sand);
    margin-top: 0.5rem;
}

/* ── Section Headers ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--walnut);
    border-bottom: 2px solid var(--caramel);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    letter-spacing: -0.01em;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.metric-pill {
    background: white;
    border: 1.5px solid var(--warm);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    flex: 1;
    min-width: 120px;
    box-shadow: 0 4px 16px rgba(28,18,8,0.07);
    text-align: center;
}

.metric-pill-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--sand);
    display: block;
    margin-bottom: 0.3rem;
}

.metric-pill-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--walnut);
}

/* ── Info Badge ── */
.info-badge {
    background: rgba(92,122,90,0.1);
    border: 1px solid rgba(92,122,90,0.4);
    border-left: 4px solid var(--sage);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--walnut);
    margin-top: 1rem;
}

/* ── Feature Tags ── */
.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.tag {
    background: rgba(196,136,42,0.12);
    border: 1px solid rgba(196,136,42,0.4);
    border-radius: 50px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--caramel);
    letter-spacing: 0.08em;
}

.tag.active {
    background: var(--caramel);
    color: white;
    border-color: var(--caramel);
}

/* ── Image Gallery ── */
.gallery-img {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(28,18,8,0.15);
    transition: transform 0.3s ease;
}

/* ── Matplotlib Figure ── */
.stPlotlyChart, .stImage {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* ── Divider ── */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--caramel), transparent);
    margin: 2rem 0;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--cream) !important;
    letter-spacing: -0.02em;
    padding: 1rem 0 0.25rem;
}

.sidebar-brand span {
    color: var(--gold) !important;
}

.sidebar-tagline {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--sand) !important;
    margin-bottom: 1.5rem;
}

/* ── Selectbox ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(196,136,42,0.4) !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
}

/* ── Number input buttons ── */
[data-testid="stSidebar"] button[kind="secondary"] {
    background: rgba(196,136,42,0.3) !important;
    border: none !important;
    color: var(--cream) !important;
}

/* ── Section dividers in sidebar ── */
.sidebar-section {
    border-top: 1px solid rgba(196,136,42,0.25);
    margin: 1rem 0 0.75rem 0;
    padding-top: 0.75rem;
}

.sidebar-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--caramel) !important;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib Dark Theme ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1C1208",
    "axes.facecolor":    "#241608",
    "axes.edgecolor":    "#3D2B1F",
    "axes.labelcolor":   "#A89070",
    "axes.titlecolor":   "#F5F0E8",
    "xtick.color":       "#A89070",
    "ytick.color":       "#A89070",
    "grid.color":        "#3D2B1F",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "#F5F0E8",
    "font.family":       "sans-serif",
})

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Estate<span>IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI-Powered Valuation</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">📐 Size & Structure</div></div>', unsafe_allow_html=True)
    area       = st.number_input("Area (sqft)", 500, 10000, 2000, step=50)
    bedrooms   = st.number_input("Bedrooms",      1, 10, 3)
    bathrooms  = st.number_input("Bathrooms",     1,  5, 2)
    stories    = st.number_input("Stories",       1,  4, 2)
    parking    = st.number_input("Parking Spaces",0,  5, 1)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🏡 Amenities</div></div>', unsafe_allow_html=True)
    mainroad       = st.selectbox("Main Road Access",  ["Yes", "No"])
    guestroom      = st.selectbox("Guest Room",        ["Yes", "No"])
    basement       = st.selectbox("Basement",          ["Yes", "No"])
    airconditioning= st.selectbox("Air Conditioning",  ["Yes", "No"])
    prefarea       = st.selectbox("Preferred Area",    ["Yes", "No"])

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">🛋 Interior</div></div>', unsafe_allow_html=True)
    furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

    predict_clicked = st.button("✦ Estimate Price", key="predict_btn")

# ─── Input DataFrame ─────────────────────────────────────────────────────────────
input_data = pd.DataFrame({
    "area":                           [area],
    "bedrooms":                       [bedrooms],
    "bathrooms":                      [bathrooms],
    "stories":                        [stories],
    "parking":                        [parking],
    "mainroad_yes":                   [1 if mainroad       == "Yes" else 0],
    "guestroom_yes":                  [1 if guestroom      == "Yes" else 0],
    "basement_yes":                   [1 if basement       == "Yes" else 0],
    "hotwaterheating_yes":            [0],
    "airconditioning_yes":            [1 if airconditioning== "Yes" else 0],
    "prefarea_yes":                   [1 if prefarea       == "Yes" else 0],
    "furnishingstatus_semi-furnished":[1 if furnishing     == "Semi-Furnished" else 0],
    "furnishingstatus_unfurnished":   [1 if furnishing     == "Unfurnished" else 0],
})

# ─── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-subtitle">Powered by Machine Learning</div>
    <div class="main-title">The Smartest Way to<br>Value a Property<span class="accent-dot">.</span></div>
</div>
""", unsafe_allow_html=True)

# ─── Results Section ─────────────────────────────────────────────────────────────
if predict_clicked:
    price = model.predict(input_data)[0]

    col_price, col_details = st.columns([1, 2], gap="large")

    with col_price:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Value</div>
            <div class="price-value">${price:,.0f}</div>
            <div class="price-sub">Based on 13 property features</div>
        </div>
        """, unsafe_allow_html=True)

        # Feature tags
        active_features = []
        if mainroad        == "Yes": active_features.append("Main Road")
        if guestroom       == "Yes": active_features.append("Guest Room")
        if basement        == "Yes": active_features.append("Basement")
        if airconditioning == "Yes": active_features.append("A/C")
        if prefarea        == "Yes": active_features.append("Pref. Area")
        active_features.append(furnishing)

        tags_html = '<div class="tag-row">'
        for f in ["Main Road","Guest Room","Basement","A/C","Pref. Area","Furnished","Semi-Furnished","Unfurnished"]:
            cls = "tag active" if f in active_features else "tag"
            tags_html += f'<span class="{cls}">{f}</span>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown('<div class="info-badge">🤖 Predicted by a trained ML regression model. Actual prices may vary based on market conditions.</div>', unsafe_allow_html=True)

    with col_details:
        st.markdown('<div class="section-title">Property Snapshot</div>', unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("🏠 Area",      f"{area:,} sqft")
        r1c2.metric("🛏 Bedrooms",   bedrooms)
        r1c3.metric("🚿 Bathrooms",  bathrooms)

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("🏗 Stories",    stories)
        r2c2.metric("🚗 Parking",    parking)
        r2c3.metric("💰 Per sqft",   f"${price/area:,.0f}")

    # ─── Charts Row ──────────────────────────────────────────────────────────────
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analytics & Insights</div>', unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        # Price Trend Chart
        years  = np.array([2020, 2021, 2022, 2023, 2024, 2025])
        growth = np.array([0.60, 0.70, 0.80, 0.90, 0.96, 1.00])
        prices = price * growth

        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.fill_between(years, prices, alpha=0.18, color="#C4882A")
        ax1.plot(years, prices, color="#E8A827", lw=2.5, marker="o",
                 markersize=6, markerfacecolor="#1C1208", markeredgecolor="#E8A827", markeredgewidth=2)

        # Annotate last point
        ax1.annotate(f"${price:,.0f}",
                     xy=(2025, price),
                     xytext=(-60, 12), textcoords="offset points",
                     fontsize=9, color="#E8A827", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#E8A827", lw=1.2))

        ax1.set_title("Estimated Price Growth (2020–2025)", fontsize=11, pad=10)
        ax1.set_xlabel("Year", fontsize=9)
        ax1.set_ylabel("Estimated Price ($)", fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e5:.1f}L"))
        ax1.grid(True, axis="y")
        ax1.spines[["top","right"]].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1)

    with ch2:
        # Feature Importance Radar-style horizontal bar
        feat_names = ["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"]
        feat_vals  = [area, bedrooms*300, bathrooms*200, stories*150, parking*100]
        feat_norm  = np.array(feat_vals) / max(feat_vals)

        colors = ["#E8A827","#C4882A","#A89070","#5C7A5A","#7A9078"]
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        bars = ax2.barh(feat_names, feat_norm, color=colors, height=0.55,
                        edgecolor="none")

        for bar, val in zip(bars, feat_norm):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f"{val:.0%}", va="center", fontsize=9, color="#F5F0E8")

        ax2.set_xlim(0, 1.22)
        ax2.set_title("Property Feature Contribution", fontsize=11, pad=10)
        ax2.set_xlabel("Relative Weight", fontsize=9)
        ax2.spines[["top","right","bottom"]].set_visible(False)
        ax2.tick_params(left=False)
        ax2.grid(False)
        ax2.xaxis.set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    # ─── Price Bracket ────────────────────────────────────────────────────────────
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Market Positioning</div>', unsafe_allow_html=True)

    low, high = price * 0.9, price * 1.1
    brackets = {
        "Budget":    (0,         price*0.6),
        "Mid-Range": (price*0.6, price*0.85),
        "Your Home": (price*0.85,price*1.15),
        "Premium":   (price*1.15,price*1.5),
        "Luxury":    (price*1.5, price*2.0),
    }

    fig3, ax3 = plt.subplots(figsize=(10, 1.6))
    palette = ["#3D2B1F","#5a3a25","#C4882A","#5C7A5A","#2a4a28"]
    prev = 0
    for (label, (lo, hi)), col in zip(brackets.items(), palette):
        width = hi - lo
        ax3.barh(0, width, left=lo, color=col, height=0.5)
        ax3.text(lo + width/2, 0, label,
                 ha="center", va="center", fontsize=8.5,
                 fontweight="bold", color="white")

    ax3.axvline(price, color="#E8A827", lw=2.5, linestyle="--")
    ax3.text(price, 0.32, f"  ▼ ${price:,.0f}", color="#E8A827",
             fontsize=9, fontweight="bold", va="bottom")
    ax3.set_xlim(0, price*2.0)
    ax3.axis("off")
    ax3.set_facecolor("#1C1208")
    fig3.patch.set_facecolor("#1C1208")
    fig3.tight_layout()
    st.pyplot(fig3)

# ─── Always-Visible Sections ────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# Gallery
st.markdown('<div class="section-title">Property Showcase</div>', unsafe_allow_html=True)
g1, g2, g3 = st.columns(3, gap="medium")
with g1:
    st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=600",
             caption="Modern Minimalist", use_container_width=True)
with g2:
    st.image("https://images.unsplash.com/photo-1572120360610-d971b9d7767c?w=600",
             caption="Luxury Villa", use_container_width=True)
with g3:
    st.image("https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=600",
             caption="Family Home", use_container_width=True)

# Map
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📍 Sample Property Location — Rajkot, Gujarat</div>', unsafe_allow_html=True)
map_data = pd.DataFrame({"lat": [22.3039], "lon": [70.8022]})
st.map(map_data, zoom=13)

# Footer
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem; color: #A89070; font-size:0.78rem; letter-spacing:0.1em;">
    ESTATEIQ  ·  AI HOUSE PRICE ESTIMATOR  ·  BUILT WITH STREAMLIT & SCIKIT-LEARN
</div>
""", unsafe_allow_html=True)
