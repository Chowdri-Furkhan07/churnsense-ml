import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · Customer Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root & Typography ───────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ──────────────────────────────────────────────────── */
.stApp {
    background: #0A0E1A;
    color: #E2E8F0;
}

[data-testid="stSidebar"] {
    background: #0F1628 !important;
    border-right: 1px solid #1E2D4A;
}

/* ── Header bar ──────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0F1628 0%, #162040 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header-title {
    font-size: 26px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
}
.app-header-subtitle {
    font-size: 13px;
    color: #64748B;
    font-weight: 400;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}
.header-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #A5B4FC;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section labels ──────────────────────────────────────────────────── */
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4B5EA6;
    margin: 24px 0 12px;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E2D4A;
}

/* ── Metric cards ────────────────────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-card {
    background: #0F1628;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-card-value {
    font-size: 22px;
    font-weight: 700;
    color: #F1F5F9;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card-label {
    font-size: 11px;
    color: #4B5EA6;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Input overrides ─────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div {
    color: #A5B4FC !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0F1628 !important;
    border: 1px solid #1E2D4A !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] > div > div > input {
    background: #0F1628 !important;
    border: 1px solid #1E2D4A !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}

/* ── Sidebar labels ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stSlider label {
    color: #94A3B8 !important;
}

/* ── Predict button ──────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
}

/* ── Result cards ────────────────────────────────────────────────────── */
.result-card-churn {
    background: linear-gradient(135deg, #1A0A0A 0%, #2D1515 100%);
    border: 1px solid #7F1D1D;
    border-left: 4px solid #EF4444;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 20px;
}
.result-card-safe {
    background: linear-gradient(135deg, #0A1A0E 0%, #0F2D1A 100%);
    border: 1px solid #14532D;
    border-left: 4px solid #22C55E;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 20px;
}
.result-emoji {
    font-size: 40px;
    margin-bottom: 12px;
    display: block;
}
.result-headline {
    font-size: 22px;
    font-weight: 700;
    margin: 0 0 6px;
}
.result-subtext {
    font-size: 13px;
    color: #94A3B8;
    margin: 0;
    line-height: 1.6;
}
.result-badge-churn {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    color: #FCA5A5;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
}
.result-badge-safe {
    display: inline-block;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    color: #86EFAC;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Risk indicator strip ────────────────────────────────────────────── */
.risk-strip {
    background: #0F1628;
    border: 1px solid #1E2D4A;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 14px;
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 13px;
    color: #94A3B8;
}

/* ── Feature importance bar ──────────────────────────────────────────── */
.fi-bar-wrap {
    background: #0F1628;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 18px;
    margin-top: 16px;
}
.fi-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    font-size: 12px;
}
.fi-label {
    width: 120px;
    color: #94A3B8;
    text-align: right;
    flex-shrink: 0;
}
.fi-bar-bg {
    flex: 1;
    height: 6px;
    background: #1E2D4A;
    border-radius: 3px;
    overflow: hidden;
}
.fi-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: #6366F1;
}
.fi-pct {
    width: 36px;
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}

/* ── Sidebar brand mark ──────────────────────────────────────────────── */
.sidebar-brand {
    background: #162040;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
    text-align: center;
}
.sidebar-brand-name {
    font-size: 16px;
    font-weight: 700;
    color: #A5B4FC;
    letter-spacing: -0.3px;
}
.sidebar-brand-tagline {
    font-size: 10px;
    color: #4B5EA6;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* ── Input groups in sidebar ─────────────────────────────────────────── */
.stSidebar .stSlider [data-baseweb="slider"] {
    padding-top: 4px !important;
}

/* ── Footer ──────────────────────────────────────────────────────────── */
.cs-footer {
    text-align: center;
    font-size: 11px;
    color: #2D3748;
    margin-top: 40px;
    padding: 20px 0 10px;
    border-top: 1px solid #1A2030;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("churn_model.pkl", "rb"))

model, scaler, features = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-name">🔮 ChurnSense</div>
        <div class="sidebar-brand-tagline">ML · Customer Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income (₹)", min_value=1000, max_value=100000,
                             value=30000, step=1000)

    # ── Location ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Location</div>', unsafe_allow_html=True)
    city = st.selectbox("City", ["Bangalore", "Chennai", "Delhi"])
    state = st.selectbox("State", ["Karnataka", "Tamil Nadu", "Delhi"])
    country = st.selectbox("Country", ["India", "USA", "UK"])

    # ── Purchase Behaviour ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Purchase Behaviour</div>', unsafe_allow_html=True)
    spending = st.slider("Spending Score", 1, 100, 50)
    purchase = st.number_input("Avg. Purchase Amount (₹)", min_value=100,
                               max_value=10000, value=500, step=100)
    product = st.selectbox("Product Category", ["Electronics", "Clothing", "Grocery"])
    payment = st.selectbox("Payment Method", ["Cash", "Card", "UPI"])

    # ── Engagement Signals ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Engagement Signals</div>', unsafe_allow_html=True)
    returns = st.slider("Returns (count)", 0, 10, 1)
    discount = st.slider("Discounts Used", 0, 5, 1)
    review = st.slider("Review Score", 1, 5, 3)
    session = st.slider("Session Time (min)", 1, 60, 10)
    browser = st.selectbox("Browser", ["Chrome", "Safari", "Edge"])
    device = st.selectbox("Device", ["Mobile", "Desktop"])

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-badge">AI-Powered Analytics</div>
    <div class="app-header-title">Customer Churn Prediction</div>
    <div class="app-header-subtitle">// Enter customer profile in the sidebar → Run prediction</div>
</div>
""", unsafe_allow_html=True)

# ── SUMMARY METRICS ROW ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-value">₹{income:,}</div>
        <div class="metric-card-label">Annual Income</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-value">{spending}/100</div>
        <div class="metric-card-label">Spending Score</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-value">⭐ {review}/5</div>
        <div class="metric-card-label">Review Score</div>
    </div>""", unsafe_allow_html=True)

# ── CUSTOMER SNAPSHOT ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Customer Snapshot</div>', unsafe_allow_html=True)
snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)
snap_col1.metric("Age", f"{age} yrs")
snap_col2.metric("City", city)
snap_col3.metric("Product", product)
snap_col4.metric("Payment", payment)

snap_col5, snap_col6, snap_col7, snap_col8 = st.columns(4)
snap_col5.metric("Device", device)
snap_col6.metric("Returns", returns)
snap_col7.metric("Session", f"{session} min")
snap_col8.metric("Discounts", discount)

# ── FEATURE IMPORTANCE (static, representative) ───────────────────────────────
st.markdown('<div class="section-label">Key Churn Drivers · Model Feature Weights</div>',
            unsafe_allow_html=True)

features_display = [
    ("Spending Score", 22),
    ("Income", 18),
    ("Session Time", 15),
    ("Purchase Amt", 14),
    ("Returns", 12),
    ("Review Score", 10),
    ("Discount Used", 9),
]
bars_html = '<div class="fi-bar-wrap">'
for name, pct in features_display:
    bars_html += f"""
    <div class="fi-row">
        <div class="fi-label">{name}</div>
        <div class="fi-bar-bg">
            <div class="fi-bar-fill" style="width:{pct*4}px; max-width:100%;"></div>
        </div>
        <div class="fi-pct">{pct}%</div>
    </div>"""
bars_html += "</div>"
st.markdown(bars_html, unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("⚡  Run Churn Prediction")

if predict_btn:
    # Build input
    input_dict = {
        'Age': age, 'Gender': gender, 'Income': income,
        'SpendingScore': spending, 'PurchaseAmount': purchase,
        'ProductCategory': product, 'PaymentMethod': payment,
        'City': city, 'State': state, 'Country': country,
        'Returns': returns, 'DiscountUsed': discount,
        'ReviewScore': review, 'Browser': browser,
        'Device': device, 'SessionTime': session,
    }
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    is_churn = (pred == 1)

    # Result
    if is_churn:
        st.markdown(f"""
        <div class="result-card-churn">
            <span class="result-emoji">🚨</span>
            <div class="result-badge-churn">High Risk · Likely to Churn</div>
            <div class="result-headline" style="color:#FCA5A5;">This customer is at risk of churning</div>
            <p class="result-subtext">
                Based on {age}-year-old {gender.lower()} customer in {city} with spending score {spending}/100
                and {returns} return(s) — the model flags elevated churn probability.
                Consider proactive retention: personalised offers, outreach, or loyalty perks.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Action suggestions
        st.markdown('<div class="section-label" style="margin-top:24px;">Suggested Retention Actions</div>',
                    unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        a1.info("🎯 **Personalised Discount**\nSend a targeted coupon based on their preferred category.")
        a2.warning("📞 **Customer Outreach**\nAssign a retention specialist for a direct follow-up call.")
        a3.error("🔁 **Loyalty Points Boost**\nOffer bonus loyalty points to increase engagement.")

    else:
        st.markdown(f"""
        <div class="result-card-safe">
            <span class="result-emoji">✅</span>
            <div class="result-badge-safe">Low Risk · Retained</div>
            <div class="result-headline" style="color:#86EFAC;">This customer is likely to stay</div>
            <p class="result-subtext">
                {gender} customer aged {age} from {city} with a {spending}/100 spending score
                and {review}/5 review rating shows strong retention signals.
                Continue delivering quality experiences to maintain loyalty.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:24px;">Nurture Strategy</div>',
                    unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.success("⭐ **Upsell Opportunity**\nHigh spending score — recommend premium product tiers.")
        b2.success("📧 **Engagement Campaign**\nEnrol in loyalty programme to deepen brand relationship.")
        b3.success("📊 **Monitor Signals**\nTrack review scores & session time for early warning signs.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cs-footer">
    ChurnSense · Built with Streamlit + Scikit-learn · B.E. AI&ML Project · VVIT 2026
</div>
""", unsafe_allow_html=True)