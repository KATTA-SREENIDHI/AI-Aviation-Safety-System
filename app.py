
import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="AeroSafe | Flight Intelligence",
    page_icon="✈️",
    layout="wide"
)

# -------------------- CUSTOM DESIGN --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Mono:wght@400;500&display=swap');

.stApp {
    background: #F1F3F5;
    color: #20242A;
    font-family: 'Space Grotesk', sans-serif;
}

header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.topbar {
    background: #20242A;
    padding: 18px 24px;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
}

.brand {
    font-size: 22px;
    font-weight: 700;
    color: white;
    letter-spacing: 2px;
}

.brand span {
    color: #FF6B35;
}

.status {
    color: #B8C2CC;
    font-family: 'IBM Mono', monospace;
    font-size: 12px;
}

.hero {
    background: #20242A;
    padding: 38px;
    border-radius: 14px;
    color: white;
    margin-bottom: 25px;
    position: relative;
}

.hero-label {
    color: #FF6B35;
    font-family: 'IBM Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    margin-bottom: 15px;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.1;
    color: white;
    font-weight: 700;
    margin: 0;
}

.hero p {
    color: #AAB3BD;
    font-size: 15px;
    margin-top: 12px;
}

.section-label {
    font-family: 'IBM Mono', monospace;
    color: #707984;
    font-size: 12px;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.panel {
    background: white;
    border: 1px solid #DCE1E6;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 18px;
}

.panel-title {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 4px;
    color: #20242A;
}

.panel-sub {
    color: #89919B;
    font-size: 12px;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    border: 1px solid #DCE1E6;
    border-left: 4px solid #FF6B35;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 12px;
}

.metric-label {
    color: #828A94;
    font-family: 'IBM Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
}

.metric-value {
    color: #20242A;
    font-size: 26px;
    font-weight: 700;
    margin-top: 5px;
}

div.stButton > button {
    background: #FF6B35;
    color: white;
    border: none;
    border-radius: 8px;
    height: 50px;
    font-weight: 700;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #E95522;
    color: white;
    border: none;
}

div[data-testid="stSlider"] label {
    color: #333941 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetric"] {
    background: #F7F8FA;
    border: 1px solid #E0E4E8;
    border-radius: 10px;
    padding: 15px;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
}

hr {
    border-color: #DCE1E6;
}

.footer {
    color: #9299A2;
    font-family: 'IBM Mono', monospace;
    font-size: 11px;
    text-align: center;
    padding: 25px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL --------------------
MODEL_PATH = "model/aviation_model.pkl"

# -------------------- HEADER --------------------
st.markdown("""
<div class="topbar">
    <div class="brand">AEROSAFE<span> / AI</span></div>
    <div class="status">● MONITORING SYSTEM ONLINE</div>
</div>

<div class="hero">
    <div class="hero-label">FLIGHT INTELLIGENCE SYSTEM / 01</div>
    <h1>Flight Safety<br>Analysis Center</h1>
    <p>Flight parameter monitoring and machine learning risk classification.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run python train_model.py first.")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------------------- OVERVIEW --------------------
st.markdown('<div class="section-label">01 / SYSTEM OVERVIEW</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL TYPE</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">DATA INPUTS</div>
        <div class="metric-value">06 Parameters</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">CLASSIFICATION</div>
        <div class="metric-value">Safe / Unsafe</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- INPUTS --------------------
st.markdown('<div class="section-label">02 / FLIGHT TELEMETRY</div>', unsafe_allow_html=True)

left, right = st.columns([1.5, 1])

with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Flight Parameters</div>
        <div class="panel-sub">Adjust telemetry values for model evaluation.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        altitude = st.slider("Altitude (ft)", 500, 40000, 10000)
        airspeed = st.slider("Airspeed (knots)", 80, 600, 250)
        vertical_speed = st.slider("Vertical Speed (ft/min)", -5000, 5000, -500)

    with c2:
        pitch = st.slider("Pitch Angle (°)", -15.0, 20.0, 3.0)
        visibility = st.slider("Visibility (km)", 0.1, 15.0, 10.0)
        temperature = st.slider("Temperature (°C)", -60.0, 40.0, 15.0)

with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Aircraft Monitoring</div>
        <div class="panel-sub">Illustrative flight monitoring graphic</div>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1436491865332-7a61a109c05f?auto=format&fit=crop&w=1000&q=85",
        use_container_width=True
    )

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ANALYSIS ENGINE</div>
        <div class="metric-value">ML / CLASSIFIER v1.0</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- PREPARE INPUT --------------------
input_data = pd.DataFrame([{
    "altitude": altitude,
    "airspeed": airspeed,
    "vertical_speed": vertical_speed,
    "pitch": pitch,
    "visibility": visibility,
    "temperature": temperature
}])

# -------------------- PREDICTION --------------------
st.markdown('<div class="section-label">03 / SAFETY ASSESSMENT</div>', unsafe_allow_html=True)

if st.button("RUN FLIGHT SAFETY ANALYSIS", use_container_width=True):

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    safe_probability = probabilities[0] * 100
    unsafe_probability = probabilities[1] * 100

    st.markdown("---")

    if prediction == 0:
        st.success("FLIGHT STATUS: SAFE")
    else:
        st.error("FLIGHT STATUS: UNSAFE")

    r1, r2 = st.columns(2)

    with r1:
        st.metric("Safe Probability", f"{safe_probability:.2f}%")

    with r2:
        st.metric("Unsafe Probability", f"{unsafe_probability:.2f}%")

    chart_data = pd.DataFrame({
        "Category": ["Safe", "Unsafe"],
        "Probability (%)": [safe_probability, unsafe_probability]
    })

    st.subheader("Prediction Confidence Distribution")
    st.bar_chart(chart_data.set_index("Category"))

    st.progress(float(probabilities[1]))

    st.caption("Model estimates are based on synthetic training data.")

# -------------------- DATA TABLE --------------------
st.markdown('<div class="section-label">04 / TELEMETRY RECORD</div>', unsafe_allow_html=True)

st.dataframe(input_data, use_container_width=True, hide_index=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
AEROSAFE AI / FLIGHT INTELLIGENCE SYSTEM &nbsp; • &nbsp;
ACADEMIC PROTOTYPE / NOT FOR REAL FLIGHT OPERATIONS
</div>
""", unsafe_allow_html=True)
