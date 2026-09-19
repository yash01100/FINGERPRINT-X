import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ============================================================
# FINGERPRINT-X
# Premium AI Biometric Research Dashboard
# ============================================================

MODEL_PATH = "fingerprint_gender_model_v2.keras"
IMG_SIZE = (128, 128)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FINGERPRINT-X | AI Biometric Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(0, 180, 255, 0.08), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(120, 70, 255, 0.08), transparent 28%),
        #07090d;
    color: #f5f7fa;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Remove Streamlit decoration */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================================================
   HEADER
   ========================================================= */

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 5px;
}

.logo {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #00c6ff, #7b61ff);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    box-shadow: 0 0 30px rgba(0, 198, 255, 0.25);
}

.brand-name {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: 2px;
}

.brand-sub {
    color: #8d96a5;
    font-size: 12px;
    letter-spacing: 1.5px;
}

.hero {
    padding: 35px 0 25px 0;
}

.hero h1 {
    font-size: 52px;
    line-height: 1.05;
    margin: 0;
    font-weight: 800;
    letter-spacing: -2px;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #ffffff,
        #69d7ff,
        #9a82ff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #929aa8;
    font-size: 17px;
    max-width: 720px;
    line-height: 1.7;
}

/* =========================================================
   STATUS BAR
   ========================================================= */

.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 16px;
    border: 1px solid #202631;
    background: rgba(15, 18, 24, 0.75);
    border-radius: 12px;
    margin: 15px 0 28px 0;
}

.status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 12px #22c55e;
}

.status-text {
    color: #aab3c2;
    font-size: 13px;
}

.status-right {
    margin-left: auto;
    color: #6e7888;
    font-size: 12px;
}

/* =========================================================
   CARDS
   ========================================================= */

.glass-card {
    background: linear-gradient(
        145deg,
        rgba(22, 27, 36, 0.95),
        rgba(12, 15, 21, 0.95)
    );
    border: 1px solid #252c38;
    border-radius: 20px;
    padding: 25px;
    box-shadow:
        0 15px 50px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.02);
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 5px;
}

.card-subtitle {
    color: #737d8d;
    font-size: 13px;
    margin-bottom: 20px;
}

/* =========================================================
   UPLOAD
   ========================================================= */

.upload-zone {
    border: 1px dashed #3b4658;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    background: rgba(10, 13, 18, 0.7);
    margin-bottom: 15px;
}

.upload-icon {
    font-size: 40px;
    margin-bottom: 8px;
}

.upload-title {
    font-size: 17px;
    font-weight: 700;
}

.upload-text {
    color: #727d8d;
    font-size: 13px;
    margin-top: 6px;
}

/* =========================================================
   RESULT
   ========================================================= */

.result-card {
    border-radius: 20px;
    padding: 28px;
    background:
        radial-gradient(
            circle at 80% 20%,
            rgba(0, 198, 255, 0.10),
            transparent 35%
        ),
        #10151d;
    border: 1px solid #293443;
}

.result-label {
    color: #778294;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-value {
    font-size: 43px;
    font-weight: 800;
    margin: 8px 0;
}

.result-description {
    color: #8791a1;
    font-size: 13px;
}

/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    background: #10141b;
    border: 1px solid #242c38;
    border-radius: 16px;
    padding: 20px;
    min-height: 130px;
}

.metric-icon {
    font-size: 24px;
}

.metric-title {
    color: #7f8999;
    font-size: 12px;
    margin-top: 10px;
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
}

/* =========================================================
   SECTION
   ========================================================= */

.section-heading {
    font-size: 25px;
    font-weight: 800;
    margin: 35px 0 17px 0;
}

.section-description {
    color: #737d8d;
    margin-top: -12px;
    margin-bottom: 20px;
}

/* =========================================================
   AI INFO
   ========================================================= */

.info-row {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #202631;
    padding: 13px 0;
}

.info-row:last-child {
    border-bottom: none;
}

.info-name {
    color: #7f8999;
}

.info-value {
    color: #e7ebf1;
    font-weight: 600;
}

/* =========================================================
   DISCLAIMER
   ========================================================= */

.disclaimer {
    background: rgba(128, 94, 0, 0.10);
    border: 1px solid rgba(201, 157, 30, 0.28);
    border-radius: 15px;
    padding: 18px;
    color: #b9b09a;
    font-size: 12px;
    line-height: 1.7;
}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    padding-top: 35px;
    color: #515b6a;
    font-size: 12px;
    line-height: 1.8;
}

/* =========================================================
   STREAMLIT BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #334155;
    background: linear-gradient(
        135deg,
        #111827,
        #182131
    );
    color: white;
    font-weight: 700;
    padding: 13px;
    transition: 0.2s;
}

.stButton > button:hover {
    border-color: #00c6ff;
    box-shadow: 0 0 25px rgba(0,198,255,0.15);
}

/* Upload button */

[data-testid="stFileUploader"] {
    background: transparent;
}

/* Progress */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #00c6ff,
        #7b61ff
    );
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="brand">
    <div class="logo">🧬</div>
    <div>
        <div class="brand-name">FINGERPRINT-X</div>
        <div class="brand-sub">AI BIOMETRIC INTELLIGENCE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STATUS
# ============================================================

st.markdown("""
<div class="status-bar">
    <div class="status-dot"></div>
    <div class="status-text">AI ENGINE ONLINE</div>
    <div class="status-right">RESEARCH PROTOTYPE • V2 MODEL</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>
        Turn a fingerprint into<br>
        <span class="gradient-text">AI-driven biometric insights.</span>
    </h1>

    <p>
        FINGERPRINT-X is an experimental biometric intelligence
        platform designed to explore how deep learning can extract
        meaningful patterns from fingerprint images.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(MODEL_PATH)


try:

    model = load_model()

    model_status = True

except Exception as e:

    model_status = False

    st.error("AI model could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">🧠</div>
        <div class="metric-title">AI ENGINE</div>
        <div class="metric-value">CNN V2</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">🖼️</div>
        <div class="metric-title">INPUT SIZE</div>
        <div class="metric-value">128 × 128</div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-title">PROCESSING</div>
        <div class="metric-value">REAL-TIME</div>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">🔬</div>
        <div class="metric-title">MODE</div>
        <div class="metric-value">RESEARCH</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">Fingerprint Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload a fingerprint image and run the experimental AI model.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1], gap="large")


# ============================================================
# LEFT - UPLOAD
# ============================================================

with left:

    st.markdown("""
    <div class="glass-card">

        <div class="card-title">🧬 Fingerprint Input</div>

        <div class="card-subtitle">
            Supported formats: PNG, JPG, JPEG, BMP
        </div>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload fingerprint",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:

        st.markdown("""
        <div class="upload-zone">

            <div class="upload-icon">📤</div>

            <div class="upload-title">
                Upload Fingerprint Image
            </div>

            <div class="upload-text">
                Select a clear fingerprint image for analysis
            </div>

        </div>
        """, unsafe_allow_html=True)

    else:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Fingerprint Input",
            use_container_width=True
        )

        st.markdown(
            f"""
            <div class="info-row">
                <span class="info-name">File</span>
                <span class="info-value">{uploaded_file.name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        analyze = st.button(
            "🔬  RUN AI ANALYSIS",
            use_container_width=True
        )


# ============================================================
# RIGHT - LIVE SYSTEM INFO
# ============================================================

with right:

    st.markdown("""
    <div class="glass-card">

        <div class="card-title">⚙️ AI System</div>

        <div class="card-subtitle">
            Current inference configuration
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">

        <div class="info-row">
            <span class="info-name">Model</span>
            <span class="info-value">CNN V2</span>
        </div>

        <div class="info-row">
            <span class="info-name">Input</span>
            <span class="info-value">Grayscale</span>
        </div>

        <div class="info-row">
            <span class="info-name">Resolution</span>
            <span class="info-value">128 × 128</span>
        </div>

        <div class="info-row">
            <span class="info-name">Inference</span>
            <span class="info-value">Live</span>
        </div>

        <div class="info-row">
            <span class="info-name">Status</span>
            <span class="info-value">● Online</span>
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None and "analyze" in locals() and analyze:

    with st.spinner("Running biometric AI analysis..."):

        time.sleep(0.8)

        image = Image.open(uploaded_file).convert("L")

        image = image.resize(IMG_SIZE)

        img_array = np.array(image)

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=-1
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        probability = float(
            model.predict(
                img_array,
                verbose=0
            )[0][0]
        )

        if probability >= 0.5:

            prediction = "MALE"

            confidence = probability * 100

        else:

            prediction = "FEMALE"

            confidence = (1 - probability) * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-heading">AI Analysis Result</div>',
        unsafe_allow_html=True
    )

    result_left, result_right = st.columns([1.25, 1], gap="large")


    with result_left:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Experimental Prediction
                </div>

                <div class="result-value">
                    {prediction}
                </div>

                <div class="result-description">
                    The CNN model generated this experimental
                    classification from the uploaded fingerprint.
                </div>

                <br>

                <div class="result-label">
                    Model Probability
                </div>

                <div style="
                    font-size:26px;
                    font-weight:700;
                    margin-top:7px;
                ">
                    {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            min(int(confidence), 100)
        )


    with result_right:

        st.markdown("""
        <div class="glass-card">

            <div class="card-title">
                📊 Analysis Summary
            </div>

            <div class="info-row">
                <span class="info-name">Prediction</span>
                <span class="info-value">
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
                    {prediction}
                </span>
            </div>

            <div class="info-row">
                <span class="info-name">Probability</span>
                <span class="info-value">
                    {confidence:.2f}%
                </span>
            </div>

            <div class="info-row">
                <span class="info-name">Model</span>
                <span class="info-value">
                    CNN V2
                </span>
            </div>

            <div class="info-row">
                <span class="info-name">Input</span>
                <span class="info-value">
                    128 × 128
                </span>
            </div>

            <div class="info-row">
                <span class="info-name">Status</span>
                <span class="info-value">
                    Analysis Complete
                </span>
            </div>

        </div>
        """,
            unsafe_allow_html=True
        )


    # ========================================================
    # RESEARCH PROFILE
    # ========================================================

    st.markdown(
        '<div class="section-heading">Biometric Research Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Additional physiological inference modules are currently under research.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-icon">🎂</div>

            <div class="metric-title">
                AGE ESTIMATION
            </div>

            <div class="metric-value">
                R&D
            </div>

            <div style="color:#687384;font-size:11px;margin-top:6px;">
                Research module
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-icon">🩸</div>

            <div class="metric-title">
                BLOOD GROUP
            </div>

            <div class="metric-value">
                R&D
            </div>

            <div style="color:#687384;font-size:11px;margin-top:6px;">
                Research module
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-icon">⚖️</div>

            <div class="metric-title">
                WEIGHT PROFILE
            </div>

            <div class="metric-value">
                R&D
            </div>

            <div style="color:#687384;font-size:11px;margin-top:6px;">
                Research module
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    '<div class="section-heading">Research Notice</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer">

<b>FINGERPRINT-X is an experimental research prototype.</b><br><br>

The current CNN model performs an experimental fingerprint
classification task. Its output is not a medically validated
measurement and should not be used for diagnosis, identification,
or health-related decision making.

Age, blood-group and weight modules shown in this interface
represent future research directions and are not currently
generated by the model.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<b>FINGERPRINT-X</b><br>

AI Biometric Intelligence • Research Prototype<br><br>

CNN V2 • Experimental Inference Engine

</div>
""", unsafe_allow_html=True)