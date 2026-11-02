
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit.components.v1 as components
import joblib

from pathlib import Path
from datetime import datetime, timedelta

# =====================================================
# CONFIGURATION
# =====================================================

API_URL = "http://127.0.0.1:8000"

LIVE_REFRESH_SECONDS = 0.5
GRAPH_REFRESH_SECONDS = 3.0

TEMP_ATTENTION = 60.0
TEMP_WARNING = 75.0
TEMP_CRITICAL = 90.0

VIBRATION_BASELINE_G = 0.05

VIBRATION_ATTENTION_MULTIPLIER = 1.5
VIBRATION_WARNING_MULTIPLIER = 2.0
VIBRATION_CRITICAL_MULTIPLIER = 3.0

VIBRATION_ATTENTION = VIBRATION_BASELINE_G * VIBRATION_ATTENTION_MULTIPLIER
VIBRATION_WARNING = VIBRATION_BASELINE_G * VIBRATION_WARNING_MULTIPLIER
VIBRATION_CRITICAL = VIBRATION_BASELINE_G * VIBRATION_CRITICAL_MULTIPLIER

CONTACTOR_INSTALLED = False

MODEL_DIR = Path("models/prototype")

MODEL_LOW = 0.40
MODEL_HIGH = 0.70

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Industrial Motor | Predictive Maintenance",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# ANIMATED BACKGROUND AND DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #050b18 !important;
    color: #edf5ff;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: rgba(5, 11, 24, 0.55) !important;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: -50%;
    width: 200%;
    height: 200%;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(ellipse at 20% 35%, rgba(0,120,255,.55) 0%, transparent 38%),
        radial-gradient(ellipse at 75% 55%, rgba(0,230,255,.42) 0%, transparent 38%),
        radial-gradient(ellipse at 50% 85%, rgba(65,85,255,.38) 0%, transparent 42%),
        radial-gradient(ellipse at 85% 15%, rgba(0,170,255,.30) 0%, transparent 35%);
    filter: blur(45px);
    animation: backgroundFlow 8s ease-in-out infinite alternate;
    will-change: transform;
}

.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background: repeating-linear-gradient(
        120deg,
        transparent 0px,
        transparent 90px,
        rgba(65,180,255,.045) 91px,
        transparent 94px
    );
    opacity: .85;
}

@keyframes backgroundFlow {
    0% { transform: translate(-7%,-5%) rotate(-5deg) scale(1); }
    30% { transform: translate(5%,5%) rotate(5deg) scale(1.12); }
    65% { transform: translate(-4%,9%) rotate(-4deg) scale(1.08); }
    100% { transform: translate(8%,-5%) rotate(7deg) scale(1.16); }
}

[data-testid="stMainBlockContainer"] {
    position: relative;
    z-index: 1;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1 {
    color: #f4f8ff !important;
    font-size: clamp(1.7rem, 3vw, 2.5rem) !important;
    font-weight: 750 !important;
    letter-spacing: -1px;
    line-height: 1.2 !important;
}

h2, h3 {
    color: #e7f2ff !important;
    font-weight: 650 !important;
}

p, label {
    color: #a9bfdc;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(19,42,73,.88), rgba(9,24,46,.88));
    backdrop-filter: blur(18px);
    border: 1px solid rgba(79,170,255,.30);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 12px 35px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.05);
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(0,210,255,.65);
    box-shadow: 0 15px 40px rgba(0,120,255,.15);
}

[data-testid="stMetricLabel"] {
    color: #9db8d8 !important;
    font-size: .9rem !important;
}

[data-testid="stMetricValue"] {
    color: #fff !important;
    font-weight: 750 !important;
}

[data-testid="stAlert"] {
    border-radius: 14px;
    backdrop-filter: blur(12px);
}

hr {
    border-color: rgba(105,170,230,.20) !important;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(9,24,45,.70);
    border: 1px solid rgba(79,170,255,.20);
    border-radius: 18px;
    padding: 10px;
}

.small-footer {
    color: #7794b7;
    text-align: center;
    font-size: .8rem;
    padding: 20px 0;
}

@media (prefers-reduced-motion: reduce) {
    .stApp::before { animation: none !important; }
    [data-testid="stMetric"] { transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HTTP SESSION
# =====================================================

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    return session


session = get_http_session()


def get_api_data(endpoint, timeout=2):
    response = session.get(API_URL + endpoint, timeout=timeout)
    response.raise_for_status()
    return response.json()


# =====================================================
# LOAD AI MODELS
# =====================================================

@st.cache_resource
def load_ai_models():
    heat_model = joblib.load(
        MODEL_DIR / "early_heat_detector.joblib"
    )

    abnormal_model = joblib.load(
        MODEL_DIR / "abnormal_loss_indicator.joblib"
    )

    return heat_model, abnormal_model


try:
    heat_model, abnormal_model = load_ai_models()
    AI_MODELS_READY = True
    AI_MODEL_ERROR = None

except Exception as error:
    heat_model = None
    abnormal_model = None
    AI_MODELS_READY = False
    AI_MODEL_ERROR = str(error)


# =====================================================
# AI PREDICTION FUNCTION
# =====================================================

def get_ai_predictions():

    if not AI_MODELS_READY:
        return None, None, None, f"AI models unavailable: {AI_MODEL_ERROR}"

    try:
        data = get_api_data("/history", timeout=3)
        readings = data.get("readings", [])

        if len(readings) < 21:
            return None, None, None, "Collect at least 21 readings."

        df = pd.DataFrame(readings)

        required_columns = [
            "id",
            "temperature_c",
            "vibration_rms_g"
        ]

        if not all(column in df.columns for column in required_columns):
            return None, None, None, "Required sensor fields are missing."

        df = df.sort_values("id").reset_index(drop=True)

        df["temp_change_30s"] = df["temperature_c"].diff(10)
        df["temp_change_60s"] = df["temperature_c"].diff(20)

        df["vibration_mean_30s"] = (
            df["vibration_rms_g"].rolling(10, min_periods=1).mean()
        )

        df["vibration_std_30s"] = (
            df["vibration_rms_g"].rolling(10, min_periods=2).std()
        )

        features = [
            "temperature_c",
            "temp_change_30s",
            "temp_change_60s",
            "vibration_rms_g",
            "vibration_mean_30s",
            "vibration_std_30s"
        ]

        latest = df[features].iloc[[-1]]

        if latest.isna().any().any():
            return None, None, None, "Insufficient valid history for prediction."

        heat_score = heat_model.predict_proba(latest)[0][1]
        abnormal_score = abnormal_model.predict_proba(latest)[0][1]

        temperature_change = float(
            df["temperature_c"].iloc[-1] -
            df["temperature_c"].iloc[-11]
        )

        return (
            float(heat_score),
            float(abnormal_score),
            temperature_change,
            None
        )

    except Exception as error:
        return None, None, None, str(error)


# =====================================================
# HUMAN-READABLE AI INDICATORS
# =====================================================

def classify_heat_pattern(score):

    if score is None:
        return "Unavailable", "AI prediction unavailable."

    if score >= MODEL_HIGH:
        return (
            "Heating signal",
            "Readings strongly resemble learned heating patterns."
        )

    if score >= MODEL_LOW:
        return (
            "Possible heating",
            "Some readings resemble learned heating patterns."
        )

    return (
        "Low heating signal",
        "Few characteristics resemble learned heating patterns."
    )


def classify_abnormal_pattern(score):

    if score is None:
        return "Unavailable", "AI prediction unavailable."

    if score >= MODEL_HIGH:
        return (
            "Unusual pattern",
            "Readings strongly resemble learned abnormal patterns."
        )

    if score >= MODEL_LOW:
        return (
            "Possible unusual pattern",
            "Some readings resemble learned abnormal patterns."
        )

    return (
        "Normal-like pattern",
        "Few characteristics resemble learned abnormal patterns."
    )


def classify_temperature_trend(change):

    if change is None:
        return "Unavailable", "Temperature trend unavailable."

    if change >= 1.0:
        return (
            "Rising",
            f"Up {change:.2f} °C over 10 readings."
        )

    if change <= -1.0:
        return (
            "Falling",
            f"Down {abs(change):.2f} °C over 10 readings."
        )

    return (
        "Stable",
        f"Change: {change:+.2f} °C over 10 readings."
    )


# =====================================================
# CONDITION ANALYSIS
# =====================================================

def classify_temperature(temp):

    if temp >= TEMP_CRITICAL:
        return "CRITICAL"
    if temp >= TEMP_WARNING:
        return "WARNING"
    if temp >= TEMP_ATTENTION:
        return "ATTENTION"

    return "HEALTHY"


def classify_vibration(vibration):

    if vibration >= VIBRATION_CRITICAL:
        return "CRITICAL"
    if vibration >= VIBRATION_WARNING:
        return "WARNING"
    if vibration >= VIBRATION_ATTENTION:
        return "ATTENTION"

    return "HEALTHY"


def calculate_motor_condition(temp, vibration):

    if temp is None or vibration is None:
        return "SENSOR FAULT"

    temp_condition = classify_temperature(temp)
    vibration_condition = classify_vibration(vibration)

    priority = {
        "HEALTHY": 0,
        "ATTENTION": 1,
        "WARNING": 2,
        "CRITICAL": 3
    }

    if priority[temp_condition] >= priority[vibration_condition]:
        return temp_condition

    return vibration_condition


def get_motor_explanation(temp, vibration):

    explanations = []

    if temp >= TEMP_CRITICAL:
        explanations.append(
            "Critical winding temperature. Stop the motor and inspect "
            "the cooling system, load and winding condition."
        )

    elif temp >= TEMP_WARNING:
        explanations.append(
            "High winding temperature. Possible overload, poor cooling "
            "or increased motor losses."
        )

    elif temp >= TEMP_ATTENTION:
        explanations.append(
            "Winding temperature is rising. Continue monitoring and "
            "check the motor load and ventilation."
        )

    if vibration >= VIBRATION_CRITICAL:
        explanations.append(
            "Critical vibration relative to the configured baseline. "
            "Possible severe imbalance, misalignment or looseness."
        )

    elif vibration >= VIBRATION_WARNING:
        explanations.append(
            "High vibration relative to the configured baseline. "
            "Inspect alignment, mounting and bearings."
        )

    elif vibration >= VIBRATION_ATTENTION:
        explanations.append(
            "Vibration is above its configured baseline. Monitor for "
            "mechanical changes."
        )

    if not explanations:
        explanations.append(
            "Temperature and vibration are currently below the "
            "configured attention thresholds."
        )

    return explanations


def display_condition(condition):

    if condition == "HEALTHY":
        st.success("MOTOR HEALTHY")
    elif condition == "ATTENTION":
        st.warning("MOTOR NEEDS ATTENTION")
    elif condition == "WARNING":
        st.warning("MOTOR WARNING")
    elif condition == "CRITICAL":
        st.error("CRITICAL MOTOR CONDITION")
    else:
        st.error("SENSOR FAULT - CONDITION UNKNOWN")


def display_shutdown_status(condition):

    if condition == "CRITICAL":

        st.error(
            "CRITICAL SHUTDOWN REQUEST: Motor should be stopped "
            "and inspected immediately."
        )

        if CONTACTOR_INSTALLED:
            st.warning(
                "Verify the independent local trip circuit."
            )
        else:
            st.error(
                "MOTOR NOT AUTOMATICALLY SWITCHED OFF. "
                "This dashboard cannot disconnect the motor supply."
            )

    elif condition == "SENSOR FAULT":

        st.error(
            "Protection status unavailable. Check sensor wiring. "
            "Do not treat the motor as healthy."
        )

    else:
        st.info(
            "No critical shutdown request at this time. "
            "Physical motor protection is not configured."
        )


# =====================================================
# HEADER
# =====================================================

st.title(
    "AI & IoT-Enabled Early Heat and Loss Prediction "
    "System for Predictive Maintenance of Industrial "
    "Motor Windings"
)

st.divider()

# =====================================================
# MOTOR CONFIGURATION
# =====================================================

st.subheader("MOTOR CONFIGURATION")

config_col1, config_col2 = st.columns(2)

with config_col1:
    st.metric("Motor Rating", "0.75 kW / 1 HP")

with config_col2:
    st.metric("Insulation Class", "F")

with st.expander("View configured monitoring thresholds"):

    st.markdown("**Winding temperature**")
    st.write(f"Healthy: Below {TEMP_ATTENTION:.1f} °C")
    st.write(
        f"Attention: {TEMP_ATTENTION:.1f} °C to "
        f"{TEMP_WARNING - 0.1:.1f} °C"
    )
    st.write(
        f"Warning: {TEMP_WARNING:.1f} °C to "
        f"{TEMP_CRITICAL - 0.1:.1f} °C"
    )
    st.write(f"Critical: {TEMP_CRITICAL:.1f} °C and above")

    st.divider()

    st.markdown("**Vibration RMS**")
    st.write(f"Configured baseline: {VIBRATION_BASELINE_G:.3f} g")
    st.write(f"Attention: {VIBRATION_ATTENTION:.3f} g and above")
    st.write(f"Warning: {VIBRATION_WARNING:.3f} g and above")
    st.write(f"Critical: {VIBRATION_CRITICAL:.3f} g and above")

    st.warning(
        "The vibration baseline is a placeholder. Validate it using "
        "normal motor operating measurements before practical use."
    )

st.divider()

# =====================================================
# MONITORING MODE
# =====================================================

st.subheader("MONITORING MODE")

mode = st.radio(
    "Select monitoring mode",
    ["LIVE", "TEST", "ABOUT"],
    horizontal=True,
    key="monitoring_mode"
)


# =====================================================
# TEST MODE
# =====================================================

if mode == "TEST":

    st.subheader("TEST CONTROL PANEL")

    st.info(
        "Move the sliders to simulate winding temperature and vibration. "
        "This mode does not operate physical hardware."
    )

    test_dashboard = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            padding: 4px 0 12px;
            background: transparent;
            color: #edf5ff;
            font-family: Arial, sans-serif;
        }}

        .slider-grid, .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0,1fr));
            gap: 16px;
            margin-bottom: 18px;
        }}

        .metric-grid {{
            grid-template-columns: repeat(3, minmax(0,1fr));
        }}

        .panel {{
            background: linear-gradient(145deg,rgba(19,42,73,.96),rgba(9,24,46,.96));
            border: 1px solid rgba(79,170,255,.30);
            border-radius: 18px;
            padding: 22px;
            min-width: 0;
        }}

        .panel-title {{
            color: #e7f2ff;
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 18px;
        }}

        .label {{
            color: #9db8d8;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .slider-value {{
            color: #20cfff;
            float: right;
            font-size: 16px;
            font-weight: 800;
        }}

        input[type=range] {{
            width: 100%;
            margin: 24px 0 12px;
            cursor: pointer;
            accent-color: #20cfff;
        }}

        .range-limits {{
            display: flex;
            justify-content: space-between;
            color: #7794b7;
            font-size: 12px;
        }}

        .metric-label {{
            color: #9db8d8;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            min-height: 30px;
        }}

        .metric-value {{
            color: #fff;
            font-size: clamp(22px,3vw,32px);
            font-weight: 800;
            margin-top: 10px;
            overflow-wrap: anywhere;
        }}

        .unit {{
            color: #9db8d8;
            font-size: 13px;
            font-weight: 500;
        }}

        .condition-pill {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 12px;
            border-radius: 9px;
            font-size: 13px;
            font-weight: 800;
        }}

        .healthy {{
            color: #4ade80;
            background: rgba(22,101,52,.22);
            border: 1px solid rgba(74,222,128,.45);
        }}

        .attention {{
            color: #facc15;
            background: rgba(133,77,14,.22);
            border: 1px solid rgba(250,204,21,.45);
        }}

        .warning {{
            color: #fb923c;
            background: rgba(154,52,18,.22);
            border: 1px solid rgba(251,146,60,.45);
        }}

        .critical {{
            color: #f87171;
            background: rgba(153,27,27,.22);
            border: 1px solid rgba(248,113,113,.50);
        }}

        .alert {{
            border-radius: 14px;
            padding: 17px 19px;
            font-size: 14px;
            line-height: 1.5;
            font-weight: 700;
            border: 1px solid;
            margin-top: 4px;
        }}

        .hint {{
            color: #7794b7;
            font-size: 12px;
            margin-top: 15px;
            line-height: 1.5;
        }}

        @media(max-width:650px) {{
            .slider-grid, .metric-grid {{ grid-template-columns: 1fr; }}
            .panel {{ padding: 18px; }}
        }}
    </style>
    </head>

    <body>

        <div class="slider-grid">

            <div class="panel">
                <div class="panel-title">Winding Temperature</div>
                <div class="label">
                    Temperature
                    <span class="slider-value" id="tempValue">35.0 °C</span>
                </div>
                <input id="tempSlider" type="range"
                    min="20" max="120" step="0.5" value="35">
                <div class="range-limits">
                    <span>20 °C</span><span>120 °C</span>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Motor Vibration</div>
                <div class="label">
                    Vibration RMS
                    <span class="slider-value" id="vibValue">0.05 g</span>
                </div>
                <input id="vibSlider" type="range"
                    min="0" max="1" step="0.01" value="0.05">
                <div class="range-limits">
                    <span>0.00 g</span><span>1.00 g</span>
                </div>
            </div>

        </div>

        <div class="metric-grid">

            <div class="panel">
                <div class="metric-label">Winding Temperature</div>
                <div class="metric-value">
                    <span id="tempMetric">35.00</span>
                    <span class="unit">°C</span>
                </div>
            </div>

            <div class="panel">
                <div class="metric-label">Vibration RMS</div>
                <div class="metric-value">
                    <span id="vibMetric">0.0500</span>
                    <span class="unit">g</span>
                </div>
            </div>

            <div class="panel">
                <div class="metric-label">Motor Condition</div>
                <div id="condition" class="condition-pill healthy">HEALTHY</div>
            </div>

        </div>

        <div id="explanation" class="alert healthy">
            Motor is operating below the configured thresholds.
        </div>

        <div id="shutdown" class="alert healthy">
            No critical shutdown request.
        </div>

        <div class="hint">
            TEST MODE — Simulated values only. No physical motor,
            relay or contactor is operated.
        </div>

    <script>
        const tempSlider = document.getElementById("tempSlider");
        const vibSlider = document.getElementById("vibSlider");

        function updateDashboard() {{

            const temp = Number(tempSlider.value);
            const vibration = Number(vibSlider.value);

            document.getElementById("tempValue").textContent =
                temp.toFixed(1) + " °C";

            document.getElementById("vibValue").textContent =
                vibration.toFixed(2) + " g";

            document.getElementById("tempMetric").textContent =
                temp.toFixed(2);

            document.getElementById("vibMetric").textContent =
                vibration.toFixed(4);

            let tempCondition = "HEALTHY";
            let vibCondition = "HEALTHY";

            if (temp >= {TEMP_CRITICAL}) {{
                tempCondition = "CRITICAL";
            }} else if (temp >= {TEMP_WARNING}) {{
                tempCondition = "WARNING";
            }} else if (temp >= {TEMP_ATTENTION}) {{
                tempCondition = "ATTENTION";
            }}

            if (vibration >= {VIBRATION_CRITICAL}) {{
                vibCondition = "CRITICAL";
            }} else if (vibration >= {VIBRATION_WARNING}) {{
                vibCondition = "WARNING";
            }} else if (vibration >= {VIBRATION_ATTENTION}) {{
                vibCondition = "ATTENTION";
            }}

            const priority = {{
                "HEALTHY": 0,
                "ATTENTION": 1,
                "WARNING": 2,
                "CRITICAL": 3
            }};

            let condition =
                priority[tempCondition] >= priority[vibCondition]
                ? tempCondition : vibCondition;

            const cssClass = condition.toLowerCase();

            const conditionElement = document.getElementById("condition");
            conditionElement.textContent = condition;
            conditionElement.className = "condition-pill " + cssClass;

            let messages = [];

            if (temp >= {TEMP_CRITICAL}) {{
                messages.push("Critical winding temperature. Stop and inspect the motor.");
            }} else if (temp >= {TEMP_WARNING}) {{
                messages.push("High winding temperature. Check load and cooling.");
            }} else if (temp >= {TEMP_ATTENTION}) {{
                messages.push("Winding temperature is rising. Monitor closely.");
            }}

            if (vibration >= {VIBRATION_CRITICAL}) {{
                messages.push("Critical vibration relative to configured baseline. Inspect mechanical condition.");
            }} else if (vibration >= {VIBRATION_WARNING}) {{
                messages.push("High vibration relative to baseline. Check alignment and mounting.");
            }} else if (vibration >= {VIBRATION_ATTENTION}) {{
                messages.push("Vibration is above its configured baseline.");
            }}

            if (messages.length === 0) {{
                messages.push("Temperature and vibration are below configured attention thresholds.");
            }}

            const explanation = document.getElementById("explanation");
            explanation.textContent = messages.join(" ");
            explanation.className = "alert " + cssClass;

            const shutdown = document.getElementById("shutdown");

            if (condition === "CRITICAL") {{
                shutdown.textContent =
                    "CRITICAL SHUTDOWN REQUEST — Stop the motor. " +
                    "No contactor is installed, so this dashboard cannot " +
                    "physically disconnect the supply.";
                shutdown.className = "alert critical";
            }} else {{
                shutdown.textContent =
                    "No critical shutdown request. This is a simulation only.";
                shutdown.className = "alert healthy";
            }}
        }}

        tempSlider.addEventListener("input", updateDashboard);
        vibSlider.addEventListener("input", updateDashboard);
        updateDashboard();
    </script>

    </body>
    </html>
    """

    components.html(test_dashboard, height=570, scrolling=False)

    st.caption(
        "TEST MODE uses provisional thresholds. It does not control "
        "physical hardware or run the AI models."
    )


# =====================================================
# ABOUT MODE
# =====================================================

elif mode == "ABOUT":

    st.subheader("ABOUT MOTORAI")

    st.markdown("""
    <div style="
        background: linear-gradient(145deg, rgba(19,42,73,.88), rgba(9,24,46,.88));
        border: 1px solid rgba(79,170,255,.30);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
    ">
        <h3 style="color:#20cfff;">MotorAI – Predictive Maintenance System</h3>
        <p style="line-height:1.8;">
            MotorAI is an AI and IoT-enabled motor monitoring system
            developed to monitor induction motor winding temperature
            and vibration behaviour. It uses real-time sensor readings
            and machine learning models to identify early heating
            patterns and unusual operating conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------
    # PROJECT OBJECTIVES
    # -------------------------------------------------

    st.markdown("### PROJECT OBJECTIVES")

    objective_col1, objective_col2 = st.columns(2)

    with objective_col1:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
            margin-bottom:15px;
        ">
            <h3 style="color:#20cfff;">Temperature Monitoring</h3>
            <p>
                Continuously monitor motor winding temperature
                and identify abnormal temperature rise.
            </p>
        </div>

        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
        ">
            <h3 style="color:#20cfff;">AI-Based Prediction</h3>
            <p>
                Use machine learning to identify early heating
                behaviour and abnormal operating patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with objective_col2:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
            margin-bottom:15px;
        ">
            <h3 style="color:#20cfff;">Vibration Monitoring</h3>
            <p>
                Monitor motor vibration to help identify
                changes in mechanical operating behaviour.
            </p>
        </div>

        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
        ">
            <h3 style="color:#20cfff;">Live Dashboard</h3>
            <p>
                Display sensor readings, operating conditions,
                historical graphs and AI analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # HARDWARE AND TECHNOLOGIES
    # -------------------------------------------------

    st.markdown("### HARDWARE & TECHNOLOGIES")

    hardware_col1, hardware_col2 = st.columns(2)

    with hardware_col1:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
            margin-bottom:15px;
        ">
            <h3 style="color:#20cfff;">Hardware Components</h3>
            <p>Raspberry Pi Pico W</p>
            <p>DS18B20 Temperature Sensor</p>
            <p>MPU6050 Accelerometer and Gyroscope</p>
        </div>
        """, unsafe_allow_html=True)

    with hardware_col2:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
            margin-bottom:15px;
        ">
            <h3 style="color:#20cfff;">Software & Technologies</h3>
            <p>Python</p>
            <p>Streamlit</p>
            <p>Machine Learning</p>
            <p>SQLite Database</p>
            <p>IoT Communication</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # MOTOR SPECIFICATIONS
    # -------------------------------------------------

    st.markdown("### MOTOR SPECIFICATIONS")

    motor_col1, motor_col2 = st.columns(2)

    with motor_col1:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
        ">
            <h3 style="color:#20cfff;">Motor Details</h3>
            <p>Motor Type: Three-Phase Induction Motor</p>
            <p>Rated Power: 0.75 kW / 1 HP</p>
            <p>Rated Voltage: 415 V ±10%</p>
            <p>Frequency: 50 Hz ±5%</p>
        </div>
        """, unsafe_allow_html=True)

    with motor_col2:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
        ">
            <h3 style="color:#20cfff;">Operating Details</h3>
            <p>Rated Current: 1.70 A (Star)</p>
            <p>Rated Speed: 1390 RPM</p>
            <p>Efficiency: 79.60%</p>
            <p>Power Factor: 0.77</p>
            <p>Insulation Class: F</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # PROJECT TEAM
    # -------------------------------------------------

    st.markdown("### PROJECT TEAM")

    team_col1, team_col2 = st.columns(2)

    with team_col1:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
        ">
            <h3 style="color:#20cfff;">Project Members</h3>
            <p>Vinayak Sharma</p>
            <p>Sakshya Tomar</p>
        </div>
        """, unsafe_allow_html=True)

    with team_col2:

        st.markdown("""
        <div style="
            background:rgba(19,42,73,.75);
            border:1px solid rgba(79,170,255,.25);
            border-radius:15px;
            padding:20px;
        ">
            <h3 style="color:#20cfff;">Project Guides</h3>
            <p>Dr. Dhananjay Bisen</p>
            <p>Dr. Saurabh Kumar Rajput</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # PROJECT VISION
    # -------------------------------------------------

    st.markdown("### PROJECT VISION")

    st.markdown("""
    <div style="
        background:linear-gradient(145deg, rgba(19,42,73,.88), rgba(9,24,46,.88));
        border:1px solid rgba(79,170,255,.30);
        border-radius:18px;
        padding:25px;
    ">
        <p style="line-height:1.8;">
            The vision of MotorAI is to develop an intelligent motor
            monitoring solution that can identify early heating
            behaviour and abnormal operating patterns, helping
            operators make informed maintenance decisions and
            improve motor reliability.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# LIVE MODE
# =====================================================

else:

    @st.fragment(run_every=LIVE_REFRESH_SECONDS)
    def live_monitoring():

        st.subheader("LIVE MOTOR CONDITION")

        try:
            data = get_api_data("/latest", timeout=1.5)

            if "temperature_c" not in data:
                st.warning("Waiting for sensor readings...")
                return

            temperature_value = data.get("temperature_c")
            vibration_value = data.get("vibration_rms_g")

            try:
                temperature = (
                    float(temperature_value)
                    if temperature_value is not None
                    else None
                )

                vibration = (
                    float(vibration_value)
                    if vibration_value is not None
                    else None
                )

                if temperature is not None and not pd.notna(temperature):
                    temperature = None

                if vibration is not None and (
                    not pd.notna(vibration) or vibration < 0
                ):
                    vibration = None

            except (ValueError, TypeError):
                temperature = None
                vibration = None

            if temperature is None or vibration is None:

                st.error(
                    "SENSOR FAULT: Temperature or vibration data is "
                    "missing or invalid. Motor condition is unknown."
                )

                st.metric(
                    "Winding Temperature",
                    f"{temperature:.2f} °C"
                    if temperature is not None else "Sensor fault"
                )

                st.metric(
                    "Vibration RMS",
                    f"{vibration:.4f} g"
                    if vibration is not None else "Sensor fault"
                )

                st.metric("Motor Condition", "SENSOR FAULT")

                st.error(
                    "Do not interpret missing sensor data as a healthy "
                    "motor. Check sensor wiring and readings."
                )

                return

            condition = calculate_motor_condition(
                temperature,
                vibration
            )

            display_condition(condition)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Winding Temperature",
                    value=f"{temperature:.2f} °C",
                    delta=f"{temperature - TEMP_ATTENTION:.2f} °C from attention level"
                )

            with col2:
                st.metric(
                    label="Vibration RMS",
                    value=f"{vibration:.4f} g",
                    delta=f"{vibration - VIBRATION_BASELINE_G:.4f} g from configured baseline"
                )

            with col3:
                st.metric(
                    label="Motor Condition",
                    value=condition
                )

            st.markdown("### CONDITION ANALYSIS")

            for explanation in get_motor_explanation(
                temperature,
                vibration
            ):
                st.write("- " + explanation)

            # =========================================
            # COMPACT AI INDICATORS
            # =========================================

            st.divider()
            st.markdown("### AI CONDITION ANALYSIS")

            heat_score, abnormal_score, temperature_change, ai_error = (
                get_ai_predictions()
            )

            if ai_error:
                st.warning(f"AI analysis unavailable: {ai_error}")

            else:
                heat_label, heat_explanation = classify_heat_pattern(
                    heat_score
                )

                abnormal_label, abnormal_explanation = (
                    classify_abnormal_pattern(abnormal_score)
                )

                trend_label, trend_explanation = (
                    classify_temperature_trend(temperature_change)
                )

                st.markdown("""
                <style>
                .ai-card {
                    background: linear-gradient(
                        145deg,
                        rgba(19,42,73,.88),
                        rgba(9,24,46,.88)
                    );
                    border: 1px solid rgba(79,170,255,.28);
                    border-radius: 14px;
                    padding: 14px 15px;
                    height: 145px;
                    overflow: hidden;
                    box-sizing: border-box;
                    display: flex;
                    flex-direction: column;
                }

                .ai-title {
                    color: #9db8d8;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: .5px;
                    text-transform: uppercase;
                    margin-bottom: 9px;
                }

                .ai-value {
                    color: #20cfff;
                    font-size: 16px;
                    line-height: 1.2;
                    font-weight: 750;
                    margin-bottom: 7px;
                    overflow-wrap: anywhere;
                }

                .ai-description {
                    color: #a9bfdc;
                    font-size: 10px;
                    line-height: 1.3;
                    overflow-wrap: anywhere;
                }
                </style>
                """, unsafe_allow_html=True)

                cards = [
                    (
                        "HEATING BEHAVIOUR",
                        heat_label,
                        heat_explanation
                    ),
                    (
                        "OPERATING PATTERN",
                        abnormal_label,
                        abnormal_explanation
                    ),
                    (
                        "TEMPERATURE TREND",
                        trend_label,
                        trend_explanation
                    )
                ]

                columns = st.columns(3)

                for column, (title, value, explanation) in zip(
                    columns, cards
                ):
                    with column:
                        st.markdown(
                            f"""
                            <div class="ai-card">
                                <div class="ai-title">{title}</div>
                                <div class="ai-value">{value}</div>
                                <div class="ai-description">
                                    {explanation}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            st.caption(
                "Experimental AI indicators trained on synthetic data. "
                "Real motor testing is required for validation."
            )

            display_shutdown_status(condition)

        except requests.RequestException as error:
            st.error(f"Backend connection problem: {error}")

        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Invalid sensor data: {error}")

    live_monitoring()

    st.divider()

    # =================================================
    # REAL-TIME TEMPERATURE AND VIBRATION GRAPHS
    # =================================================

    @st.fragment(run_every=GRAPH_REFRESH_SECONDS)
    def realtime_graphs():

        st.subheader("TEMPERATURE VS TIME")

        if "live_graph_data" not in st.session_state:
            st.session_state.live_graph_data = {
                "time": [],
                "temperature": [],
                "vibration": []
            }

        try:
            data = get_api_data("/latest", timeout=2)

            temperature_value = data.get("temperature_c")
            vibration_value = data.get("vibration_rms_g")

            if temperature_value is None or vibration_value is None:
                st.warning(
                    "Waiting for valid temperature and vibration readings..."
                )
                return

            temperature = float(temperature_value)
            vibration = float(vibration_value)

            if pd.isna(temperature) or pd.isna(vibration):
                st.warning("Invalid sensor readings received.")
                return

            current_time = datetime.now()
            graph_data = st.session_state.live_graph_data

            graph_data["time"].append(current_time)
            graph_data["temperature"].append(temperature)
            graph_data["vibration"].append(vibration)

            cutoff_time = current_time - timedelta(seconds=60)

            valid_indices = [
                index
                for index, timestamp in enumerate(graph_data["time"])
                if timestamp >= cutoff_time
            ]

            for key in graph_data:
                graph_data[key] = [
                    graph_data[key][index]
                    for index in valid_indices
                ]

            # TEMPERATURE GRAPH

            temperature_fig = go.Figure()

            temperature_fig.add_trace(
                go.Scatter(
                    x=graph_data["time"],
                    y=graph_data["temperature"],
                    mode="lines",
                    name="Winding Temperature",
                    line=dict(
                        color="#20cfff",
                        width=3,
                        shape="spline"
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(0,180,255,.07)"
                )
            )

            temperature_fig.add_hline(
                y=TEMP_ATTENTION,
                line_dash="dot",
                line_color="#facc15",
                annotation_text="Attention"
            )

            temperature_fig.add_hline(
                y=TEMP_WARNING,
                line_dash="dot",
                line_color="#fb923c",
                annotation_text="Warning"
            )

            temperature_fig.add_hline(
                y=TEMP_CRITICAL,
                line_dash="dash",
                line_color="#f87171",
                annotation_text="Critical"
            )

            temperature_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified"
            )

            temperature_fig.update_xaxes(
                tickformat="%H:%M:%S",
                range=[
                    current_time - timedelta(seconds=60),
                    current_time
                ]
            )

            st.plotly_chart(
                temperature_fig,
                use_container_width=True,
                key="realtime_temperature_graph"
            )

            # VIBRATION GRAPH

            st.subheader("VIBRATION VS TIME")

            vibration_fig = go.Figure()

            vibration_fig.add_trace(
                go.Scatter(
                    x=graph_data["time"],
                    y=graph_data["vibration"],
                    mode="lines",
                    name="Vibration RMS",
                    line=dict(
                        color="#9b8cff",
                        width=3,
                        shape="spline"
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(155,140,255,.07)"
                )
            )

            vibration_fig.add_hline(
                y=VIBRATION_ATTENTION,
                line_dash="dot",
                line_color="#facc15",
                annotation_text="Attention"
            )

            vibration_fig.add_hline(
                y=VIBRATION_WARNING,
                line_dash="dot",
                line_color="#fb923c",
                annotation_text="Warning"
            )

            vibration_fig.add_hline(
                y=VIBRATION_CRITICAL,
                line_dash="dash",
                line_color="#f87171",
                annotation_text="Critical"
            )

            vibration_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Time",
                yaxis_title="Vibration (g)",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified"
            )

            vibration_fig.update_xaxes(
                tickformat="%H:%M:%S",
                range=[
                    current_time - timedelta(seconds=60),
                    current_time
                ]
            )

            st.plotly_chart(
                vibration_fig,
                use_container_width=True,
                key="realtime_vibration_graph"
            )

            st.caption(
                f"Real-time monitoring | Last 60 seconds | "
                f"{len(graph_data['time'])} readings | "
                f"Updated every {GRAPH_REFRESH_SECONDS:.0f} seconds"
            )

        except requests.RequestException as error:
            st.error(f"Backend connection problem: {error}")

        except (ValueError, TypeError, KeyError) as error:
            st.error(f"Invalid sensor data: {error}")

    realtime_graphs()


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="small-footer">
    INDUSTRIAL MOTOR PREDICTIVE MAINTENANCE SYSTEM
    <br>
    Real-time IoT monitoring • Condition analysis • AI prediction
</div>
""", unsafe_allow_html=True)
