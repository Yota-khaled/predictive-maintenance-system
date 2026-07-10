"""
Predictive Maintenance System
==============================
A Streamlit application that loads a pre-trained Random Forest model and predicts
whether a machine is expected to experience a failure, based on sensor
readings (air/process temperature, rotational speed, torque, tool wear)
and machine type.

Run with:
    streamlit run app.py
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# Absolute path to the trained model on the developer's machine.
# If this exact file exists, it is used directly. This makes local
# development convenient while still allowing the app to be deployed
# elsewhere (see MODEL_FILENAME_CANDIDATES fallback below).
LOCAL_MODEL_PATH = r"C:\Users\Ayakhaled\Downloads\DPI_Course\Manufacturing_Project\Best_Model\best_machine_failure_model.pkl"

# Candidate filenames for the saved model (searched next to app.py / cwd)
# if LOCAL_MODEL_PATH is not found — useful when sharing the app or
# deploying it on another machine/server.
MODEL_FILENAME_CANDIDATES = [
    "best_machine_failure_model.pkl",
    "best_model.pkl",
    "random_forest_model.pkl",
    "model.pkl",
]

# Mapping used during training: L -> 0, M -> 1, H -> 2
TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}
TYPE_DESCRIPTIONS = {"L": "Low", "M": "Medium", "H": "High"}

# Exact feature order expected by the trained model
FEATURE_ORDER = [
    "Type",
    "Air_Temperature",
    "Process_Temperature",
    "Rotational_Speed",
    "Torque",
    "Tool_Wear",
]

# Numerical input ranges/defaults derived from the training dataset.
# "decimals" controls how many digits are shown in the dashboard readouts
# (this is what keeps the summary cards from printing long float tails).
NUMERIC_FIELDS = {
    "Air_Temperature": {
        "min": 295.3, "max": 304.5, "default": 300.0, "step": 0.1,
        "label": "Air Temperature", "unit": "K", "decimals": 2, "icon": "🌡️",
    },
    "Process_Temperature": {
        "min": 305.7, "max": 313.8, "default": 310.0, "step": 0.1,
        "label": "Process Temperature", "unit": "K", "decimals": 2, "icon": "♨️",
    },
    "Rotational_Speed": {
        "min": 1168, "max": 2886, "default": 1539, "step": 1,
        "label": "Rotational Speed", "unit": "rpm", "decimals": 0, "icon": "🌀",
    },
    "Torque": {
        "min": 3.8, "max": 76.6, "default": 40.0, "step": 0.1,
        "label": "Torque", "unit": "Nm", "decimals": 2, "icon": "🔩",
    },
    "Tool_Wear": {
        "min": 0, "max": 253, "default": 108, "step": 1,
        "label": "Tool Wear", "unit": "min", "decimals": 0, "icon": "⏱️",
    },
}


# --------------------------------------------------------------------------
# Visual Theme — Industrial Control-Room Styling
# --------------------------------------------------------------------------
def inject_theme():
    """
    Inject the app's custom design system (fonts, colors, cards, gauge)
    as a single CSS block. Keeps visuals consistent regardless of the
    viewer's system Streamlit theme (light/dark).
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap');

        :root {
            --bg: #0B0E14;
            --panel: #12161F;
            --panel-border: #232A38;
            --accent: #F2A93B;
            --accent-soft: rgba(242, 169, 59, 0.14);
            --success: #34D399;
            --danger: #FB6A6A;
            --text: #E8ECF3;
            --text-muted: #8B93A7;
            --track: #232A38;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(242, 169, 59, 0.06), transparent 40%),
                repeating-linear-gradient(180deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
                var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }

        section[data-testid="stSidebar"] {
            background: #0E121A;
            border-right: 1px solid var(--panel-border);
        }

        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

        /* Eyebrow / kicker label */
        .kicker {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.35rem;
        }

        .app-subtitle { color: var(--text-muted); font-size: 0.98rem; max-width: 780px; }

        /* Status strip */
        .status-strip {
            display: flex; justify-content: space-between; align-items: center;
            background: var(--panel); border: 1px solid var(--panel-border);
            border-radius: 10px; padding: 0.6rem 1rem; margin: 1rem 0 1.4rem 0;
            font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--text-muted);
        }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--success); margin-right: 6px; box-shadow: 0 0 8px var(--success); }

        /* Readout cards (sensor summary) */
        .readout-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.7rem; margin-bottom: 0.4rem; }
        .readout-card {
            background: var(--panel); border: 1px solid var(--panel-border); border-radius: 12px;
            padding: 0.85rem 0.9rem; position: relative; overflow: hidden;
        }
        .readout-card::before {
            content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent);
        }
        .readout-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
        .readout-value { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 600; color: var(--text); }
        .readout-unit { font-size: 0.78rem; color: var(--text-muted); margin-left: 3px; }

        /* Section divider label */
        .section-label {
            font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem;
            display: flex; align-items: center; gap: 0.5rem; margin: 1.6rem 0 0.8rem 0;
        }

        /* Verdict banner */
        .verdict {
            border-radius: 14px; padding: 1.2rem 1.4rem; display: flex; gap: 1rem; align-items: flex-start;
            border: 1px solid; margin-bottom: 1rem;
        }
        .verdict-ok { background: rgba(52, 211, 153, 0.08); border-color: rgba(52, 211, 153, 0.35); }
        .verdict-fail { background: rgba(251, 106, 106, 0.08); border-color: rgba(251, 106, 106, 0.35); }
        .verdict-icon { font-size: 1.8rem; line-height: 1; }
        .verdict-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem; margin-bottom: 0.2rem; }
        .verdict-body { color: var(--text-muted); font-size: 0.92rem; }

        /* Gauge */
        .gauge-row { display: flex; align-items: center; gap: 2rem; flex-wrap: wrap; }
        .gauge {
            width: 168px; height: 168px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            position: relative;
        }
        .gauge-inner {
            width: 128px; height: 128px; border-radius: 50%; background: var(--panel);
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            border: 1px solid var(--panel-border);
        }
        .gauge-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; }
        .gauge-caption { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

        .prob-bars { flex: 1; min-width: 260px; }
        .prob-row { margin-bottom: 0.9rem; }
        .prob-row-head { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; }
        .prob-track { background: var(--track); border-radius: 6px; height: 10px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 6px; }

        .footer-note { color: var(--text-muted); font-size: 0.8rem; text-align: center; margin-top: 2rem; }

        @media (max-width: 900px) {
            .readout-grid { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_numeric(field: str, value) -> str:
    """Format a numeric reading according to its configured decimal precision."""
    config = NUMERIC_FIELDS[field]
    if config["decimals"] == 0:
        return f"{int(round(value)):,}"
    return f"{value:.{config['decimals']}f}"


def readout_card_html(icon: str, label: str, value_str: str, unit: str = "") -> str:
    """Build the HTML for a single console-style readout card."""
    return (
        f'<div class="readout-card">'
        f'<div class="readout-label">{icon} {label}</div>'
        f'<div class="readout-value">{value_str}<span class="readout-unit">{unit}</span></div>'
        f"</div>"
    )


def gauge_html(confidence: float, is_failure: bool) -> str:
    """Build a conic-gradient dial gauge showing the model's confidence."""
    color = "var(--danger)" if is_failure else "var(--success)"
    deg = max(0.0, min(100.0, confidence)) * 3.6
    return f"""<div class="gauge" style="background: conic-gradient({color} {deg}deg, var(--track) {deg}deg);">
        <div class="gauge-inner">
            <div class="gauge-value" style="color:{color}">{confidence:.2f}%</div>
            <div class="gauge-caption">Confidence</div>
        </div>
    </div>""".strip()


def prob_bars_html(prob_no_failure: float, prob_failure: float) -> str:
    """Build the two horizontal probability bars (no-failure / failure)."""
    return f"""<div class="prob-bars">
        <div class="prob-row">
            <div class="prob-row-head">
                <span>✅ No Failure</span><span style="font-family:'JetBrains Mono',monospace;">{prob_no_failure:.2f}%</span>
            </div>
            <div class="prob-track"><div class="prob-fill" style="width:{prob_no_failure:.2f}%; background: var(--success);"></div></div>
        </div>
        <div class="prob-row">
            <div class="prob-row-head">
                <span>🚨 Failure</span><span style="font-family:'JetBrains Mono',monospace;">{prob_failure:.2f}%</span>
            </div>
            <div class="prob-track"><div class="prob-fill" style="width:{prob_failure:.2f}%; background: var(--danger);"></div></div>
        </div>
    </div>""".strip()


# --------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained Random Forest model...")
def load_model():
    """
    Load the saved Random Forest model from disk using joblib.
    Searches a small list of common filenames so the app doesn't break
    if the model was saved under a slightly different name.

    Returns
    -------
    model : object or None
        The loaded model, or None if no model file could be found.
    model_path : str or None
        Path of the model file that was loaded.
    """
    # 1. Try the known absolute path first (developer's local machine).
    if os.path.exists(LOCAL_MODEL_PATH):
        model = joblib.load(LOCAL_MODEL_PATH)
        return model, LOCAL_MODEL_PATH

    # 2. Fall back to searching common filenames near the app.
    search_locations = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]

    for location in search_locations:
        for filename in MODEL_FILENAME_CANDIDATES:
            candidate_path = os.path.join(location, filename)
            if os.path.exists(candidate_path):
                model = joblib.load(candidate_path)
                return model, candidate_path

    # Fallback: look for any .pkl file that looks like a model
    for location in search_locations:
        pkl_files = glob.glob(os.path.join(location, "*.pkl"))
        for pkl_file in pkl_files:
            if "encoder" not in os.path.basename(pkl_file).lower():
                model = joblib.load(pkl_file)
                return model, pkl_file

    return None, None


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def build_input_dataframe(machine_type: str, numeric_values: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame with the exact feature order and naming
    used during model training.

    Parameters
    ----------
    machine_type : str
        One of "L", "M", "H".
    numeric_values : dict
        Dictionary with keys matching NUMERIC_FIELDS.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame ready to be passed to the model.
    """
    encoded_type = TYPE_ENCODING[machine_type]

    row = {
        "Type": encoded_type,
        "Air_Temperature": numeric_values["Air_Temperature"],
        "Process_Temperature": numeric_values["Process_Temperature"],
        "Rotational_Speed": numeric_values["Rotational_Speed"],
        "Torque": numeric_values["Torque"],
        "Tool_Wear": numeric_values["Tool_Wear"],
    }

    return pd.DataFrame([row], columns=FEATURE_ORDER)


def validate_inputs(numeric_values: dict) -> list:
    """
    Validate that all numeric inputs fall within the allowed training ranges.

    Returns
    -------
    list of str
        A list of human-readable error messages. Empty if everything is valid.
    """
    errors = []
    for field, config in NUMERIC_FIELDS.items():
        value = numeric_values.get(field)
        if value is None:
            errors.append(f"{config['label']} is missing.")
            continue
        if value < config["min"] or value > config["max"]:
            errors.append(
                f"{config['label']} must be between {config['min']} and {config['max']}."
            )
    return errors


def render_prediction_result(prediction: int, probabilities: np.ndarray):
    """
    Render the prediction outcome, gauge, and probability breakdown.

    Parameters
    ----------
    prediction : int
        0 for no failure, 1 for failure.
    probabilities : np.ndarray
        Array of shape (2,) with [P(no failure), P(failure)].
    """
    prob_no_failure = float(probabilities[0]) * 100
    prob_failure = float(probabilities[1]) * 100
    confidence = max(prob_no_failure, prob_failure)
    is_failure = prediction == 1

    st.markdown('<div class="section-label">📊 Prediction Result</div>', unsafe_allow_html=True)

    if not is_failure:
        st.markdown(
            """
            <div class="verdict verdict-ok">
                <div class="verdict-icon">✅</div>
                <div>
                    <div class="verdict-title">No Machine Failure</div>
                    <div class="verdict-body">The machine is expected to operate normally based on the current sensor readings.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="verdict verdict-fail">
                <div class="verdict-icon">🚨</div>
                <div>
                    <div class="verdict-title">Machine Failure Predicted</div>
                    <div class="verdict-body">The current sensor readings indicate that the machine is likely to experience a failure. Preventive maintenance is recommended.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">📈 Prediction Confidence</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="gauge-row">{gauge_html(confidence, is_failure)}{prob_bars_html(prob_no_failure, prob_failure)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar — User Inputs
# --------------------------------------------------------------------------
def render_sidebar():
    """
    Render all user input widgets inside the Streamlit sidebar.

    Returns
    -------
    tuple(str, dict, bool)
        Selected product type, a dict of numeric input values, and whether
        the Predict button was clicked.
    """
    st.sidebar.markdown('<div class="kicker">Configuration</div>', unsafe_allow_html=True)
    st.sidebar.markdown("## ⚙️ Machine Setup")
    st.sidebar.caption("Set the sensor readings, then run the prediction.")

    user_email = st.sidebar.text_input(
        "📧 Email Address",
        placeholder="example@gmail.com"
    )
    st.sidebar.divider()

    st.sidebar.divider()

   

    product_type = st.sidebar.selectbox(
        "🏭 Product Type",
        options=list(TYPE_ENCODING.keys()),
        index=0,
        format_func=lambda t: f"{t} — {TYPE_DESCRIPTIONS[t]} quality",
        help="L = Low, M = Medium, H = High quality variant",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🌡️ Sensor Readings")

    numeric_values = {}
    for field, config in NUMERIC_FIELDS.items():
        is_float = isinstance(config["step"], float)
        numeric_values[field] = st.sidebar.number_input(
            f"{config['icon']} {config['label']} ({config['unit']})",
            min_value=float(config["min"]) if is_float else int(config["min"]),
            max_value=float(config["max"]) if is_float else int(config["max"]),
            value=float(config["default"]) if is_float else int(config["default"]),
            step=config["step"],
            format="%.1f" if is_float else "%d",
        )

    st.sidebar.divider()

    st.sidebar.divider()
    predict_clicked = st.sidebar.button("🔍 Run Prediction", use_container_width=True, type="primary")

    return product_type, numeric_values, user_email, predict_clicked



webhook_url = "http://localhost:5678/webhook/machine-failure"


# --------------------------------------------------------------------------
# Main Application
# --------------------------------------------------------------------------
def main():
    inject_theme()

    # ---------------------- Header ----------------------
    st.markdown('<div class="kicker">🔧 Manufacturing · Machine Health</div>', unsafe_allow_html=True)
    st.title("Predictive Maintenance System")
    st.markdown(
        '<p class="app-subtitle">This dashboard predicts the likelihood of '
        "<b>machine failure</b> from live sensor readings using a trained "
        "<b>Random Forest</b> classifier. Configure the machine's parameters in the "
        "sidebar, then run the prediction.</p>",
        unsafe_allow_html=True,
    )

    # ---------------------- Load Model ----------------------
    model, model_path = load_model()

    if model is None:
        st.error(
            "❌ No saved model file (.pkl) could be found. Please make sure the "
            "trained Random Forest model file is placed alongside `app.py`."
        )
        st.stop()

    st.markdown(
        f"""
        <div class="status-strip">
            <span>📦 Model loaded from: <code>{os.path.basename(model_path)}</code></span>
            <span><span class="status-dot"></span>System Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------- Sidebar Inputs ----------------------
    product_type, numeric_values, user_email, predict_clicked = render_sidebar()
    # ---------------------- Overview Panel ----------------------
    st.markdown('<div class="section-label">🧾 Current Sensor Readout</div>', unsafe_allow_html=True)

    type_card = readout_card_html("🏭", "Product Type", f"{product_type}", TYPE_DESCRIPTIONS[product_type])
    field_cards = "".join(
        readout_card_html(
            NUMERIC_FIELDS[f]["icon"],
            NUMERIC_FIELDS[f]["label"],
            format_numeric(f, numeric_values[f]),
            NUMERIC_FIELDS[f]["unit"],
        )
        for f in FEATURE_ORDER[1:]
    )
    st.markdown(f'<div class="readout-grid">{type_card}{field_cards}</div>', unsafe_allow_html=True)

    st.divider()

    # ---------------------- Prediction ----------------------
    if predict_clicked:
        errors = validate_inputs(numeric_values)

        if errors:
            st.warning("⚠️ Please fix the following issues before predicting:")
            for err in errors:
                st.write(f"- {err}")
            st.stop()

        if not user_email or not str(user_email).strip():
            st.warning("⚠️ Please enter your email address before running a prediction.")
            st.stop()

        try:
            input_df = build_input_dataframe(product_type, numeric_values)

            with st.expander("🔍 View exact model input", expanded=False):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]

            prob_no_failure = float(probabilities[0]) * 100
            prob_failure = float(probabilities[1]) * 100
            confidence = max(prob_no_failure, prob_failure)
            prediction_label = "Failure" if prediction == 1 else "No Failure"

            payload = {
                "email": user_email,
                "product_type": product_type,
                "air_temperature": numeric_values["Air_Temperature"],
                "process_temperature": numeric_values["Process_Temperature"],
                "rotational_speed": numeric_values["Rotational_Speed"],
                "torque": numeric_values["Torque"],
                "tool_wear": numeric_values["Tool_Wear"],
                "prediction": prediction_label,
                "confidence": round(confidence, 2),
            }

            try:
                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as exc:
                st.warning(f"⚠️ Could not send prediction to webhook: {exc}")

            render_prediction_result(prediction, probabilities)

        except Exception as exc:
            st.error(
                "❌ An error occurred while generating the prediction. "
                "Please verify your inputs and try again.\n\n"
                f"Technical details: {exc}"
            )
    else:
        st.info("👈 Set the machine parameters in the sidebar and click **Run Prediction** to see results.")

    # ---------------------- Footer ----------------------
    st.markdown(
        '<div class="footer-note"> Predictive Maintenance System · Powered by Random Forest & Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()