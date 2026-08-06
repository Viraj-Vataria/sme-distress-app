# =============================================================================
# UK SME Financial Distress Screening Tool
# BEMM828 MSc Project | Candidate 750069561
#
# A decision-support prototype. It loads the calibrated XGBoost model trained on
# UK open government data and returns, for a single company, a probability of
# financial distress, an interpretable Business Health Score (300-850), an
# indicative risk-based charge, and a SHAP explanation of the drivers. No raw
# training data is shipped with the app; only the serialised model and small
# sector lookup tables.
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration and light styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UK SME Distress Screening",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main { background-color: #F4F6F8; }
        h1, h2, h3 { color: #1B2A4A; }
        .stMetric { background-color: #FFFFFF; border-radius: 10px;
                    padding: 12px; border: 1px solid #E1E5EA; }
        div[data-testid="stSidebar"] { background-color: #1B2A4A; }
        div[data-testid="stSidebar"] * { color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True,
)

ASSETS = Path(__file__).parent / "assets"

# Fixed Business Health Score reference range, derived from the scored test set.
# Using fixed bounds (not the min/max of a single input) is what makes the score
# meaningful for one company at a time.
LOGIT_FLOOR, LOGIT_CEILING = -5.7666, -0.3918

# Friendly names must match the section labels used when the model was trained.
SECTION_NAMES = {
    "B": "Mining", "C": "Manufacturing", "D": "Energy", "E": "Water and waste",
    "F": "Construction", "G": "Wholesale and retail", "H": "Transport and storage",
    "I": "Accommodation and food", "J": "Information and communication",
    "K": "Finance and insurance", "L": "Real estate",
    "M": "Professional and technical", "N": "Administrative and support",
    "O": "Public administration", "P": "Education", "Q": "Health and social work",
    "R": "Arts and recreation", "S": "Other services",
    "T": "Household activities", "U": "Extraterritorial",
}

REGIONS = [
    "London", "South East", "South West", "East of England", "East Midlands",
    "West Midlands", "North West", "North East", "Yorkshire and The Humber",
    "Scotland", "Wales", "Northern Ireland",
]

ACCOUNT_CATEGORIES = [
    "MICRO ENTITY", "SMALL", "TOTAL EXEMPTION FULL",
    "UNAUDITED ABRIDGED", "MEDIUM", "NO ACCOUNTS FILED",
]


# -----------------------------------------------------------------------------
# Load model and reference tables once and cache them
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    schema = json.loads((ASSETS / "model_schema.json").read_text())
    calibrated = joblib.load(ASSETS / "xgboost_structural_calibrated.joblib")
    raw = joblib.load(ASSETS / "xgboost_structural_raw.joblib")
    bics = pd.read_csv(ASSETS / "table_08_bics_sector_risk.csv")
    ons = pd.read_csv(ASSETS / "table_09_ons_sector_death_rate.csv")

    bics_by_section = dict(zip(bics["sic_section"], bics["bics_insolvency_risk"]))
    ons_by_section = dict(zip(ons["sic_section"], ons["ons_sector_death_rate"]))
    bics_fallback = float(bics["bics_insolvency_risk"].median())
    ons_fallback = float(ons["ons_sector_death_rate"].median())
    return schema, calibrated, raw, bics_by_section, ons_by_section, bics_fallback, ons_fallback


(schema, calibrated_model, raw_model,
 bics_by_section, ons_by_section, bics_fallback, ons_fallback) = load_model_assets()

MODEL_FEATURES = schema["model_features"]
LOSS_GIVEN_DEFAULT = schema["loss_given_default"]
EXPOSURE_AT_DEFAULT = schema["exposure_at_default"]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def assemble_feature_vector(age, sic_count, charges_outstanding, charges_satisfied,
                            section_letter, region, account_category):
    """Turn a set of plain inputs into the exact 48-column model row."""
    row = {feature: 0 for feature in MODEL_FEATURES}

    row["company_age_years"] = age
    row["n_sic_codes"] = sic_count
    row["charges_outstanding"] = charges_outstanding
    row["charges_satisfied"] = charges_satisfied
    row["charges_total"] = charges_outstanding + charges_satisfied
    row["has_outstanding_charge"] = 1 if charges_outstanding > 0 else 0

    sector_risk = bics_by_section.get(section_letter, bics_fallback)
    row["bics_insolvency_risk"] = sector_risk if not pd.isna(sector_risk) else bics_fallback
    row["ons_sector_death_rate"] = ons_by_section.get(section_letter, ons_fallback)

    for prefix, value in [("sic_section_", section_letter),
                          ("region_", region),
                          ("Accounts_AccountCategory_", account_category)]:
        column = f"{prefix}{value}"
        if column in row:
            row[column] = 1

    return pd.DataFrame([row])[MODEL_FEATURES]


def probability_to_health_score(probability):
    """Map a distress probability to the 300-850 Business Health Score."""
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped))
    normalised = (LOGIT_CEILING - log_odds) / (LOGIT_CEILING - LOGIT_FLOOR)
    return int(np.clip(round(300 + normalised * 550), 300, 850))


def score_to_tier(score):
    if score < 500:
        return "Very high risk", "#C0392B"
    if score < 600:
        return "High risk", "#E67E22"
    if score < 700:
        return "Moderate", "#F1C40F"
    if score < 800:
        return "Low risk", "#A8DADC"
    return "Very low risk", "#2E86AB"


# -----------------------------------------------------------------------------
# Sidebar: company inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Company details")

selected_section_name = st.sidebar.selectbox(
    "Industry sector",
    options=sorted(SECTION_NAMES.values()),
    index=sorted(SECTION_NAMES.values()).index("Construction"),
)
section_letter = {name: letter for letter, name in SECTION_NAMES.items()}[selected_section_name]

selected_region = st.sidebar.selectbox("Region", options=REGIONS)
company_age = st.sidebar.slider("Company age (years)", 0.0, 60.0, 5.0, 0.5)
sic_count = st.sidebar.slider("Number of SIC codes registered", 1, 4, 1)
account_category = st.sidebar.selectbox("Accounts filing category", options=ACCOUNT_CATEGORIES)
charges_outstanding = st.sidebar.number_input("Outstanding charges (secured debts)", 0, 50, 0)
charges_satisfied = st.sidebar.number_input("Satisfied charges (repaid secured debts)", 0, 50, 0)

assess = st.sidebar.button("Assess distress risk", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Main panel
# -----------------------------------------------------------------------------
st.title("UK SME Financial Distress Screening")
st.caption(
    "A decision-support prototype built on UK open government data "
    "(Companies House, the Insolvency Service and the Office for National Statistics). "
    "Outputs are indicative and not a lending decision."
)

if not assess:
    st.info("Enter a company's details in the sidebar, then select **Assess distress risk**.")
    st.stop()

feature_row = assemble_feature_vector(
    company_age, sic_count, charges_outstanding, charges_satisfied,
    section_letter, selected_region, account_category,
)

distress_probability = float(calibrated_model.predict_proba(feature_row)[:, 1][0])
health = probability_to_health_score(distress_probability)
tier_label, tier_colour = score_to_tier(health)
expected_loss = distress_probability * LOSS_GIVEN_DEFAULT * EXPOSURE_AT_DEFAULT

left, middle, right = st.columns(3)
left.metric("Probability of distress", f"{distress_probability * 100:.1f}%")
middle.metric("Business Health Score", f"{health} / 850")
right.metric("Indicative risk charge", f"£{expected_loss:,.0f}")

st.markdown(
    f"<h3 style='color:{tier_colour};'>Risk tier: {tier_label}</h3>",
    unsafe_allow_html=True,
)
st.progress(min(1.0, distress_probability / 0.5))

# ---- Explanation of the drivers with SHAP ----
st.subheader("Why this result")
st.write(
    "The chart below shows which factors pushed this company's risk up (red) "
    "or down (blue), relative to the average company."
)

xgb_stage = raw_model.named_steps["clf"]
explainer = shap.TreeExplainer(xgb_stage)
shap_row = explainer(feature_row)

fig, ax = plt.subplots(figsize=(9, 4.5))
shap.plots.waterfall(shap_row[0], max_display=8, show=False)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

with st.expander("View the exact inputs used"):
    st.dataframe(
        pd.DataFrame({
            "Attribute": ["Sector", "Region", "Age (years)", "SIC codes",
                          "Accounts category", "Charges outstanding", "Charges satisfied"],
            "Value": [selected_section_name, selected_region, company_age, sic_count,
                      account_category, charges_outstanding, charges_satisfied],
        }),
        hide_index=True, use_container_width=True,
    )

st.caption(
    "Model: calibrated XGBoost on structural features. ROC-AUC 0.83, "
    "Brier 0.04 on held-out test data. For academic demonstration only."
)