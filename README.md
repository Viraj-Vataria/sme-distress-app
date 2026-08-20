# UK SME Financial Distress Screening

An explainable machine learning tool that estimates the probability of financial
distress for a UK small or medium-sized enterprise (SME) from open government
data, and returns an interpretable Business Health Score with a transparent,
feature-level explanation of every result.

**Live app:** https://sme-distress-app.streamlit.app/ 

---

## Overview

Traditional SME risk assessment relies either on financial-statement models that
small firms rarely file in full, or on opaque proprietary scores. This tool takes
a different approach: it predicts distress from structural, publicly observable
company attributes, calibrates the output into a probability that can be trusted
in absolute terms, and explains each prediction using SHAP so that the result is
auditable rather than a black box.

Given a company's characteristics, the app returns:

- a calibrated probability of financial distress
- a Business Health Score on a 300–850 scale, mapped to a risk tier
- an indicative expected-loss charge
- a SHAP explanation showing which factors raised or lowered the risk

The application ships with the trained model only. No underlying company data is
bundled or exposed.

## Model

- **Algorithm:** XGBoost, trained on structurally observable features and
  calibrated with Platt scaling.
- **Data:** UK open government sources — Companies House (firm attributes),
  the Insolvency Service (distress outcomes), and the Office for National
  Statistics (sector-level context), all under the Open Government Licence v3.
- **Test performance:** ROC-AUC 0.83, calibrated Brier score 0.04.

A deliberate design choice was to exclude filing-behaviour features that are
partly a consequence of insolvency, which inflate apparent accuracy through
target leakage. The deployed model uses leakage-free structural features only,
so its outputs reflect what can honestly be inferred before failure occurs.

## Repository structure
├── app.py # Streamlit application

├── requirements.txt # Dependencies

├── assets/ # Calibrated model and sector lookup tables

└── README.md



## Tech stack

Python · Streamlit · XGBoost · scikit-learn · SHAP · pandas

## Disclaimer

This tool is an academic prototype built for research and demonstration. Its
outputs are indicative only and do not constitute a lending, underwriting or
credit decision. It is intended as decision-support, not automated
decision-making.

