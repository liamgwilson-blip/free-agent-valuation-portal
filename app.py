# Save this as app.py in the same folder as your .pkl files

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Contract Predictor",
    page_icon="⚾",
    layout="centered"
)

# ── Load models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        'rf_hit':       joblib.load('rf_hit.pkl'),
        'rf_pitch':     joblib.load('rf_pitch.pkl'),
        'clf_hit':      joblib.load('clf_hit.pkl'),
        'clf_pitch':    joblib.load('clf_pitch.pkl'),
        'features_hit': joblib.load('features_hit.pkl'),
        'features_pitch': joblib.load('features_pitch.pkl'),
    }

models = load_models()

# ── Styling ────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .title { font-size: 2.2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
    .subtitle { font-size: 1rem; color: #8b8fa8; margin-bottom: 2rem; }
    .result-box {
        background: #1c2333;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        border-left: 4px solid #e63946;
    }
    .result-value { font-size: 2.4rem; font-weight: 800; color: #ffffff; }
    .result-label { font-size: 0.85rem; color: #8b8fa8; text-transform: uppercase;
                    letter-spacing: 1px; margin-bottom: 0.2rem; }
    .result-ci { font-size: 1rem; color: #a8b2d8; margin-top: 0.4rem; }
    .result-length { font-size: 1.1rem; color: #e63946; font-weight: 600;
                     margin-top: 0.8rem; }
    .warning { font-size: 0.85rem; color: #f4a261; margin-top: 1rem; }
    .divider { border-top: 1px solid #2a2f45; margin: 1.5rem 0; }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="title">⚾ MLB Contract Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter a player\'s walk-year stats to predict their free agent contract value and duration. '
    'Built on ~1,200 MLB free agent signings from 2017–2026.</div>',
    unsafe_allow_html=True
)

# ── Player type selector ───────────────────────────────────
player_type = st.radio(
    "Player type",
    ["Hitter", "Starting Pitcher", "Relief Pitcher"],
    horizontal=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Input fields ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    walk_fwar = st.number_input("Walk-year fWAR", min_value=-3.0, max_value=15.0,
                                 value=3.0, step=0.1)
    three_yr_avg = st.number_input("3-year fWAR average", min_value=-3.0, max_value=15.0,
                                    value=2.5, step=0.1)
    age = st.number_input("Age at signing", min_value=20, max_value=45, value=28)

with col2:
    if player_type == "Hitter":
        wrc_plus = st.number_input("Walk-year wRC+", min_value=0, max_value=300, value=115)
        xwoba_diff = st.number_input("xwOBA minus wOBA (luck indicator)",
                              min_value=-0.150, max_value=0.150, value=0.0, step=0.001,
                              format="%.3f",
                              help="Subtract xwOBA from wOBA using Baseball Savant values. Positive = outperformed expected stats (regression risk). "
                                   "Negative = underperformed (potential upside).")
        pa_pct = st.slider("PA% of full season (650 PA)", 0.0, 1.2, 0.85, 0.01,
                            help="Durability proxy. 1.0 = 650 PA.")
    else:
        xera = st.number_input("xERA", min_value=0.0, max_value=10.0, value=3.80, step=0.01, help="Expected ERA from Baseball Savant. Enter as shown, e.g. 3.45")
        brl_pa = st.number_input("Barrel% (per batted ball event)", min_value=0.0, max_value=0.20,
                                  value=0.05, step=0.001, format="%.3f", help="Brl% from Baseball Savant player page. Format as decimal, e.g. 7.1% = 0.071")
        ev95 = st.number_input("EV95+ (hard-hit rate)", min_value=0.0, max_value=1.0,
                                value=0.40, step=0.01, format="%.3f")
        ip_pct = st.slider("IP% of full season", 0.0, 1.2, 0.85, 0.01,
                            help="Innings pitched/'full season' benchmark. SP: full season ≈ 180 IP. RP: full season ≈ 65 IP.")

is_international = st.checkbox("International signing (no prior MLB service)",
                                help="Check for NPB/KBO players signing their first MLB contract.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────
if st.button("Predict Contract", type="primary", use_container_width=True):

    fwar_vs_avg = walk_fwar - three_yr_avg

    if player_type == "Hitter":
        years_past_peak = age - 27
        base = {
            'walk_year_fWAR':           walk_fwar,
            '3_yr_fWAR_avg':            three_yr_avg,
            'walk_year_age':            age,
            'walk_WRC+':                wrc_plus,
            'est_woba_minus_woba_diff': xwoba_diff,
            'pa_pct':                   pa_pct,
            'is_international':         int(is_international),
            'fWAR_vs_avg':              fwar_vs_avg,
            'years_past_peak':          years_past_peak,
        }
        features  = models['features_hit']
        reg_model = models['rf_hit']
        clf_model = models['clf_hit']
        ci_lo_pct, ci_hi_pct = 15, 85
        ci_label = "70%"

    else:
        is_rp = 1 if player_type == "Relief Pitcher" else 0
        years_past_peak = age - 26
        base = {
            'walk_year_fWAR':  walk_fwar,
            '3_yr_fWAR_avg':   three_yr_avg,
            'walk_year_age':   age,
            'xera':            xera,
            'brl_pa':          brl_pa,
            'ev95percent':     ev95,
            'fWAR_vs_avg':     fwar_vs_avg,
            'years_past_peak': years_past_peak,
            'is_rp':           is_rp,
        }
        features  = models['features_pitch']
        reg_model = models['rf_pitch']
        clf_model = models['clf_pitch']
        ci_lo_pct, ci_hi_pct = 5, 95
        ci_label = "90%"

    X_pred = pd.DataFrame([base])[features]

    tree_preds         = np.array([t.predict(X_pred.values)[0] for t in reg_model.estimators_])
    tree_preds_dollars = np.expm1(tree_preds)

    pred_value  = round(float(np.mean(tree_preds_dollars)), 2)
    ci_low      = round(float(np.percentile(tree_preds_dollars, ci_lo_pct)), 2)
    ci_high     = round(float(np.percentile(tree_preds_dollars, ci_hi_pct)), 2)
    pred_length = clf_model.predict(X_pred)[0]

    # ── Results display ────────────────────────────────────
    st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Total Value</div>
            <div class="result-value">${pred_value:,.1f}M</div>
            <div class="result-ci">{ci_label} Confidence Interval: ${ci_low:,.1f}M — ${ci_high:,.1f}M</div>
            <div class="result-length">Predicted Duration: {pred_length}</div>
        </div>
    """, unsafe_allow_html=True)

    # ── Distribution chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#1c2333')
    ax.set_facecolor('#1c2333')

    ax.hist(tree_preds_dollars, bins=40, color='#2a3555', edgecolor='none', alpha=0.9)
    ax.axvline(pred_value, color='#e63946', linewidth=2, label=f'Prediction: ${pred_value:.0f}M')
    ax.axvline(ci_low,  color='#a8b2d8', linewidth=1, linestyle='--', alpha=0.7)
    ax.axvline(ci_high, color='#a8b2d8', linewidth=1, linestyle='--', alpha=0.7,
               label=f'{ci_label} CI: ${ci_low:.0f}M — ${ci_high:.0f}M')

    ax.set_xlabel('Predicted Contract Value ($M)', color='#8b8fa8', fontsize=9)
    ax.set_ylabel('Tree count', color='#8b8fa8', fontsize=9)
    ax.tick_params(colors='#8b8fa8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(fontsize=8, framealpha=0, labelcolor='#a8b2d8')

    st.pyplot(fig)
    plt.close()

    # ── Star warning ───────────────────────────────────────
    if walk_fwar >= 7.0:
        st.markdown(
            '<div class="warning">⚠️ Elite player detected (fWAR ≥ 7.0). '
            'The model tends to underpredict generational talent — market premiums '
            'for star power, revenue impact, and opt-out value are not captured '
            'in WAR-based features. Treat this as a floor, not a ceiling.</div>',
            unsafe_allow_html=True
        )

# ── Footer ─────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Model trained on ~1,200 MLB free agent signings, 2017–2026. '
    'Contract values inflation-adjusted to 2025 dollars. '
    'Built with Random Forest — separate models for hitters and pitchers.</div>',
    unsafe_allow_html=True
)
