import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
import shap
import joblib
 
st.set_page_config(
    page_title="Heart Disease Detector",
    page_icon="❤️",
    layout="wide"
)
 
# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#fff0f3 0%,#fdf4ff 50%,#f0f4ff 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#be123c 0%,#e11d48 40%,#9333ea 100%) !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.2) !important;
    border: 1.5px solid rgba(255,255,255,0.4) !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: white !important; color: #be123c !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 14px !important; width: 100% !important;
    padding: 0.7rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
}
[data-testid="stMetric"] {
    background: white !important; border-radius: 20px !important;
    padding: 1.2rem 1.5rem !important;
    border: 1px solid rgba(225,29,72,0.15) !important;
    box-shadow: 0 4px 24px rgba(190,18,60,0.08) !important;
}
[data-testid="stMetricLabel"] {
    color: #be123c !important; font-weight: 600 !important;
    font-size: 0.78rem !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: #1a1a2e !important; font-weight: 800 !important;
}
[data-baseweb="tab-list"] {
    background: white !important; border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid rgba(225,29,72,0.15) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg,#be123c,#e11d48) !important;
    color: white !important; border-radius: 10px !important;
}
[data-testid="stImage"] img {
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(190,18,60,0.1) !important;
}
hr { border-color: rgba(225,29,72,0.15) !important; }
[data-testid="stAlert"][kind="warning"] { display: none !important; }
[data-testid="stExpander"] {
    background: white !important; border-radius: 16px !important;
    border: 1px solid rgba(225,29,72,0.12) !important;
}
</style>
""", unsafe_allow_html=True)
 
# ── Load artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("models/xgb_model.pkl")
    explainer    = joblib.load("models/shap_explainer.pkl")
    scaler       = joblib.load("models/scaler.pkl")
    feat_names   = joblib.load("models/feature_names.pkl")
    full_df      = joblib.load("models/full_dataset.pkl")
    boundary     = joblib.load("models/boundary_clf.pkl")
    cv_scores    = joblib.load("models/cv_scores.pkl")
    return model, explainer, scaler, feat_names, full_df, boundary, cv_scores
 
model, explainer, scaler, feat_names, full_df, boundary, cv_scores = load_artifacts()
f1_name, f2_name, clf2 = boundary
 
if "history" not in st.session_state:
    st.session_state.history = []
 
# ── Helpers ───────────────────────────────────────────────────────
def risk_badge(prob):
    if prob < 0.35:
        return "Low Risk", "#16a34a", "#dcfce7"
    elif prob < 0.65:
        return "Moderate Risk", "#d97706", "#fef3c7"
    else:
        return "High Risk", "#dc2626", "#fee2e2"
 
def chest_pain_label(cp):
    return {1:"Typical angina",2:"Atypical angina",
            3:"Non-anginal pain",4:"Asymptomatic"}[cp]
 
def draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.3), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#fff5f7")
    ax.set_facecolor("#fff5f7")
    theta_bg  = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            linewidth=20, color="#fce7f3", solid_capstyle="round", zorder=1)
    theta_val = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    c = "#16a34a" if prob < 0.35 else "#d97706" if prob < 0.65 else "#dc2626"
    ax.plot(np.cos(theta_val), np.sin(theta_val),
            linewidth=20, color=c, solid_capstyle="round", zorder=2)
    ax.text(0, 0.18, f"{prob:.0%}", ha="center", va="center",
            fontsize=30, fontweight="bold", color="#1a1a2e")
    ax.text(0, -0.22, "disease probability", ha="center",
            fontsize=9, color="#9ca3af")
    ax.text(-1.15, -0.05, "0%",   fontsize=8, color="#9ca3af", ha="center")
    ax.text( 1.15, -0.05, "100%", fontsize=8, color="#9ca3af", ha="center")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.5, 1.25); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig
 
def get_percentiles(input_dict):
    out = {}
    for col, val in input_dict.items():
        if col in full_df.columns:
            pct = (full_df[col] < val).mean() * 100
            out[col] = round(pct, 1)
    return out
 
# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.4rem;">
  <div style="background:linear-gradient(135deg,#be123c,#e11d48,#9333ea);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              font-size:2.4rem;font-weight:800;line-height:1.1;">
    ❤️ Heart Disease Detector
  </div>
  <p style="color:#6b7280;margin-top:0.4rem;font-size:0.95rem;">
    Clinical features → XGBoost + SHAP explanation + What-If Simulator + Percentile Benchmarking
  </p>
</div>""", unsafe_allow_html=True)
st.divider()
 
# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:0.5rem 0 1.2rem;">
  <div style="font-size:1.3rem;font-weight:800;">Patient Details</div>
  <div style="font-size:0.82rem;opacity:0.8;margin-top:4px;">Enter clinical measurements</div>
</div>""", unsafe_allow_html=True)
 
age      = st.sidebar.slider("Age (years)", 20, 80, 55)
sex      = st.sidebar.selectbox("Sex", [1,0],
           format_func=lambda x: "Male" if x==1 else "Female")
cp       = st.sidebar.selectbox("Chest Pain Type", [1,2,3,4],
           format_func=chest_pain_label)
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
chol     = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0,1],
           format_func=lambda x: "No" if x==0 else "Yes")
restecg  = st.sidebar.selectbox("Resting ECG", [0,1,2],
           format_func=lambda x: {0:"Normal",1:"ST-T abnormality",
                                   2:"LV hypertrophy"}[x])
thalach  = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
exang    = st.sidebar.selectbox("Exercise Induced Angina", [0,1],
           format_func=lambda x: "No" if x==0 else "Yes")
oldpeak  = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope    = st.sidebar.selectbox("Slope of Peak ST", [1,2,3],
           format_func=lambda x: {1:"Upsloping",2:"Flat",3:"Downsloping"}[x])
ca       = st.sidebar.selectbox("Major Vessels (0–3)", [0,1,2,3])
thal     = st.sidebar.selectbox("Thalassemia", [3,6,7],
           format_func=lambda x: {3:"Normal",6:"Fixed defect",
                                   7:"Reversible defect"}[x])
 
st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.sidebar.button("❤️ Analyse Risk", type="primary")
 
input_dict = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                  fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                  oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
input_df = pd.DataFrame([input_dict])[feat_names]
 
live_prob = float(model.predict_proba(input_df)[0][1])
risk_label, risk_color, risk_bg = risk_badge(live_prob)
 
# ══════════════════════════════════════════════════════════════════
# LIVE RISK METER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#be123c;margin-bottom:0.8rem;">⚡ Live Risk Meter</div>', unsafe_allow_html=True)
 
col_g, col_b, col_i = st.columns([1.2, 1, 1])
with col_g:
    st.markdown('<div style="background:white;border-radius:20px;padding:0.6rem;border:1px solid rgba(225,29,72,0.12);">', unsafe_allow_html=True)
    st.pyplot(draw_gauge(live_prob), use_container_width=True)
    plt.close()
    st.markdown("</div>", unsafe_allow_html=True)
 
with col_b:
    st.markdown(f"""
    <div style="background:white;border-radius:20px;padding:1.4rem;
                border:1px solid rgba(225,29,72,0.12);min-height:160px;">
      <div style="font-size:0.7rem;font-weight:700;color:#9ca3af;
                  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">Risk Level</div>
      <div style="display:inline-block;background:{risk_bg};color:{risk_color};
                  font-weight:800;font-size:1.05rem;padding:0.4rem 1rem;
                  border-radius:99px;margin-bottom:0.6rem;">{risk_label}</div>
      <div style="font-size:0.82rem;color:#9ca3af;">Age {age} · {"Male" if sex==1 else "Female"}</div>
      <div style="font-size:0.82rem;color:#9ca3af;">BP {trestbps} mmHg · Chol {chol}</div>
    </div>""", unsafe_allow_html=True)
 
with col_i:
    st.markdown(f"""
    <div style="background:white;border-radius:20px;padding:1.4rem;
                border:1px solid rgba(225,29,72,0.12);min-height:160px;">
      <div style="font-size:0.7rem;font-weight:700;color:#9ca3af;
                  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">Key Vitals</div>
      <div style="font-size:0.85rem;color:#374151;line-height:2;">
        Max HR: <b>{thalach} bpm</b><br>
        ST depression: <b>{oldpeak}</b><br>
        Chest pain: <b>{chest_pain_label(cp)}</b><br>
        Vessels: <b>{ca}</b>
      </div>
    </div>""", unsafe_allow_html=True)
 
st.divider()
 
# ══════════════════════════════════════════════════════════════════
# FULL PREDICTION
# ══════════════════════════════════════════════════════════════════
if predict_clicked:
    prediction = int(model.predict(input_df)[0])
    prob       = float(model.predict_proba(input_df)[0][1])
    risk_label, risk_color, risk_bg = risk_badge(prob)
 
    st.session_state.history.append({
        "Risk": f"{prob:.0%}", "Verdict": risk_label,
        "Age": age, "Sex": "M" if sex==1 else "F",
        "Chol": chol, "MaxHR": thalach, "BP": trestbps,
    })
    if len(st.session_state.history) > 8:
        st.session_state.history = st.session_state.history[-8:]
 
    if prediction == 1:
        st.error(f"⚠️ **Heart disease likely** — model confidence: {prob:.1%}")
    else:
        st.success(f"✅ **No disease detected** — model confidence: {1-prob:.1%}")
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disease Probability", f"{prob:.1%}")
    c2.metric("Risk Level", risk_label)
    c3.metric("Max Heart Rate", f"{thalach} bpm")
    c4.metric("ST Depression", f"{oldpeak}")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── SHAP + Clinical bars ──────────────────────────────────────
    col_shap, col_bars = st.columns([1.3, 1])
 
    with col_shap:
        st.markdown('<div style="font-size:1rem;font-weight:700;margin-bottom:0.4rem;">🔍 Why this prediction?</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.82rem;color:#9ca3af;margin-bottom:0.7rem;">Red = raised risk · Blue = lowered risk</div>', unsafe_allow_html=True)
        shap_vals = explainer.shap_values(input_df)
        mpl.rcParams.update({"font.family":"DejaVu Sans",
            "axes.facecolor":"#fff5f7","figure.facecolor":"#fff5f7",
            "axes.edgecolor":"#fce7f3","xtick.color":"#6b7280","ytick.color":"#1a1a2e"})
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.waterfall_plot(
            shap.Explanation(values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0], feature_names=feat_names),
            show=False, max_display=10)
        fig.patch.set_facecolor("#fff5f7")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(); mpl.rcdefaults()
 
    with col_bars:
        st.markdown('<div style="font-size:1rem;font-weight:700;margin-bottom:0.8rem;">📊 Clinical Factors</div>', unsafe_allow_html=True)
        factors = [
            ("Age",            age,      20,  80,  55,  "yrs"),
            ("Blood Pressure", trestbps, 80,  200, 140, "mmHg"),
            ("Cholesterol",    chol,     100, 600, 200, "mg/dl"),
            ("Max Heart Rate", thalach,  70,  210, 100, "bpm"),
            ("ST Depression",  oldpeak,  0,   6,   2,   ""),
        ]
        for fname, val, lo, hi, danger, unit in factors:
            pct   = int((val - lo) / (hi - lo) * 100)
            color = "#dc2626" if val >= danger else "#16a34a"
            st.markdown(f"""
            <div style="margin-bottom:0.65rem;">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.82rem;font-weight:600;margin-bottom:3px;">
                <span>{fname}</span>
                <span style="color:{color};font-weight:700;">{val} {unit}</span>
              </div>
              <div style="background:#fce7f3;border-radius:99px;height:7px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:99px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
 
    # Top SHAP driver
    shap_vals = explainer.shap_values(input_df)
    top_idx   = np.argmax(np.abs(shap_vals[0]))
    top_feat  = feat_names[top_idx]
    top_val   = shap_vals[0][top_idx]
    direction = "increased" if top_val > 0 else "decreased"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#fff0f3,#fdf4ff);
                border-radius:16px;padding:1rem 1.4rem;
                border-left:4px solid #e11d48;margin-top:0.5rem;">
      <span style="font-weight:700;color:#be123c;">Key driver: </span>
      <b>{top_feat}</b> {direction} the risk score by <b>{abs(top_val):.3f}</b>.
    </div>""", unsafe_allow_html=True)
 
    st.divider()
 
    # ══════════════════════════════════════════════════════════════
    # FEATURE 1: PATIENT PERCENTILE CARD
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:0.8rem;">📍 Patient Percentile Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#9ca3af;margin-bottom:1rem;">How this patient compares to all 303 patients in the training dataset</div>', unsafe_allow_html=True)
 
    percentiles = get_percentiles(input_dict)
    bench_cols  = st.columns(5)
    bench_items = [
        ("Age",         age,      percentiles.get("age",0),      "yrs"),
        ("Cholesterol", chol,     percentiles.get("chol",0),     "mg/dl"),
        ("Max HR",      thalach,  percentiles.get("thalach",0),  "bpm"),
        ("BP",          trestbps, percentiles.get("trestbps",0), "mmHg"),
        ("ST Depress",  oldpeak,  percentiles.get("oldpeak",0),  ""),
    ]
    for col, (lbl, val, pct, unit) in zip(bench_cols, bench_items):
        p_color = "#dc2626" if pct > 75 else "#d97706" if pct > 50 else "#16a34a"
        col.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1rem;
                    border:1px solid rgba(225,29,72,0.12);text-align:center;">
          <div style="font-size:0.72rem;font-weight:700;color:#9ca3af;
                      text-transform:uppercase;margin-bottom:0.4rem;">{lbl}</div>
          <div style="font-size:1.3rem;font-weight:800;color:#1a1a2e;">{val} <span style="font-size:0.8rem;color:#9ca3af;">{unit}</span></div>
          <div style="font-size:0.82rem;color:{p_color};font-weight:700;margin-top:0.3rem;">
            {pct:.0f}th percentile
          </div>
          <div style="background:#f3f4f6;border-radius:99px;height:5px;
                      margin-top:0.4rem;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{p_color};
                        border-radius:99px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)
 
    st.divider()
 
    # ══════════════════════════════════════════════════════════════
    # FEATURE 2: WHAT-IF RISK SIMULATOR
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div style="font-size:1.2rem;font-weight:700;color:#1a1a2e;margin-bottom:0.4rem;">🔮 What-If Risk Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#9ca3af;margin-bottom:1rem;">Adjust modifiable risk factors to see how the prediction changes</div>', unsafe_allow_html=True)
 
    with st.container():
        st.markdown('<div style="background:white;border-radius:20px;padding:1.5rem;border:1px solid rgba(225,29,72,0.12);">', unsafe_allow_html=True)
 
        w1, w2, w3 = st.columns(3)
        wi_chol    = w1.slider("What if Cholesterol was…",   100, 600, chol,     key="wi_chol")
        wi_thalach = w2.slider("What if Max Heart Rate was…", 70, 210, thalach,  key="wi_hr")
        wi_oldpeak = w3.slider("What if ST Depression was…",  0.0, 6.0, oldpeak, step=0.1, key="wi_op")
 
        whatif_dict = input_dict.copy()
        whatif_dict["chol"]    = wi_chol
        whatif_dict["thalach"] = wi_thalach
        whatif_dict["oldpeak"] = wi_oldpeak
        whatif_df   = pd.DataFrame([whatif_dict])[feat_names]
        whatif_prob = float(model.predict_proba(whatif_df)[0][1])
        delta       = whatif_prob - prob
        delta_str   = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
        delta_color = "#dc2626" if delta > 0 else "#16a34a"
 
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Original Risk",    f"{prob:.1%}")
        wc2.metric("Simulated Risk",   f"{whatif_prob:.1%}")
        wc3.metric("Change",           delta_str,
                   delta=delta_str,
                   delta_color="inverse")
 
        # Mini gauge comparison
        fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(7, 2.4),
                                          subplot_kw=dict(aspect="equal"))
        fig_cmp.patch.set_facecolor("#ffffff")
        for ax_i, (p, title) in enumerate([(prob, "Original"),
                                            (whatif_prob, "Simulated")]):
            ax_i_obj = axes_cmp[ax_i]
            ax_i_obj.set_facecolor("#ffffff")
            th_bg  = np.linspace(np.pi, 0, 200)
            ax_i_obj.plot(np.cos(th_bg), np.sin(th_bg),
                          lw=14, color="#fce7f3", solid_capstyle="round")
            th_v = np.linspace(np.pi, np.pi - p * np.pi, 200)
            cc = "#16a34a" if p < 0.35 else "#d97706" if p < 0.65 else "#dc2626"
            ax_i_obj.plot(np.cos(th_v), np.sin(th_v),
                          lw=14, color=cc, solid_capstyle="round")
            ax_i_obj.text(0, 0.1, f"{p:.0%}", ha="center", va="center",
                          fontsize=20, fontweight="bold", color="#1a1a2e")
            ax_i_obj.set_title(title, fontsize=10, fontweight="bold", pad=4)
            ax_i_obj.set_xlim(-1.3,1.3); ax_i_obj.set_ylim(-0.4,1.2)
            ax_i_obj.axis("off")
        plt.tight_layout(pad=1)
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
 
    st.divider()
 
    # ══════════════════════════════════════════════════════════════
    # FEATURE 3: DECISION BOUNDARY with patient dot
    # ══════════════════════════════════════════════════════════════
    st.markdown(f'<div style="font-size:1.2rem;font-weight:700;margin-bottom:0.4rem;">🗺️ Where Does This Patient Fall? (Decision Boundary)</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.85rem;color:#9ca3af;margin-bottom:1rem;">Plotted on the two most important SHAP features: <b>{f1_name}</b> vs <b>{f2_name}</b></div>', unsafe_allow_html=True)
 
    fig_db, ax_db = plt.subplots(figsize=(8, 5))
    fig_db.patch.set_facecolor("#fff5f7")
    ax_db.set_facecolor("#fff5f7")
 
    X2_all = full_df[[f1_name, f2_name]].values
    y_all  = full_df["target"].values
 
    from sklearn.inspection import DecisionBoundaryDisplay
    DecisionBoundaryDisplay.from_estimator(
        clf2, X2_all,
        response_method="predict_proba",
        plot_method="pcolormesh",
        xlabel=f1_name, ylabel=f2_name,
        alpha=0.22, ax=ax_db, cmap="RdYlGn_r"
    )
    # All training patients
    sc = ax_db.scatter(
        X2_all[:, 0], X2_all[:, 1],
        c=y_all, cmap="RdYlGn_r",
        edgecolors="white", linewidth=0.6,
        s=45, zorder=3, alpha=0.7
    )
    # Current patient — big star
    px = float(input_df[f1_name].values[0])
    py = float(input_df[f2_name].values[0])
    ax_db.scatter(px, py,
                  marker="*", s=500, color="#1a1a2e",
                  edgecolors="white", linewidth=1.5,
                  zorder=6, label="This patient")
    ax_db.legend(fontsize=10, loc="upper right")
    plt.colorbar(sc, ax=ax_db, label="Disease probability")
    ax_db.set_title(f"Decision Boundary — {f1_name} vs {f2_name}",
                    fontsize=13, fontweight="bold", pad=10)
    ax_db.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_db, use_container_width=True)
    plt.close()
 
    st.divider()
 
    # ── Session History ───────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown('<div style="font-size:1.1rem;font-weight:700;margin-bottom:0.8rem;">📋 Session History</div>', unsafe_allow_html=True)
 
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df.index = [f"Patient {i+1}" for i in range(len(hist_df))]
 
        fig_h, ax_h = plt.subplots(figsize=(8, 2.5))
        fig_h.patch.set_facecolor("#fff5f7")
        ax_h.set_facecolor("#fff5f7")
        raw_probs = [float(r["Risk"].strip("%"))/100
                     for r in st.session_state.history]
        bar_colors_h = ["#dc2626" if p >= 0.65 else
                        "#d97706" if p >= 0.35 else "#16a34a"
                        for p in raw_probs]
        bars_h = ax_h.bar(hist_df.index, raw_probs,
                           color=bar_colors_h, width=0.5,
                           edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars_h, raw_probs):
            ax_h.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.02,
                      f"{val:.0%}", ha="center", fontsize=9,
                      fontweight="bold", color="#374151")
        ax_h.axhline(0.5, color="#9ca3af", lw=1, linestyle="--", alpha=0.7)
        ax_h.set_ylim(0, 1.1)
        ax_h.set_ylabel("Risk probability", fontsize=9)
        ax_h.spines[:].set_visible(False)
        ax_h.yaxis.grid(True, color="#fce7f3", linewidth=0.8)
        ax_h.set_axisbelow(True)
        plt.xticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)
        plt.close()
 
        st.dataframe(hist_df, use_container_width=True)
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
 
else:
    st.markdown("""
    <div style="background:white;border-radius:20px;padding:2.5rem;
                border:2px dashed rgba(225,29,72,0.25);text-align:center;">
      <div style="font-size:2.5rem;">🫀</div>
      <div style="font-weight:700;color:#be123c;font-size:1.1rem;margin-top:0.5rem;">
        Enter patient details and click Analyse Risk
      </div>
      <div style="color:#9ca3af;font-size:0.88rem;margin-top:0.3rem;">
        SHAP explanation · Percentile benchmarking · What-If simulator · Decision boundary · Session history
      </div>
    </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════
# MODEL EVALUATION CHARTS
# ══════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div style="font-size:1.3rem;font-weight:700;margin-bottom:1rem;">📊 Model Evaluation Charts</div>', unsafe_allow_html=True)
 
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📊 Correlation Heatmap",
    "📈 ROC Curve",
    "🔲 Confusion Matrix",
    "🤝 Model Comparison",
    "🔍 SHAP Importance",
    "🎯 SHAP Direction",
    "✅ Cross Validation",
])
with tab1:
    st.image("charts/correlation_heatmap.png", width=820)
    st.caption("Each cell = Pearson correlation between two features. Green = positive · Red = negative.")
with tab2:
    st.image("charts/roc_curve.png", width=700)
    st.caption("AUC = area under the curve. Higher = better. Dots = optimal threshold per model.")
with tab3:
    st.image("charts/confusion_matrix.png", width=480)
    st.caption("Diagonal = correct predictions. Off-diagonal = errors. False Negatives (missed disease) are the most dangerous.")
with tab4:
    st.image("charts/model_comparison.png", width=820)
    st.caption("XGBoost wins across all 5 metrics — accuracy, F1, AUC, precision, and recall.")
with tab5:
    st.image("charts/shap_importance.png", width=750)
    st.caption("Bar length = how much that feature moves the prediction on average across all test patients.")
with tab6:
    st.image("charts/shap_dot.png", width=750)
    st.caption("Each dot = one patient. Red = high feature value. Right of centre = raised risk.")
with tab7:
    st.image("charts/cross_validation.png", width=750)
    st.caption("5-fold CV confirms the model generalises — μ = mean AUC, σ = standard deviation across folds.")
 
st.divider()
st.markdown("""
<div style="text-align:center;color:#9ca3af;font-size:0.8rem;padding-bottom:1rem;">
  Model: XGBoost · Dataset: UCI Cleveland · Explainability: SHAP · Not a substitute for medical advice.
</div>""", unsafe_allow_html=True)
 
