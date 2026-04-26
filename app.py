import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="ChurnRadar · Telecom AI", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%); color: #e2e8f0; }
.hero-card { background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 100%); border: 1px solid rgba(99,102,241,0.4); border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; }
.hero-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; line-height: 1.1; }
.hero-sub { color: #94a3b8; font-size: 1.05rem; margin-top: 0.5rem; }
.metric-card { background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,27,75,0.7)); border: 1px solid rgba(99,102,241,0.25); border-radius: 16px; padding: 1.4rem 1.6rem; text-align: center; transition: transform 0.2s, border-color 0.2s; }
.metric-card:hover { transform: translateY(-3px); border-color: rgba(129,140,248,0.6); }
.metric-label { font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; color: #64748b; margin-bottom: 0.3rem; }
.metric-value { font-size: 2rem; font-weight: 800; color: #818cf8; }
.result-churn { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1)); border: 2px solid rgba(239,68,68,0.6); border-radius: 20px; padding: 2rem; text-align: center; margin: 1rem 0; }
.result-safe { background: linear-gradient(135deg, rgba(52,211,153,0.2), rgba(16,185,129,0.1)); border: 2px solid rgba(52,211,153,0.6); border-radius: 20px; padding: 2rem; text-align: center; margin: 1rem 0; }
.result-title { font-size: 2rem; font-weight: 800; margin: 0; }
.result-sub { font-size: 1rem; color: #94a3b8; margin-top: 0.4rem; }
.section-header { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #818cf8; margin: 1.5rem 0 0.8rem; }
.fancy-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent); margin: 1.5rem 0; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.85rem 2rem !important; font-size: 1rem !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(15,23,42,0.7); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(99,102,241,0.4) !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best_rf_model_py313.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

FEATURE_NAMES = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
]

def build_feature_vector(inputs):
    v = {f: 0.0 for f in FEATURE_NAMES}
    v['SeniorCitizen']  = float(inputs['senior'])
    v['tenure']         = float(inputs['tenure'])
    v['MonthlyCharges'] = float(inputs['monthly'])
    v['TotalCharges']   = float(inputs['total'])
    v[f"gender_{inputs['gender']}"]            = 1.0
    v[f"Partner_{inputs['partner']}"]          = 1.0
    v[f"Dependents_{inputs['dependents']}"]    = 1.0
    v[f"PhoneService_{inputs['phone']}"]       = 1.0
    v[f"MultipleLines_{inputs['multilines']}"] = 1.0
    v[f"InternetService_{inputs['internet']}"] = 1.0
    v[f"OnlineSecurity_{inputs['security']}"]  = 1.0
    v[f"OnlineBackup_{inputs['backup']}"]      = 1.0
    v[f"DeviceProtection_{inputs['device']}"]  = 1.0
    v[f"TechSupport_{inputs['techsupport']}"]  = 1.0
    v[f"StreamingTV_{inputs['tv']}"]           = 1.0
    v[f"StreamingMovies_{inputs['movies']}"]   = 1.0
    v[f"Contract_{inputs['contract']}"]        = 1.0
    v[f"PaperlessBilling_{inputs['paperless']}"] = 1.0
    v[f"PaymentMethod_{inputs['payment']}"]    = 1.0
    return np.array([v[f] for f in FEATURE_NAMES]).reshape(1, -1)

st.markdown('<div class="hero-card"><div class="hero-title">📡 ChurnRadar</div><div class="hero-sub">AI-powered telecom customer churn prediction · Random Forest Classifier</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">🌲 Trees</div><div class="metric-value">{model.n_estimators}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">🔬 Features</div><div class="metric-value">{model.n_features_in_}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">📏 Max Depth</div><div class="metric-value">{model.max_depth}</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">🎯 Criterion</div><div class="metric-value" style="font-size:1.3rem;padding-top:0.3rem;">{model.criterion.upper()}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯  Predict Churn", "📊  Feature Importance", "ℹ️  Model Info"])

with tab1:
    st.markdown("")
    col_form, col_result = st.columns([1, 1], gap="large")
    with col_form:
        st.markdown('<div class="section-header">👤 Customer Profile</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            gender     = st.selectbox("Gender", ["Female", "Male"])
            senior     = st.checkbox("Senior Citizen (65+)")
        with r2:
            partner    = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        st.markdown('<div class="section-header">📅 Account Details</div>', unsafe_allow_html=True)
        tenure  = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total   = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=round(tenure * monthly, 2), step=1.0)
        contract  = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment   = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        st.markdown('<div class="section-header">📞 Services</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            phone      = st.selectbox("Phone Service", ["Yes", "No"])
            multilines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet   = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with p2:
            security = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
            backup   = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
            device   = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        p3, p4 = st.columns(2)
        with p3:
            techsupport = st.selectbox("Tech Support",    ["No", "Yes", "No internet service"])
            tv          = st.selectbox("Streaming TV",    ["No", "Yes", "No internet service"])
        with p4:
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        st.markdown("")
        predict_btn = st.button("🔍 Run Churn Analysis", use_container_width=True)

    with col_result:
        if predict_btn:
            inputs = dict(senior=int(senior), tenure=tenure, monthly=monthly, total=total,
                          gender=gender, partner=partner, dependents=dependents,
                          phone=phone, multilines=multilines, internet=internet,
                          security=security, backup=backup, device=device,
                          techsupport=techsupport, tv=tv, movies=movies,
                          contract=contract, paperless=paperless, payment=payment)
            X          = build_feature_vector(inputs)
            pred       = model.predict(X)[0]
            proba      = model.predict_proba(X)[0]
            churn_prob = proba[list(model.classes_).index(1)]
            stay_prob  = 1 - churn_prob
            is_churn   = pred == 1
            gauge_color = "#ef4444" if is_churn else "#34d399"

            if is_churn:
                st.markdown('<div class="result-churn"><div class="result-title">⚠️ HIGH CHURN RISK</div><div class="result-sub">This customer is likely to leave</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-safe"><div class="result-title">✅ LOW CHURN RISK</div><div class="result-sub">This customer is likely to stay</div></div>', unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=round(churn_prob * 100, 1),
                title={'text': "Churn Probability", 'font': {'color': '#94a3b8', 'size': 14}},
                number={'suffix': "%", 'font': {'color': gauge_color, 'size': 36}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickfont': {'color': '#475569'}},
                       'bar': {'color': gauge_color, 'thickness': 0.25}, 'bgcolor': 'rgba(0,0,0,0)',
                       'bordercolor': 'rgba(99,102,241,0.3)',
                       'steps': [{'range': [0, 30], 'color': 'rgba(52,211,153,0.15)'},
                                  {'range': [30, 60], 'color': 'rgba(251,191,36,0.1)'},
                                  {'range': [60, 100], 'color': 'rgba(239,68,68,0.15)'}],
                       'threshold': {'line': {'color': '#f59e0b', 'width': 3}, 'thickness': 0.75, 'value': 50}}))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    height=240, margin=dict(t=30, b=0, l=20, r=20), font={'color': '#e2e8f0'})
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

            fig_donut = go.Figure(go.Pie(labels=["Will Churn", "Will Stay"], values=[churn_prob, stay_prob],
                                          hole=0.65, marker_colors=["#ef4444", "#34d399"],
                                          hovertemplate="%{label}: %{percent}<extra></extra>"))
            fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
                                    legend=dict(font=dict(color='#94a3b8')), height=220,
                                    margin=dict(t=10, b=10, l=10, r=10),
                                    annotations=[dict(text="CHURN" if is_churn else "SAFE", x=0.5, y=0.5,
                                                      font=dict(size=20, color=gauge_color, family='Inter'), showarrow=False)])
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

            st.markdown('<div class="section-header">⚡ Risk Signals</div>', unsafe_allow_html=True)
            signals = []
            if contract == "Month-to-month":  signals.append(("📋 Month-to-month contract", "#ef4444"))
            if internet == "Fiber optic":     signals.append(("🌐 Fiber optic internet", "#f97316"))
            if tenure < 12:                   signals.append(("⏱️ Short tenure (< 1 year)", "#ef4444"))
            if monthly > 80:                  signals.append(("💸 High monthly charges", "#f97316"))
            if security == "No":              signals.append(("🔒 No online security", "#eab308"))
            if techsupport == "No":           signals.append(("🛠️ No tech support", "#eab308"))
            if payment == "Electronic check": signals.append(("💳 Electronic check payment", "#f97316"))
            if signals:
                for sig, col in signals[:6]:
                    st.markdown(f'<div style="padding:0.4rem 0.8rem;margin:0.3rem 0;background:rgba(0,0,0,0.3);border-left:3px solid {col};border-radius:6px;font-size:0.88rem;color:#cbd5e1">{sig}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#34d399;font-size:0.9rem;">✅ No major risk signals detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;padding:4rem 2rem;"><div style="font-size:4rem;">📡</div><div style="color:#64748b;">Fill in the customer details and click Run Churn Analysis</div></div>', unsafe_allow_html=True)

with tab2:
    fi_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True).tail(20)
    fig_fi = go.Figure(go.Bar(x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
                               marker=dict(color=fi_df['Importance'], colorscale=[[0,'#312e81'],[0.4,'#6366f1'],[1,'#c084fc']],
                                           showscale=True, colorbar=dict(tickfont=dict(color='#94a3b8'), title=dict(text='Score', font=dict(color='#94a3b8')))),
                               hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'))
    fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,13,28,0.6)', height=600,
                          margin=dict(l=20, r=20, t=40, b=20),
                          xaxis=dict(gridcolor='rgba(99,102,241,0.1)', tickfont=dict(color='#64748b'), title=dict(text='Importance Score', font=dict(color='#94a3b8'))),
                          yaxis=dict(tickfont=dict(color='#94a3b8')),
                          title=dict(text='Top 20 Most Important Features', font=dict(color='#e2e8f0', size=16), x=0.5))
    st.plotly_chart(fig_fi, use_container_width=True)
    top5 = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(5)
    st.markdown('<div class="section-header">🏆 Top 5 Predictors</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    colors = ['#818cf8','#a78bfa','#c084fc','#e879f9','#f0abfc']
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            st.markdown(f'<div class="metric-card"><div class="metric-label" style="font-size:0.65rem;">{row["Feature"]}</div><div class="metric-value" style="font-size:1.3rem;color:{colors[i]};">{row["Importance"]:.3f}</div></div>', unsafe_allow_html=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">⚙️ Hyperparameters</div>', unsafe_allow_html=True)
        param_df = pd.DataFrame(list(model.get_params().items()), columns=["Parameter", "Value"])
        st.dataframe(param_df, use_container_width=True, height=420, hide_index=True)
    with c2:
        st.markdown('<div class="section-header">📐 Tree Depth Distribution</div>', unsafe_allow_html=True)
        depths = [est.get_depth() for est in model.estimators_[:60]]
        fig_hist = go.Figure(go.Histogram(x=depths, nbinsx=12,
                                           marker=dict(color='rgba(99,102,241,0.7)', line=dict(color='rgba(129,140,248,1)', width=1))))
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,13,28,0.6)', height=220,
                                margin=dict(t=10, b=30, l=10, r=10),
                                xaxis=dict(title='Tree Depth', gridcolor='rgba(99,102,241,0.1)', tickfont=dict(color='#64748b'), title_font=dict(color='#94a3b8')),
                                yaxis=dict(title='Count', gridcolor='rgba(99,102,241,0.1)', tickfont=dict(color='#64748b'), title_font=dict(color='#94a3b8')))
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""<div style="background:rgba(15,23,42,0.8);border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:1rem 1.4rem;">
            <div style="color:#94a3b8;font-size:0.85rem;line-height:2.2;">
            🌲 <b style="color:#a5b4fc">Estimators:</b> {model.n_estimators}<br>
            📐 <b style="color:#a5b4fc">Features:</b> {model.n_features_in_}<br>
            🎯 <b style="color:#a5b4fc">Classes:</b> {", ".join(str(c) for c in model.classes_)}<br>
            🔀 <b style="color:#a5b4fc">Max Features:</b> {model.max_features}<br>
            📏 <b style="color:#a5b4fc">Max Depth:</b> {model.max_depth}<br>
            🌱 <b style="color:#a5b4fc">Criterion:</b> {model.criterion}<br>
            🔒 <b style="color:#a5b4fc">Bootstrap:</b> {model.bootstrap}
            </div></div>""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#334155;font-size:0.8rem;padding:0.5rem 0;">📡 ChurnRadar · Telecom Customer Intelligence · Random Forest Classifier</div>', unsafe_allow_html=True)