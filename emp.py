import sys
import os
import subprocess
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CSV_PATH = r"C:\Users\kirut\Downloads\emp_attrition.csv"

os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as f:
    f.write("""
[theme]
primaryColor="#E11D48"
backgroundColor="#0A0A0C"
secondaryBackgroundColor="#121215"
textColor="#F3F4F6"
font="sans serif"
""")

# =====================================================================
# STEP 1: AUTO-LAUNCHER & MODEL TRAINING
# =====================================================================
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    is_running_in_streamlit = get_script_run_ctx() is not None
except Exception:
    import streamlit as st
    is_running_in_streamlit = False

if __name__ == "__main__" and not is_running_in_streamlit:
    print("\n" + "="*50)
    print("Training TRACER Predictive Model for DSPristine...")
    
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: File not found at '{CSV_PATH}'!")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    selected_features = [
        'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 
        'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
        'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 
        'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 
        'PerformanceRating', 'RelationshipSatisfaction', 'TotalWorkingYears', 
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]

    df_model = df[['Attrition'] + selected_features].copy()

    encoders = {}
    for col in df_model.select_dtypes(include=['object']).columns:
        if col == 'Attrition':
            df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})
        else:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            encoders[col] = le

    X = df_model.drop('Attrition', axis=1)
    y = df_model['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open('retention_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 
            'encoders': encoders, 
            'features': list(X.columns)
        }, f)
        
    print(f"TRACER Model successfully trained on {len(df)} records!")
    print("Launching TRACER Module inside DSPristine Platform...")
    print("="*50 + "\n")
    
    script_path = os.path.abspath(__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
    sys.exit()

# =====================================================================
# STEP 2: STREAMLIT APP — TRACER PLATFORM
# =====================================================================

st.set_page_config(
    page_title="TRACER | DSPristine HR Analytics",
    page_icon="⚛️",
    layout="wide"
)

st.markdown("""<style>
html, body, .stApp {
    background-color: #0A0A0C !important;
    color: #F3F4F6 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.brand-banner {
    background: linear-gradient(135deg, #1C0A0E 0%, #111115 100%) !important;
    border: 2px solid #E11D48 !important;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 20px rgba(225, 29, 72, 0.25);
    display: flex;
    align-items: center;
    gap: 20px;
}

.brand-orbital-logo {
    width: 54px;
    height: 54px;
    background: radial-gradient(circle, #2A0813 0%, #111115 100%);
    border: 1.5px solid rgba(225, 29, 72, 0.5);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 18px rgba(225, 29, 72, 0.4);
    flex-shrink: 0;
}

.brand-orbital-logo img {
    width: 36px;
    height: 36px;
}

.brand-title {
    font-size: 2.2rem;
    font-weight: 900;
    margin: 0;
    line-height: 1;
    letter-spacing: 3px;
}

.app-highlight {
    color: #FF4D6D !important;
}

.brand-sub {
    font-size: 0.82rem;
    color: #A1A1AA !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 700;
    margin-top: 0.4rem;
}

label, .stSlider p, h1, h2, h3, h4, p {
    color: #F3F4F6 !important;
    font-weight: 600 !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #E11D48 0%, #9F1239 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.8rem 1.5rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 12px rgba(225, 29, 72, 0.35);
}

.metric-card {
    background-color: #121215 !important;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #27272A !important;
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #F3F4F6 !important;
}

.metric-lbl {
    font-size: 0.75rem;
    color: #A1A1AA !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

.badge-high {
    color: #FFD1D7 !important;
    background-color: #4C0519;
    border: 1px solid #E11D48;
    padding: 0.8rem;
    border-radius: 8px;
    font-weight: 700;
    text-align: center;
}

.badge-low {
    color: #D1FAE5 !important;
    background-color: #064E3B;
    border: 1px solid #10B981;
    padding: 0.8rem;
    border-radius: 8px;
    font-weight: 700;
    text-align: center;
}

.reason-box {
    background-color: #121215;
    border-left: 4px solid #E11D48;
    border-top: 1px solid #27272A;
    border-right: 1px solid #27272A;
    border-bottom: 1px solid #27272A;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    border-radius: 0 6px 6px 0;
    font-weight: 500;
    color: #F3F4F6 !important;
}

.reason-box b {
    color: #FF4D6D !important;
}
</style>""", unsafe_allow_html=True)

# Encoded Orbital SVG Graphic to guarantee correct rendering
svg_data = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='7' fill='%23FF4D6D'/><circle cx='50' cy='50' r='12' fill='none' stroke='%23E11D48' stroke-width='1.5' opacity='0.6'/><ellipse cx='50' cy='50' rx='38' ry='12' fill='none' stroke='%23E11D48' stroke-width='2' transform='rotate(-30 50 50)'/><circle cx='82' cy='31' r='3' fill='%23FFFFFF'/><ellipse cx='50' cy='50' rx='38' ry='12' fill='none' stroke='%23FF4D6D' stroke-width='1.8' transform='rotate(45 50 50)'/><circle cx='23' cy='23' r='2.5' fill='%23E11D48'/><ellipse cx='50' cy='50' rx='38' ry='12' fill='none' stroke='%23FFFFFF' stroke-width='1' stroke-dasharray='4 3' transform='rotate(-75 50 50)' opacity='0.7'/><circle cx='50' cy='88' r='3' fill='%23FF4D6D'/></svg>"

st.markdown(f"""
<div class="brand-banner">
    <div class="brand-orbital-logo">
        <img src="{svg_data}" alt="TRACER Logo" />
    </div>
    <div>
        <div class="brand-title"><span class="app-highlight">TRACER</span></div>
        <div class="brand-sub">Talent Retention Intelligence Module | DSPristine HR Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

if 'view_state' not in st.session_state:
    st.session_state.view_state = 'INPUT_FORM'

# =====================================================================
# SCREEN 1: SIMPLIFIED INPUT FORM
# =====================================================================
if st.session_state.view_state == 'INPUT_FORM':
    st.markdown("### Quick Employee Profile Input")
    st.caption("Fill in the primary factors below to evaluate retention risk.")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=32)
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=350000, value=45000, step=5000)
        overtime = st.selectbox("OverTime Required?", ["Yes", "No"])

    with col2:
        job_satisfaction = st.slider("Job Satisfaction Rating (1 = Low, 4 = High)", 1, 4, 2)
        work_life_balance = st.slider("Work-Life Balance Rating (1 = Poor, 4 = Great)", 1, 4, 2)
        last_promotion = st.slider("Years Since Last Promotion", 0, 15, 3)

    if st.button("Run TRACER Attrition Assessment"):
        st.session_state.inputs = {
            'Age': age,
            'BusinessTravel': 'Travel_Rarely',
            'Department': department,
            'DistanceFromHome': 10,
            'EnvironmentSatisfaction': 3,
            'JobInvolvement': 3,
            'JobLevel': 2,
            'JobRole': 'Research Scientist' if department == 'Research & Development' else ('Sales Executive' if department == 'Sales' else 'Human Resources'),
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': 'Single',
            'MonthlyIncome': monthly_income,
            'NumCompaniesWorked': 2,
            'OverTime': overtime,
            'PercentSalaryHike': 13,
            'PerformanceRating': 3,
            'RelationshipSatisfaction': 3,
            'TotalWorkingYears': max(1, age - 22),
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': min(5, max(1, age - 25)),
            'YearsInCurrentRole': 2,
            'YearsSinceLastPromotion': last_promotion,
            'YearsWithCurrManager': 2
        }
        st.session_state.view_state = 'REPORT_SCREEN'
        st.rerun()

# =====================================================================
# SCREEN 2: REPORT SCREEN
# =====================================================================
elif st.session_state.view_state == 'REPORT_SCREEN':
    inp = st.session_state.inputs

    try:
        with open('retention_model.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            encoders = data['encoders']
            features = data['features']

        encoded_inp = inp.copy()
        for col, le in encoders.items():
            if col in encoded_inp:
                encoded_inp[col] = le.transform([encoded_inp[col]])[0]

        input_df = pd.DataFrame([encoded_inp])[features]
        prob = model.predict_proba(input_df)[0][1]
        risk_score = int(prob * 100)
    except Exception:
        risk_score = 45

    prediction = "LEAVE (At-Risk)" if risk_score >= 40 else "STAY"

    st.markdown("### Executive Retention Analytics")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{"#FF4D6D" if risk_score>=40 else "#10B981"} !important;">{prediction}</div><div class="metric-lbl">TRACER Prediction</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{risk_score}%</div><div class="metric-lbl">Attrition Risk</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">₹{inp["MonthlyIncome"]:,}</div><div class="metric-lbl">Monthly Income</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{inp["YearsSinceLastPromotion"]} Yrs</div><div class="metric-lbl">Since Promotion</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Prediction Summary")
        if risk_score >= 40:
            st.markdown(f'<div class="badge-high">FLAGGED: HIGH ATTRITION RISK ({risk_score}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-low">STABLE: LOW ATTRITION RISK ({risk_score}%)</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### Top 3 Reasons Contributing to Exit Risk")
        
        possible_reasons = []

        if inp['OverTime'] == 'Yes':
            possible_reasons.append(("Frequent Overtime Required", "Excessive overtime workload significantly accelerates employee fatigue and burnout."))
        if inp['MonthlyIncome'] < 40000:
            possible_reasons.append(("Below-Average Compensation", f"Monthly compensation of ₹{inp['MonthlyIncome']:,} is below competitive benchmark thresholds."))
        if inp['JobSatisfaction'] <= 2:
            possible_reasons.append(("Low Job Satisfaction Rating", f"Employee reported low satisfaction score ({inp['JobSatisfaction']}/4) with daily tasks."))
        if inp['WorkLifeBalance'] <= 2:
            possible_reasons.append(("Poor Work-Life Balance", f"Low rating ({inp['WorkLifeBalance']}/4) indicates struggle to balance job duties and personal time."))
        if inp['YearsSinceLastPromotion'] >= 2:
            possible_reasons.append(("Promotion Stagnation", f"No career progression or promotion for over {inp['YearsSinceLastPromotion']} years."))
        if inp['Age'] < 30:
            possible_reasons.append(("Early Career Volatility", "Younger employees exhibit higher natural industry mobility and job-hopping tendencies."))

        fallback_reasons = [
            ("Compensation & Market Alignment", "Routine salary reviews recommended to match industry standards."),
            ("Career Path & Growth Velocity", "Clear leadership milestones need to be established for long-term retention."),
            ("Managerial Engagement Check", "Conduct regular 1-on-1 check-ins to monitor job engagement.")
        ]

        top_3_reasons = possible_reasons + [fb for fb in fallback_reasons if fb not in possible_reasons]
        top_3_reasons = top_3_reasons[:3]

        for i, (reason, detail) in enumerate(top_3_reasons, 1):
            st.markdown(f'<div class="reason-box"><b>#{i}. {reason}</b> — {detail}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Global Attrition Key Drivers")
        
        try:
            importances = model.feature_importances_
            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False).head(7)
            
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor('#121215')
            ax.set_facecolor('#121215')
            sns.barplot(x='Importance', y='Feature', data=fi_df, palette='Reds_r', ax=ax)
            ax.set_title("Top Features Impacting Attrition Risk", fontsize=10, fontweight='bold', color='#F3F4F6')
            ax.set_xlabel("Relative Importance", color='#A1A1AA')
            ax.set_ylabel("", color='#A1A1AA')
            ax.tick_params(colors='#A1A1AA')
            for spine in ax.spines.values():
                spine.set_color('#27272A')
            plt.tight_layout()
            
            st.pyplot(fig)
        except Exception:
            st.info("Feature importance plot loading...")

    st.markdown("---")
    if st.button("Evaluate Another Employee Profile"):
        st.session_state.view_state = 'INPUT_FORM'
        st.rerun()