# ----------------------------------------------------------
# 🏥 Hospital AI Decision Support System (Business Edition + Triage Colors)
# ----------------------------------------------------------
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------
# ⚙️ Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Hospital AI Decision Support", page_icon="🏥", layout="wide")
st.title("Hospital AI for Clinical Decision Support")
st.caption("ระบบสนับสนุนการตัดสินใจทางการแพทย์และการบริหารทรัพยากรโรงพยาบาล")

# ----------------------------------------------------------
# 📦 Load Models + Show in Expander
# ----------------------------------------------------------
@st.cache_resource
def load_all():
    msgs = []  # Store messages to show in the expander

    # 🔹 CatBoost Model
    try:
        model = joblib.load("predict_catboost_multi.pkl")
        msgs.append("✅ predict_catboost_multi.pkl — Clinical Severity Model")
    except FileNotFoundError:
        model = None
        msgs.append("❌ predict_catboost_multi.pkl not found")

    # 🔹 Encoders
    try:
        encoders = joblib.load("encoders_multi.pkl")
        msgs.append("✅ encoders_multi.pkl — Encoders for Clinical Data")
    except FileNotFoundError:
        encoders = None
        msgs.append("⚠️ encoders_multi.pkl not found")

    # 🔹 Features
    try:
        with open("features_multi.json", "r") as f:
            features = json.load(f)
        msgs.append("✅ features_multi.json — Model Features Configuration")
    except FileNotFoundError:
        features = []
        msgs.append("⚠️ features_multi.json not found")

    # 🔹 K-Means
    try:
        kmeans = joblib.load("kmeans_cluster_model.pkl")
        scaler = joblib.load("scaler_cluster.pkl")
        msgs.append("✅ kmeans_cluster_model.pkl / scaler_cluster.pkl — Clustering Models")
    except FileNotFoundError:
        kmeans = scaler = None
        msgs.append("⚠️ K-Means / Scaler files not found")

    # 🔹 Apriori
    try:
        rules_minor = joblib.load("apriori_rules_minor.pkl")
        rules_severe = joblib.load("apriori_rules_severe.pkl")
        rules_fatal = joblib.load("apriori_rules_fatal.pkl")
        msgs.append("✅ apriori_rules_[minor/severe/fatal].pkl — Risk Pattern Mining Rules")
    except FileNotFoundError:
        rules_minor = rules_severe = rules_fatal = None
        msgs.append("⚠️ Apriori rules files not found")

    # ✅ Display loaded files in an expander
    with st.expander("📂 Loaded Files Status", expanded=False):
        for m in msgs:
            st.caption(m)

    return model, encoders, features, kmeans, scaler, rules_minor, rules_severe, rules_fatal


# Load all necessary files
model, encoders, features, kmeans, scaler, rules_minor, rules_severe, rules_fatal = load_all()


# ----------------------------------------------------------
# 🧩 Manual Mappings
# ----------------------------------------------------------
activity_mapping = {
    "0": "Walking", "1": "Public Transport", "2": "Private Vehicle",
    "3": "Driving", "4": "Working", "5": "Sports", "6": "Other Activities"
}
aplace_mapping = {
    "10": "Home", "11": "Road/Highway", "12": "Workplace",
    "13": "School/Institution", "14": "Public Area", "15": "Other"
}
prov_mapping = {
    "10": "Bangkok", "20": "Chiang Mai", "30": "Khon Kaen",
    "40": "Phuket", "50": "Nakhon Ratchasima", "60": "Songkhla", "99": "Other"
}
severity_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
advice_map = {
    "Low Risk": "General observation, re-evaluate every 15–30 minutes.",
    "Medium Risk": "Send for further investigation, provide fluids/painkillers, monitor vital signs closely.",
    "High Risk": "Alert multidisciplinary team, secure airway, prepare ER or urgent transfer."
}
triage_color = {
    "Low Risk": "#4CAF50",      # Green
    "Medium Risk": "#FFC107",  # Yellow
    "High Risk": "#F44336"     # Red
}

# ==========================================================
# 🩺 TAB SYSTEM
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Clinical Risk Prediction",
    "👥 Cluster Insight",
    "🧩 Risk Association",
    "📊 Clinical Summary Dashboard"
])

# Initialize session state for prediction results
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# ----------------------------------------------------------
# 🧠 TAB 1 — CatBoost Prediction
# ----------------------------------------------------------
with tab1:
    st.subheader("🧠 Clinical Severity Prediction")
    st.caption("Real-time assessment system for patient injury severity.")

    st.subheader("📋 Patient Information")
    with st.form("input_form"):
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=35)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        is_night = st.checkbox("Occurred at night", value=False)
        head_injury = st.checkbox("Head injury", value=False)
        mass_casualty = st.checkbox("Mass casualty incident", value=False)

        st.markdown("**Risk Factors**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: risk1 = st.checkbox("No helmet/seatbelt")
        with c2: risk2 = st.checkbox("Speeding/Reckless driving")
        with c3: risk3 = st.checkbox("Drunk driving")
        with c4: risk4 = st.checkbox("Elderly/Young child")
        with c5: risk5 = st.checkbox("Multiple injuries")

        st.markdown("**Substances Detected**")
        d1, d2, d3 = st.columns(3)
        with d1: cannabis = st.checkbox("Cannabis")
        with d2: amphetamine = st.checkbox("Amphetamine")
        with d3: drugs = st.checkbox("Other drugs")

        st.markdown("**Incident Context**")
        activity = st.selectbox("Activity at time of incident", list(activity_mapping.values()))
        aplace = st.selectbox("Place of incident", list(aplace_mapping.values()))
        prov = st.selectbox("Province of incident", list(prov_mapping.values()))
        submit = st.form_submit_button("🔎 Assess Risk Level")

    def preprocess_input(data_dict):
        df = pd.DataFrame([data_dict])
        reverse_activity = {v: k for k, v in activity_mapping.items()}
        reverse_aplace = {v: k for k, v in aplace_mapping.items()}
        reverse_prov = {v: k for k, v in prov_mapping.items()}
        df["activity"] = df["activity"].map(reverse_activity)
        df["aplace"] = df["aplace"].map(reverse_aplace)
        df["prov"] = df["prov"].map(reverse_prov)

        for col in ["activity", "aplace", "prov"]:
            val = str(df.at[0, col])
            if encoders and col in encoders:
                le = encoders[col]
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = -1 # Use -1 for unknown categories
            else:
                df[col] = -1

        df["age_group_60plus"] = (df["age"] >= 60).astype(int)
        df["risk_count"] = df[["risk1","risk2","risk3","risk4","risk5"]].sum(axis=1)
        df["night_flag"] = df["is_night"].astype(int)
        df = df.reindex(columns=features, fill_value=0)
        return df

    if submit:
        input_data = {
            "age": age, "sex": 1 if sex == "Male" else 0,
            "is_night": int(is_night), "head_injury": int(head_injury),
            "mass_casualty": int(mass_casualty),
            "risk1": int(risk1), "risk2": int(risk2), "risk3": int(risk3),
            "risk4": int(risk4), "risk5": int(risk5),
            "cannabis": int(cannabis), "amphetamine": int(amphetamine), "drugs": int(drugs),
            "activity": activity, "aplace": aplace, "prov": prov
        }

        X_input = preprocess_input(input_data)
        if model is not None:
            probs = model.predict_proba(X_input)[0]
            pred_class = int(np.argmax(probs))
            label = severity_map.get(pred_class, "Unknown")
            color = triage_color.get(label, "#2196F3")

            # Store result in session state
            st.session_state.prediction_result = {
                'label': label,
                'color': color,
                'advice': advice_map.get(label),
                'confidence': probs[pred_class],
                'input_data': input_data,
                'X_input': X_input
            }

            # Log prediction to file
            log_file = "prediction_log.csv"
            new_row = pd.DataFrame([{
                "timestamp": pd.Timestamp.now(), "age": age, "sex": sex, "predicted_severity": label
            }])
            if os.path.exists(log_file):
                new_row.to_csv(log_file, mode="a", index=False, header=False)
            else:
                new_row.to_csv(log_file, index=False)
            st.success("📁 Assessment result saved to the Dashboard system.")
        else:
            st.error("Model is not loaded. Cannot perform prediction.")

    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        st.markdown(
            f"<div style='background-color:{res['color']};padding:12px;border-radius:10px;'>"
            f"<h3 style='color:white;text-align:center;'>Predicted Risk Level: {res['label']}</h3></div>",
            unsafe_allow_html=True
        )
        st.info(f"💡 Preliminary Medical Guideline: {res['advice']}")
        st.caption(f"🧠 System Confidence: {res['confidence']*100:.1f}%")

# ----------------------------------------------------------
# 👥 TAB 2 — K-Means Cluster Analysis
# ----------------------------------------------------------
with tab2:
    st.subheader("👥 Patient Segmentation")
    st.caption("Analyze patient groups for resource allocation and proactive prevention.")

    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        data = res['input_data']

        st.markdown("### 🧾 Patient Information Summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Age", f"{data['age']} years")
        summary_cols[1].metric("Sex", "Male" if data['sex'] == 1 else "Female")
        summary_cols[2].metric("Predicted Risk Level", res['label'])

        risk_summary = [name for name, val in {"❗ No helmet": data['risk1'], "⚡ Speeding": data['risk2'], "🍺 Drunk driving": data['risk3'], "👶 Child/Elderly": data['risk4'], "🩸 Multiple injuries": data['risk5']}.items() if val]
        st.markdown(f"**Risk Factors:** {', '.join(risk_summary) if risk_summary else '-'}")

        if kmeans and scaler:
            X_input = res['X_input']
            if hasattr(scaler, "feature_names_in_"):
                valid_cols = [c for c in scaler.feature_names_in_ if c in X_input.columns]
                X_cluster = X_input[valid_cols]
            else:
                X_cluster = X_input.select_dtypes(include=[np.number])

            X_scaled = scaler.transform(X_cluster)
            cluster_label = int(kmeans.predict(X_scaled)[0])

            cluster_desc = {
                0: "👵 Elderly / Fall at home → Low Risk",
                1: "🚗 Working Age / DUI / Speeding → High Risk",
                2: "⚽ Child/Adolescent / Sports Injury → Medium Risk",
                3: "👷 Laborer / Construction Accident → High Risk",
                4: "🙂 General / No dominant factor → Low Risk"
            }
            st.markdown("---")
            st.markdown(f"### 📊 Patient Grouping Result: **Cluster {cluster_label}**")
            st.info(f"{cluster_desc.get(cluster_label, 'No description for this cluster yet.')}")
            st.caption("💡 Use this for trend analysis and resource planning (e.g., ER teams, doctor shifts, accident prevention campaigns).")
        else:
            st.warning("⚠️ K-Means or Scaler model file not found. Cannot perform segmentation.")
    else:
        st.info("🕐 Once you assess a patient's risk in the first tab, the results will appear here.")

# ----------------------------------------------------------
# 🧩 TAB 3 — Apriori Risk Association
# ----------------------------------------------------------
with tab3:
    st.subheader("🧩 Risk Association Analysis")
    st.caption("Analyze risk factor relationships to plan prevention and support policy decisions.")

    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        data = res['input_data']
        label = res['label']

        st.markdown("### 🧾 Patient Information Summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Age", f"{data['age']} years")
        summary_cols[1].metric("Sex", "Male" if data['sex'] == 1 else "Female")
        summary_cols[2].metric("Risk Level", label)

        risk_tags = [name for name, val in {"No helmet": data['risk1'], "Speeding": data['risk2'], "Drunk driving": data['risk3'], "Elderly/Child": data['risk4'], "Multiple injuries": data['risk5']}.items() if val]
        st.markdown("**Risk Factors:** " + (", ".join(risk_tags) if risk_tags else "No dominant risk factors"))
        st.markdown("---")

        def decode_set(x):
            if isinstance(x, (frozenset, set)):
                replacements = {
                    "risk1": "No helmet/seatbelt", "risk2": "Speeding", "risk3": "Drunk driving",
                    "risk4": "Elderly/Child", "risk5": "Multiple injuries", "head_injury": "Head injury",
                    "mass_casualty": "Mass casualty", "cannabis": "Cannabis detected",
                    "amphetamine": "Amphetamine detected", "drugs": "Other drugs detected",
                    "sex": "Male", "age60plus": "Age > 60"
                }
                return ", ".join([replacements.get(str(i), str(i)) for i in x])
            return str(x)

        rules_map = {"Low Risk": rules_minor, "Medium Risk": rules_severe, "High Risk": rules_fatal}
        selected_rules = rules_map.get(label)

        if selected_rules is not None and not selected_rules.empty:
            df_rules = selected_rules.head(5).copy()
            df_rules["antecedents"] = df_rules["antecedents"].apply(decode_set)
            df_rules["consequents"] = df_rules["consequents"].apply(decode_set)

            top_rule = df_rules.iloc[0]
            st.markdown(
                f"""
                <div style='background-color:#1E1E1E;border-radius:10px;padding:12px;margin-bottom:10px'>
                💡 <b>Insight:</b> Patients with <b style='color:#FFC107'>{top_rule['antecedents']}</b> 
                tend to also have <b style='color:#FF5252'>{top_rule['consequents']}</b> 
                (Confidence {top_rule['confidence']*100:.1f}%, Lift = {top_rule['lift']:.2f})
                </div>
                """, unsafe_allow_html=True
            )
            st.dataframe(
                df_rules[["antecedents", "consequents", "support", "confidence", "lift"]],
                use_container_width=True, hide_index=True
            )
            st.markdown("📘 **Interpretation:**\n- **Support:** Frequency of the rule in the dataset.\n- **Confidence:** Probability that the rule is true.\n- **Lift > 1:** Indicates the relationship is more significant than random chance.")
        else:
            st.info("📭 No association rules available for this risk category.")
    else:
        st.info("🕐 Once you assess a patient's risk in the first tab, the results will appear here.")


# ----------------------------------------------------------
# 📊 TAB 4 — Clinical Summary & Insights Dashboard
# ----------------------------------------------------------
with tab4:
    st.subheader("📊 Clinical Summary & Insights")
    st.caption("Dashboard summarizing AI-driven patient trends for strategic planning and hospital resource management.")

    # Load or Reset Log File
    log_file = "prediction_log.csv"
    c1, c2 = st.columns([4,1])
    with c1:
        st.markdown("#### 🩺 Summary of All Assessments")
    with c2:
        if st.button("🧹 Clear All Data"):
            if os.path.exists(log_file):
                os.remove(log_file)
                st.success("✅ Data cleared successfully. You can start collecting new data.")
            else:
                st.info("⚠️ No data to clear.")

    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        st.success(f"📁 Loaded {len(df_log):,} records successfully.")
    else:
        st.warning("⚠️ No prediction data available yet.")
        df_log = pd.DataFrame(columns=["timestamp", "age", "sex", "predicted_severity"])

    total_cases = len(df_log)

    # Only display the dashboard if there is data
    if total_cases > 0:
        # KPI Overview
        st.markdown("### 💡 KPI Overview")
        kpi1, kpi2, kpi3 = st.columns(3)

        severe_ratio = df_log["predicted_severity"].eq("High Risk").mean() * 100
        male_ratio = (df_log["sex"] == "Male").mean() * 100
        female_ratio = 100 - male_ratio

        kpi1.metric("Total Cases", f"{total_cases:,}")
        kpi2.metric("High-Risk Patient Ratio", f"{severe_ratio:.1f}%")
        kpi3.metric("Male : Female Ratio", f"{male_ratio:.1f}% : {female_ratio:.1f}%")

        # Distribution by Severity (Pie Chart)
        st.markdown("### 🩸 Patient Distribution by Risk Level")
        fig, ax = plt.subplots(figsize=(4, 4))
        # Define colors based on actual labels in the data to avoid errors
        severity_counts = df_log['predicted_severity'].value_counts()
        pie_colors = [triage_color.get(label, '#808080') for label in severity_counts.index]

        severity_counts.plot.pie(
            autopct='%1.1f%%', startangle=90, colors=pie_colors, ax=ax,
            textprops={'color': 'white', 'fontsize': 10}
        )
        ax.set_ylabel('')
        ax.set_title("Patient Severity Levels", color='white', fontsize=12)
        st.pyplot(fig)

        st.markdown("---")

        # Insight Summary
        st.markdown("### 🩺 Clinical Insights & Strategic Recommendations")
        top_severity = df_log["predicted_severity"].value_counts().idxmax()
        if top_severity == "High Risk":
            msg = "Trend shows a high number of severe injuries. Consider allocating more ER resources during peak incident times."
        elif top_severity == "Medium Risk":
            msg = "The majority of patients are at medium risk. Emphasize close monitoring and re-assessment protocols."
        else:
            msg = "Most cases are low-risk. Focus can be directed towards public education and prevention campaigns."
        st.info(f"📊 Current Situation Summary: {msg}")
        st.caption("💡 Use this data to support prioritization and hospital resource management.")
    else:
        st.info("Waiting for first prediction to generate dashboard...")
