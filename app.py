# ----------------------------------------------------------
# 🏥 Hospital AI Decision Support System (Refactored)
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
# 📦 Load Models and Cached Resources
# ----------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Loads all models and necessary files, caching them for performance."""
    resources = {}
    msgs = []

    try:
        resources["model"] = joblib.load("predict_catboost_multi.pkl")
        msgs.append("✅ predict_catboost_multi.pkl — Clinical Severity Model")
    except FileNotFoundError:
        resources["model"] = None
        msgs.append("❌ ไม่พบ predict_catboost_multi.pkl")

    try:
        resources["encoders"] = joblib.load("encoders_multi.pkl")
        msgs.append("✅ encoders_multi.pkl — Encoders for Clinical Data")
    except FileNotFoundError:
        resources["encoders"] = None
        msgs.append("⚠️ ไม่พบ encoders_multi.pkl")

    try:
        with open("features_multi.json", "r") as f:
            resources["features"] = json.load(f)
        msgs.append("✅ features_multi.json — Model Features Configuration")
    except FileNotFoundError:
        resources["features"] = []
        msgs.append("⚠️ ไม่พบ features_multi.json")

    try:
        resources["kmeans"] = joblib.load("kmeans_cluster_model.pkl")
        resources["scaler"] = joblib.load("scaler_cluster.pkl")
        msgs.append("✅ kmeans_cluster_model.pkl / scaler_cluster.pkl — Clustering Models")
    except FileNotFoundError:
        resources["kmeans"] = resources["scaler"] = None
        msgs.append("⚠️ ไม่พบไฟล์ K-Means / Scaler")

    try:
        resources["rules_minor"] = joblib.load("apriori_rules_minor.pkl")
        resources["rules_severe"] = joblib.load("apriori_rules_severe.pkl")
        resources["rules_fatal"] = joblib.load("apriori_rules_fatal.pkl")
        msgs.append("✅ apriori_rules_[minor/severe/fatal].pkl — Risk Pattern Mining Rules")
    except FileNotFoundError:
        resources["rules_minor"] = resources["rules_severe"] = resources["rules_fatal"] = None
        msgs.append("⚠️ ไม่พบไฟล์กฎ Apriori")

    with st.expander("📂 สถานะการโหลดไฟล์โมเดล", expanded=False):
        for m in msgs:
            st.caption(m)
    return resources

# Load all resources
resources = load_all_models()

# ----------------------------------------------------------
# 🧩 Manual Mappings and Constants
# ----------------------------------------------------------
# (Mappings remain unchanged)
activity_mapping = {"0": "เดินเท้า", "1": "โดยสารพาหนะสาธารณะ", "2": "โดยสารพาหนะส่วนบุคคล", "3": "ขับขี่พาหนะส่วนบุคคล", "4": "ทำงาน", "5": "เล่นกีฬา", "6": "กิจกรรมอื่น ๆ"}
aplace_mapping = {"10": "บ้านพักอาศัย", "11": "ถนน/ทางหลวง", "12": "สถานที่ทำงาน", "13": "โรงเรียน/สถาบันศึกษา", "14": "พื้นที่สาธารณะ", "15": "อื่น ๆ"}
prov_mapping = {"10": "กรุงเทพมหานคร", "20": "เชียงใหม่", "30": "ขอนแก่น", "40": "ภูเก็ต", "50": "นครราชสีมา", "60": "สงขลา", "99": "อื่น ๆ"}
severity_map = {0: "เสี่ยงน้อย", 1: "เสี่ยงปานกลาง", 2: "เสี่ยงมาก"}
advice_map = {"เสี่ยงน้อย": "ดูแลอาการทั่วไป เฝ้าระวังซ้ำทุก 15–30 นาที", "เสี่ยงปานกลาง": "ส่งตรวจเพิ่มเติม ให้สารน้ำ / ยาแก้ปวด / เฝ้าสัญญาณชีพใกล้ชิด", "เสี่ยงมาก": "แจ้งทีมสหสาขา เปิดทางเดินหายใจ เตรียมห้องฉุกเฉินหรือส่งต่อด่วน"}
triage_color = {"เสี่ยงน้อย": "#4CAF50", "เสี่ยงปานกลาง": "#FFC107", "เสี่ยงมาก": "#F44336"}


# ----------------------------------------------------------
# 📝 Helper Function for Preprocessing
# ----------------------------------------------------------
def preprocess_input(data_dict, encoders, features):
    df = pd.DataFrame([data_dict])
    # Reverse map string values back to keys for processing
    reverse_activity = {v: k for k, v in activity_mapping.items()}
    reverse_aplace = {v: k for k, v in aplace_mapping.items()}
    reverse_prov = {v: k for k, v in prov_mapping.items()}
    df["activity"] = df["activity"].map(reverse_activity)
    df["aplace"] = df["aplace"].map(reverse_aplace)
    df["prov"] = df["prov"].map(reverse_prov)
    
    # Apply encoders
    for col in ["activity", "aplace", "prov"]:
        val = str(df.at[0, col])
        if encoders and col in encoders:
            le = encoders[col]
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = -1 # Handle unknown category
        else:
            df[col] = -1

    # Feature Engineering
    df["age_group_60plus"] = (df["age"] >= 60).astype(int)
    df["risk_count"] = df[["risk1","risk2","risk3","risk4","risk5"]].sum(axis=1)
    df["night_flag"] = df["is_night"].astype(int)
    
    # Ensure all required features are present
    df = df.reindex(columns=features, fill_value=0)
    return df

# ==========================================================
# 🩺 TAB 1: Clinical Risk Prediction
# ==========================================================
def render_tab1():
    st.subheader("🧠 Clinical Severity Prediction")
    st.caption("ระบบประเมินระดับความรุนแรงของผู้บาดเจ็บแบบเรียลไทม์")

    with st.form("input_form"):
        st.subheader("📋 ข้อมูลผู้บาดเจ็บ")
        # --- Input Fields (same as before) ---
        age = st.number_input("อายุ (ปี)", min_value=0, max_value=120, value=35)
        sex = st.radio("เพศ", ["หญิง", "ชาย"], horizontal=True)
        is_night = st.checkbox("เกิดเหตุเวลากลางคืน", value=False)
        head_injury = st.checkbox("บาดเจ็บที่ศีรษะ", value=False)
        mass_casualty = st.checkbox("เหตุการณ์ผู้บาดเจ็บจำนวนมาก", value=False)
        st.markdown("**ปัจจัยเสี่ยง (Risk Factors)**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: risk1 = st.checkbox("ไม่สวมหมวกนิรภัย / เข็มขัดนิรภัย")
        with c2: risk2 = st.checkbox("ขับรถเร็ว / ประมาท")
        with c3: risk3 = st.checkbox("เมา / ดื่มสุรา")
        with c4: risk4 = st.checkbox("ผู้สูงอายุ / เด็กเล็ก")
        with c5: risk5 = st.checkbox("บาดเจ็บหลายตำแหน่ง")
        st.markdown("**สารเสพติด/ยาในร่างกาย**")
        d1, d2, d3 = st.columns(3)
        with d1: cannabis = st.checkbox("กัญชา")
        with d2: amphetamine = st.checkbox("ยาบ้า / แอมเฟตามีน")
        with d3: drugs = st.checkbox("ยาอื่น ๆ")
        st.markdown("**บริบทของเหตุการณ์**")
        activity = st.selectbox("กิจกรรมขณะเกิดเหตุ", list(activity_mapping.values()))
        aplace = st.selectbox("สถานที่เกิดเหตุ", list(aplace_mapping.values()))
        prov = st.selectbox("จังหวัดที่เกิดเหตุ", list(prov_mapping.values()))
        
        submit = st.form_submit_button("🔎 ประเมินระดับความเสี่ยง")

    if submit:
        input_data = {
            "age": age, "sex": 1 if sex == "ชาย" else 0,
            "is_night": int(is_night), "head_injury": int(head_injury), "mass_casualty": int(mass_casualty),
            "risk1": int(risk1), "risk2": int(risk2), "risk3": int(risk3), "risk4": int(risk4), "risk5": int(risk5),
            "cannabis": int(cannabis), "amphetamine": int(amphetamine), "drugs": int(drugs),
            "activity": activity, "aplace": aplace, "prov": prov
        }
        
        model = resources.get("model")
        if model:
            X_input = preprocess_input(input_data, resources.get("encoders"), resources.get("features"))
            probs = model.predict_proba(X_input)[0]
            pred_class = int(np.argmax(probs))
            label = severity_map.get(pred_class, "ไม่ทราบ")
            
            # Save result to session state to share with other tabs
            st.session_state['prediction_result'] = {
                'label': label, 'color': triage_color.get(label),
                'advice': advice_map.get(label), 'confidence': probs[pred_class],
                'input_data': input_data, 'X_input': X_input
            }

            # Log prediction to file
            log_file = "prediction_log.csv"
            new_row = pd.DataFrame([{"timestamp": pd.Timestamp.now(), "age": age, "sex": sex, "predicted_severity": label}])
            new_row.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
            st.success("📁 บันทึกผลการประเมินเข้าสู่ระบบ Dashboard แล้ว")
        else:
            st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์")

    # Display result if available in session state
    if 'prediction_result' in st.session_state:
        res = st.session_state['prediction_result']
        st.markdown(f"<div style='background-color:{res['color']};padding:12px;border-radius:10px;'><h3 style='color:white;text-align:center;'>ระดับความเสี่ยงที่คาดการณ์: {res['label']}</h3></div>", unsafe_allow_html=True)
        st.info(f"💡 แนวทางทางการแพทย์เบื้องต้น: {res['advice']}")
        st.caption(f"🧠 ความมั่นใจของระบบ: {res['confidence']*100:.1f}%")

# ==========================================================
# 👥 TAB 2: Cluster Insight
# ==========================================================
def render_tab2():
    st.subheader("👥 Patient Segmentation")
    st.caption("วิเคราะห์กลุ่มผู้บาดเจ็บ เพื่อใช้ในการจัดสรรทรัพยากรและการป้องกันเชิงรุก")

    if 'prediction_result' not in st.session_state:
        st.info("🕐 กรุณาประเมินความเสี่ยงในแท็บแรกก่อน เพื่อดูผลการวิเคราะห์กลุ่มผู้ป่วยที่นี่")
        return

    res = st.session_state['prediction_result']
    data = res['input_data']

    # --- Display patient summary ---
    st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
    summary_cols = st.columns(3)
    summary_cols[0].metric("อายุ", f"{data['age']} ปี")
    summary_cols[1].metric("เพศ", "ชาย" if data['sex'] == 1 else "หญิง")
    summary_cols[2].metric("ระดับความเสี่ยงที่คาดการณ์", res['label'])

    # --- Clustering analysis ---
    kmeans, scaler = resources.get("kmeans"), resources.get("scaler")
    if kmeans and scaler:
        X_cluster = res['X_input']
        if hasattr(scaler, "feature_names_in_"):
            valid_cols = [c for c in scaler.feature_names_in_ if c in X_cluster.columns]
            X_cluster = X_cluster[valid_cols]
        
        X_scaled = scaler.transform(X_cluster)
        cluster_label = int(kmeans.predict(X_scaled)[0])

        cluster_desc = {
            0: "👵 กลุ่มผู้สูงอายุ / ลื่นล้มในบ้าน → ความเสี่ยงต่ำ",
            1: "🚗 กลุ่มวัยทำงาน / เมา / ขับรถเร็ว → ความเสี่ยงสูง",
            2: "⚽ กลุ่มเด็กและวัยรุ่น / เล่นกีฬา → ความเสี่ยงปานกลาง",
            3: "👷 กลุ่มแรงงาน / ก่อสร้าง → ความเสี่ยงสูง",
            4: "🙂 กลุ่มทั่วไป / ไม่มีปัจจัยเด่น → ความเสี่ยงต่ำ"
        }
        st.markdown("---")
        st.markdown(f"### 📊 ผลการจัดกลุ่มผู้บาดเจ็บ: **Cluster {cluster_label}**")
        st.info(f"{cluster_desc.get(cluster_label, 'ยังไม่มีคำอธิบายกลุ่มนี้')}")
    else:
        st.warning("⚠️ ไม่พบไฟล์โมเดล K-Means / Scaler")

# ==========================================================
# 🧩 TAB 3: Risk Association
# ==========================================================
def render_tab3():
    st.subheader("🧩 Risk Association Analysis")
    st.caption("วิเคราะห์ความสัมพันธ์ของปัจจัยเสี่ยง เพื่อวางแผนป้องกันและสนับสนุนการตัดสินใจเชิงนโยบาย")
    
    if 'prediction_result' not in st.session_state:
        st.info("🕐 กรุณาประเมินความเสี่ยงในแท็บแรกก่อน เพื่อดูกฎความสัมพันธ์ที่เกี่ยวข้อง")
        return

    res = st.session_state['prediction_result']
    label = res['label']
    
    rules_map = {"เสี่ยงน้อย": resources.get("rules_minor"), "เสี่ยงปานกลาง": resources.get("rules_severe"), "เสี่ยงมาก": resources.get("rules_fatal")}
    df_rules = rules_map.get(label)

    if df_rules is not None and not df_rules.empty:
        st.markdown(f"### 🔗 กฎความสัมพันธ์ที่พบบ่อยสำหรับกลุ่ม '{label}'")
        # Display top 5 rules
        st.dataframe(df_rules.head(5), use_container_width=True)
    else:
        st.info(f"📭 ไม่พบกฎความสัมพันธ์สำหรับกลุ่ม '{label}'")

# ==========================================================
# 📊 TAB 4: Clinical Summary Dashboard
# ==========================================================
def render_tab4():
    st.subheader("📊 Clinical Summary & Insights Dashboard")
    st.caption("สรุปแนวโน้มผู้บาดเจ็บจากระบบ AI เพื่อใช้วางแผนเชิงกลยุทธ์และบริหารทรัพยากรโรงพยาบาล")

    log_file = "prediction_log.csv"
    
    if st.button("🧹 ล้างข้อมูล Dashboard ทั้งหมด"):
        if os.path.exists(log_file):
            os.remove(log_file)
            st.success("✅ ล้างข้อมูลเรียบร้อยแล้ว!")
            # Use st.experimental_rerun() to refresh the page immediately after deleting the file
            st.experimental_rerun() 
        else:
            st.info("⚠️ ยังไม่มีข้อมูลให้ลบ")

    if not os.path.exists(log_file):
        st.warning("⚠️ ยังไม่มีข้อมูลจากการทำนายเพื่อสร้าง Dashboard")
        return

    df_log = pd.read_csv(log_file)
    total_cases = len(df_log)
    
    if total_cases > 0:
        st.markdown("### 💡 ภาพรวมสถานการณ์ (KPI Overview)")
        c1, c2, c3 = st.columns(3)
        severe_ratio = df_log["predicted_severity"].eq("เสี่ยงมาก").mean() * 100
        male_ratio = (df_log["sex"] == "ชาย").mean() * 100
        
        c1.metric("จำนวนเคสทั้งหมด", f"{total_cases:,}")
        c2.metric("สัดส่วนผู้บาดเจ็บรุนแรง (เสี่ยงมาก)", f"{severe_ratio:.1f}%")
        c3.metric("เพศชาย : หญิง", f"{male_ratio:.1f}% : {100-male_ratio:.1f}%")
        
        st.markdown("### 🩸 สัดส่วนผู้บาดเจ็บตามระดับความเสี่ยง")
        fig, ax = plt.subplots(figsize=(5, 4))
        severity_counts = df_log['predicted_severity'].value_counts()
        pie_colors = [triage_color.get(label, '#808080') for label in severity_counts.index]
        
        severity_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=pie_colors, ax=ax, textprops={'color': 'white', 'fontsize': 10})
        ax.set_ylabel('')
        ax.set_title("ระดับความรุนแรง", color='white', fontsize=12)
        st.pyplot(fig)
        plt.close(fig) # Prevent figure from being re-rendered
    else:
        st.info("มีไฟล์ข้อมูลแต่ยังไม่มีรายการบันทึก")

# ==========================================================
# 🖥️ Main App Interface
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Clinical Risk Prediction",
    "👥 Cluster Insight",
    "🧩 Risk Association",
    "📊 Clinical Summary Dashboard"
])

with tab1:
    render_tab1()

with tab2:
    render_tab2()

with tab3:
    render_tab3()

with tab4:
    render_tab4()

