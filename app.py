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
    msgs = []  # เก็บข้อความไว้แสดงใน expander

    # 🔹 CatBoost Model
    try:
        model = joblib.load("predict_catboost_multi.pkl")
        msgs.append("✅ predict_catboost_multi.pkl — Clinical Severity Model")
    except:
        model = None
        msgs.append("❌ ไม่พบ predict_catboost_multi.pkl")

    # 🔹 Encoders
    try:
        encoders = joblib.load("encoders_multi.pkl")
        msgs.append("✅ encoders_multi.pkl — Encoders for Clinical Data")
    except:
        encoders = None
        msgs.append("⚠️ ไม่พบ encoders_multi.pkl")

    # 🔹 Features
    try:
        with open("features_multi.json", "r") as f:
            features = json.load(f)
        msgs.append("✅ features_multi.json — Model Features Configuration")
    except:
        features = []
        msgs.append("⚠️ ไม่พบ features_multi.json")

    # 🔹 K-Means
    try:
        kmeans = joblib.load("kmeans_cluster_model.pkl")
        scaler = joblib.load("scaler_cluster.pkl")
        msgs.append("✅ kmeans_cluster_model.pkl / scaler_cluster.pkl — Clustering Models")
    except:
        kmeans = scaler = None
        msgs.append("⚠️ ไม่พบไฟล์ K-Means / Scaler")

    # 🔹 Apriori
    try:
        rules_minor = joblib.load("apriori_rules_minor.pkl")
        rules_severe = joblib.load("apriori_rules_severe.pkl")
        rules_fatal = joblib.load("apriori_rules_fatal.pkl")
        msgs.append("✅ apriori_rules_[minor/severe/fatal].pkl — Risk Pattern Mining Rules")
    except:
        rules_minor = rules_severe = rules_fatal = None
        msgs.append("⚠️ ไม่พบไฟล์กฎ Apriori")

    # ✅ แสดงผลแบบเรียบใน expander (คุณอยากได้ตรงนี้)
    with st.expander("📂 รายการไฟล์ที่โหลดแล้ว", expanded=False):
        for m in msgs:
            st.caption(m)

    return model, encoders, features, kmeans, scaler, rules_minor, rules_severe, rules_fatal


# เรียกใช้
model, encoders, features, kmeans, scaler, rules_minor, rules_severe, rules_fatal = load_all()


# ----------------------------------------------------------
# 🧩 Manual Mappings
# ----------------------------------------------------------
activity_mapping = {
    "0": "เดินเท้า",
    "1": "โดยสารพาหนะสาธารณะ",
    "2": "โดยสารพาหนะส่วนบุคคล",
    "3": "ขับขี่พาหนะส่วนบุคคล",
    "4": "ทำงาน",
    "5": "เล่นกีฬา",
    "6": "กิจกรรมอื่น ๆ"
}

aplace_mapping = {
    "10": "บ้านพักอาศัย",
    "11": "ถนน/ทางหลวง",
    "12": "สถานที่ทำงาน",
    "13": "โรงเรียน/สถาบันศึกษา",
    "14": "พื้นที่สาธารณะ",
    "15": "อื่น ๆ"
}

prov_mapping = {
    "10": "กรุงเทพมหานคร",
    "20": "เชียงใหม่",
    "30": "ขอนแก่น",
    "40": "ภูเก็ต",
    "50": "นครราชสีมา",
    "60": "สงขลา",
    "99": "อื่น ๆ"
}

severity_map = {0: "เสี่ยงน้อย", 1: "เสี่ยงปานกลาง", 2: "เสี่ยงมาก"}
advice_map = {
    "เสี่ยงน้อย": "ดูแลอาการทั่วไป เฝ้าระวังซ้ำทุก 15–30 นาที",
    "เสี่ยงปานกลาง": "ส่งตรวจเพิ่มเติม ให้สารน้ำ / ยาแก้ปวด / เฝ้าสัญญาณชีพใกล้ชิด",
    "เสี่ยงมาก": "แจ้งทีมสหสาขา เปิดทางเดินหายใจ เตรียมห้องฉุกเฉินหรือส่งต่อด่วน"
}

# สี triage
triage_color = {
    "เสี่ยงน้อย": "#4CAF50",      # เขียว
    "เสี่ยงปานกลาง": "#FFC107",  # เหลือง
    "เสี่ยงมาก": "#F44336"        # แดง
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

# ----------------------------------------------------------
# 🧠 TAB 1 — CatBoost Prediction (UNCHANGED LOGIC)
# ----------------------------------------------------------
with tab1:
    st.subheader("🧠 Clinical Severity Prediction")
    st.caption("ระบบประเมินระดับความรุนแรงของผู้บาดเจ็บแบบเรียลไทม์")

    # ORIGINAL INPUT FORM (unchanged)
    st.subheader("📋 ข้อมูลผู้บาดเจ็บ")
    with st.form("input_form"):
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

    def preprocess_input(data_dict):
        df = pd.DataFrame([data_dict])
        reverse_activity = {v: k for k, v in activity_mapping.items()}
        reverse_aplace = {v: k for k, v in aplace_mapping.items()}
        reverse_prov = {v: k for k, v in prov_mapping.items()}
        if df.at[0, "activity"] in reverse_activity:
            df.at[0, "activity"] = reverse_activity[df.at[0, "activity"]]
        if df.at[0, "aplace"] in reverse_aplace:
            df.at[0, "aplace"] = reverse_aplace[df.at[0, "aplace"]]
        if df.at[0, "prov"] in reverse_prov:
            df.at[0, "prov"] = reverse_prov[df.at[0, "prov"]]
        for col in [
            "age", "sex", "is_night", "head_injury", "mass_casualty",
            "risk1", "risk2", "risk3", "risk4", "risk5",
            "cannabis", "amphetamine", "drugs"
        ]:
            df[col] = df[col].astype(float)
        for col in ["activity", "aplace", "prov"]:
            val = str(df.at[0, col])
            if encoders and col in encoders:
                le = encoders[col]
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = 0
            else:
                df[col] = 0
        if "age_group_60plus" not in df.columns:
            df["age_group_60plus"] = (df["age"] >= 60).astype(int)
        if "risk_count" not in df.columns:
            df["risk_count"] = df[["risk1","risk2","risk3","risk4","risk5"]].sum(axis=1)
        if "night_flag" not in df.columns:
            df["night_flag"] = df["is_night"].astype(int)
        df = df.reindex(columns=features, fill_value=0)
        return df

    if submit:
        input_data = {
            "age": age,
            "sex": 1 if sex == "ชาย" else 0,
            "is_night": int(is_night),
            "head_injury": int(head_injury),
            "mass_casualty": int(mass_casualty),
            "risk1": int(risk1),
            "risk2": int(risk2),
            "risk3": int(risk3),
            "risk4": int(risk4),
            "risk5": int(risk5),
            "cannabis": int(cannabis),
            "amphetamine": int(amphetamine),
            "drugs": int(drugs),
            "activity": activity,
            "aplace": aplace,
            "prov": prov
        }

        X_input = preprocess_input(input_data)
        if model is not None:
            probs = model.predict_proba(X_input)[0]
            pred_class = int(np.argmax(probs))
            label = severity_map.get(pred_class, "ไม่ทราบ")
            color = triage_color.get(label, "#2196F3")

            # 🎨 แสดงผลสี triage
            st.markdown(
                f"<div style='background-color:{color};padding:12px;border-radius:10px;'>"
                f"<h3 style='color:white;text-align:center;'>ระดับความเสี่ยงที่คาดการณ์: {label}</h3></div>",
                unsafe_allow_html=True
            )

            st.info(f"💡 แนวทางทางการแพทย์เบื้องต้น: {advice_map[label]}")
            st.caption(f"🧠 ความมั่นใจของระบบ: {probs[pred_class]*100:.1f}%")

            
# ----------------------------------------------------------
# 🗂️ Save prediction log for dashboard summary
# ----------------------------------------------------------
            log_file = "prediction_log.csv"
            new_row = pd.DataFrame([{
                "timestamp": pd.Timestamp.now(),
                "age": age,
                "sex": sex,
                "predicted_severity": label
            }])
            if os.path.exists(log_file):
                new_row.to_csv(log_file, mode="a", index=False, header=False)
            else:
                new_row.to_csv(log_file, index=False)
            st.success("📁 บันทึกผลการประเมินเข้าสู่ระบบ Dashboard แล้ว")


# ----------------------------------------------------------
# 👥 TAB 2 — K-Means Cluster Analysis (Improved)
# ----------------------------------------------------------
with tab2:
    st.subheader("👥 Patient Segmentation")
    st.caption("วิเคราะห์กลุ่มผู้บาดเจ็บ เพื่อใช้ในการจัดสรรทรัพยากรและการป้องกันเชิงรุก")

    # ------------------------------------------------------
    # 🔹 Patient Summary (ข้อมูลจากการกรอก Tab1)
    # ------------------------------------------------------
    if submit:
        st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
        summary_cols = st.columns(3)
        summary_cols[0].metric("อายุ", f"{age} ปี")
        summary_cols[1].metric("เพศ", sex)
        summary_cols[2].metric("ระดับความเสี่ยงที่คาดการณ์", label)

        # แสดงปัจจัยเสี่ยงสั้น ๆ
        risk_summary = []
        if risk1: risk_summary.append("❗ ไม่สวมหมวกนิรภัย")
        if risk2: risk_summary.append("⚡ ขับรถเร็ว")
        if risk3: risk_summary.append("🍺 เมาแล้วขับ")
        if risk4: risk_summary.append("👶 เด็ก/ผู้สูงอายุ")
        if risk5: risk_summary.append("🩸 บาดเจ็บหลายตำแหน่ง")
        if not risk_summary:
            risk_summary = ["- ไม่มีปัจจัยเสี่ยงชัดเจน"]

        st.markdown(f"**ปัจจัยเสี่ยง:** {', '.join(risk_summary)}")

    else:
        st.info("🕐 เมื่อกรอกข้อมูลและประเมินความเสี่ยงในแท็บแรก ผลจะปรากฏที่นี่")

    # ------------------------------------------------------
    # 🔹 โหลดโมเดล K-Means
    # ------------------------------------------------------
    try:
        kmeans = joblib.load("kmeans_cluster_model.pkl")
        scaler = joblib.load("scaler_cluster.pkl")
        
    except:
        st.warning("⚠️ ไม่พบไฟล์โมเดล K-Means หรือ Scaler")
        kmeans, scaler = None, None

    # ------------------------------------------------------
    # 🔹 วิเคราะห์กลุ่มผู้บาดเจ็บ
    # ------------------------------------------------------
    if model and kmeans and scaler and submit:
        if hasattr(scaler, "feature_names_in_"):
            valid_cols = scaler.feature_names_in_
            X_cluster = X_input[[c for c in valid_cols if c in X_input.columns]]
        else:
            X_cluster = X_input.select_dtypes(include=[np.number])

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

        st.caption("💡 ใช้เพื่อวิเคราะห์แนวโน้มและวางแผนทรัพยากร เช่น ทีมฉุกเฉิน, เวรแพทย์, หรือโครงการป้องกันอุบัติเหตุ")


# ----------------------------------------------------------
# 🧩 TAB 3 — Apriori Risk Association (Improved Summary)
# ----------------------------------------------------------
with tab3:
    st.subheader("🧩 Risk Association Analysis")
    st.caption("วิเคราะห์ความสัมพันธ์ของปัจจัยเสี่ยง เพื่อวางแผนป้องกันและสนับสนุนการตัดสินใจเชิงนโยบาย")

    # ✅ ส่วนสรุปข้อมูลผู้บาดเจ็บจากการประเมิน
    if submit:
        st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
        summary_cols = st.columns(3)
        summary_cols[0].metric("อายุ", f"{age} ปี")
        summary_cols[1].metric("เพศ", sex)
        summary_cols[2].metric("ระดับความเสี่ยง", label)

        risk_tags = []
        if risk1: risk_tags.append("ไม่สวมหมวกนิรภัย")
        if risk2: risk_tags.append("ขับรถเร็ว/ประมาท")
        if risk3: risk_tags.append("เมาแล้วขับ")
        if risk4: risk_tags.append("ผู้สูงอายุ/เด็กเล็ก")
        if risk5: risk_tags.append("บาดเจ็บหลายตำแหน่ง")
        if not risk_tags:
            risk_tags = ["ไม่มีปัจจัยเสี่ยงเด่น"]
        st.markdown("**ปัจจัยเสี่ยง:** " + ", ".join(risk_tags))
        st.markdown("---")

    # ✅ โหลดกฎ Apriori
    try:
        rules_minor = joblib.load("apriori_rules_minor.pkl")
        rules_severe = joblib.load("apriori_rules_severe.pkl")
        rules_fatal = joblib.load("apriori_rules_fatal.pkl")
        
    except:
        st.warning("⚠️ ไม่พบไฟล์กฎ Apriori")
        rules_minor = rules_severe = rules_fatal = None

    # ✅ ฟังก์ชันแปลง frozenset → คำอ่านง่าย
    def decode_set(x):
        if isinstance(x, (frozenset, set)):
            replacements = {
                "risk1": "ไม่สวมหมวกนิรภัย/เข็มขัดนิรภัย",
                "risk2": "ขับรถเร็ว/ประมาท",
                "risk3": "เมาแล้วขับ",
                "risk4": "ผู้สูงอายุ/เด็กเล็ก",
                "risk5": "บาดเจ็บหลายตำแหน่ง",
                "head_injury": "บาดเจ็บที่ศีรษะ",
                "mass_casualty": "เหตุการณ์หมู่",
                "cannabis": "พบกัญชาในร่างกาย",
                "amphetamine": "พบแอมเฟตามีนในร่างกาย",
                "drugs": "พบยาอื่น ๆ ในร่างกาย",
                "sex": "เพศชาย",
                "age60plus": "อายุมากกว่า 60 ปี"
            }
            readable = [replacements.get(str(i), str(i)) for i in list(x)]
            return ", ".join(readable)
        return str(x)

    # ✅ เลือกชุดกฎตามผลการทำนาย
    if model and submit:
        if label == "เสี่ยงน้อย" and rules_minor is not None:
            df_rules = rules_minor.head(5)
        elif label == "เสี่ยงปานกลาง" and rules_severe is not None:
            df_rules = rules_severe.head(5)
        elif label == "เสี่ยงมาก" and rules_fatal is not None:
            df_rules = rules_fatal.head(5)
        else:
            df_rules = pd.DataFrame()

        if not df_rules.empty:
            df_rules["antecedents"] = df_rules["antecedents"].apply(decode_set)
            df_rules["consequents"] = df_rules["consequents"].apply(decode_set)

            top_rule = df_rules.iloc[0]
            st.markdown(
                f"""
                <div style='background-color:#1E1E1E;border-radius:10px;padding:12px;margin-bottom:10px'>
                💡 <b>Insight:</b> พบว่าผู้ที่มี <b style='color:#FFC107'>{top_rule['antecedents']}</b> 
                มักจะมีแนวโน้ม <b style='color:#FF5252'>{top_rule['consequents']}</b> 
                (ความมั่นใจ {top_rule['confidence']*100:.1f}%, Lift = {top_rule['lift']:.2f})
                </div>
                """,
                unsafe_allow_html=True
            )

            st.dataframe(
                df_rules[["antecedents", "consequents", "support", "confidence", "lift"]],
                use_container_width=True, hide_index=True
            )

            st.markdown("📘 **การตีความ:**")
            st.markdown("- **Support:** ความถี่ของกฎในข้อมูลทั้งหมด")
            st.markdown("- **Confidence:** ความน่าจะเป็นที่กฎนี้เกิดขึ้นจริงในข้อมูล")
            st.markdown("- **Lift > 1:** แสดงว่าความสัมพันธ์มีนัยสำคัญมากกว่าความบังเอิญ")
        else:
            st.info("📭 ยังไม่มีกฎสำหรับประเภทนี้")



# ----------------------------------------------------------
# 📊 TAB 4 — Clinical Summary & Insights Dashboard (Final Version)
# ----------------------------------------------------------
with tab4:
    st.subheader("📊 Clinical Summary & Insights")
    st.caption("สรุปแนวโน้มผู้บาดเจ็บจากระบบ AI เพื่อใช้วางแผนเชิงกลยุทธ์และบริหารทรัพยากรโรงพยาบาล")

    # ======================================================
    # 🗂️ 1. Load or Reset Log File
    # ======================================================
    log_file = "prediction_log.csv"

    c1, c2 = st.columns([4,1])
    with c1:
        st.markdown("#### 🩺 สรุปข้อมูลจากการประเมินทั้งหมด")
    with c2:
        if st.button("🧹 ล้างข้อมูลทั้งหมด"):
            if os.path.exists(log_file):
                os.remove(log_file)
                st.success("✅ ล้างข้อมูลเรียบร้อยแล้ว (เริ่มเก็บใหม่ได้เลย)")
            else:
                st.info("⚠️ ยังไม่มีข้อมูลให้ลบ")

    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        st.success(f"📁 โหลดข้อมูลสำเร็จ: {len(df_log):,} รายการ")
    else:
        st.warning("⚠️ ยังไม่มีข้อมูลจากการทำนาย")
        df_log = pd.DataFrame(columns=[
            "timestamp", "age", "sex", "predicted_severity",
            "prov", "is_night", "risk1","risk2","risk3","risk4","risk5"
        ])

    total_cases = len(df_log)

    # ======================================================
# 📌 2. KPI Overview (Updated to show gender ratio)
# ======================================================
    st.markdown("### 💡 ภาพรวมสถานการณ์ (KPI Overview)")
    c1, c2, c3 = st.columns(3)

    total_cases = len(df_log)
    severe_ratio = df_log["predicted_severity"].eq("เสี่ยงมาก").mean() * 100 if total_cases > 0 else 0

    # ✅ คำนวณสัดส่วนเพศ
    if total_cases > 0 and "sex" in df_log.columns:
        male_ratio = (df_log["sex"] == "ชาย").mean() * 100
        female_ratio = (df_log["sex"] == "หญิง").mean() * 100
    else:
        male_ratio = female_ratio = 0

    c1.metric("จำนวนเคสทั้งหมด", f"{total_cases:,}")
    c2.metric("สัดส่วนผู้บาดเจ็บรุนแรง", f"{severe_ratio:.1f}%")
    c3.metric("เพศชาย : หญิง", f"{male_ratio:.1f}% : {female_ratio:.1f}%")


    # ======================================================
    # 🩸 3. Distribution by Severity (Pie Chart)
    # ======================================================
    if total_cases > 0:
        st.markdown("### 🩸 สัดส่วนผู้บาดเจ็บตามระดับความเสี่ยง")
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        colors = ['#4CAF50', '#FFC107', '#F44336']
        df_log['predicted_severity'].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=90, colors=colors, ax=ax,
            textprops={'color': 'white', 'fontsize': 10}
        )
        ax.set_ylabel('')
        ax.set_title("ระดับความรุนแรงของผู้บาดเจ็บ", color='white', fontsize=11)
        st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    
    # ======================================================
# 🧭 6. Insight Summary (Textual)
# ======================================================
if total_cases > 0:
    st.markdown("---")
    st.markdown("### 🩺 Insight ทางคลินิก & ข้อเสนอเชิงกลยุทธ์")

    # 🔹 ตัวอย่างการสรุปเชิงอัตโนมัติ
    if not df_log.empty and "predicted_severity" in df_log.columns:
        top_severity = df_log["predicted_severity"].value_counts().idxmax()

        if top_severity == "เสี่ยงมาก":
            msg = "มีแนวโน้มผู้บาดเจ็บรุนแรงสูง ควรจัดสรรทีมฉุกเฉินและทรัพยากรเพิ่มในช่วงเวลาที่พบเคสสูงสุด"
        elif top_severity == "เสี่ยงปานกลาง":
            msg = "กลุ่มผู้บาดเจ็บส่วนใหญ่มีความเสี่ยงปานกลาง ควรเน้นการติดตามอาการและประเมินซ้ำ"
        else:
            msg = "ส่วนใหญ่เป็นกลุ่มความเสี่ยงต่ำ สามารถใช้แนวทางป้องกันและให้ความรู้ประชาชน"

        st.info(f"📊 สรุปสถานการณ์ปัจจุบัน: {msg}")
        st.caption("💡 ใช้ข้อมูลนี้เพื่อสนับสนุนการจัดลำดับความสำคัญและบริหารทรัพยากรโรงพยาบาล")
    else:
        st.warning("⚠️ ยังไม่มีข้อมูลเพียงพอสำหรับการสรุป Insight")

