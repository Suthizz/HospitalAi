import streamlit as st
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าให้ matplotlib แสดงผลได้เหมาะสม
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (6, 4)
# Configure matplotlib to support Thai font
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a font that supports Thai characters
plt.rcParams['axes.unicode_minus'] = False  # Allow minus sign to be displayed correctly


# ----------------------------------------------------------
# 📦 Load Models + Configs
# ----------------------------------------------------------

@st.cache_resource
def load_all():
    st.write("⏳ Attempting to Load Models...")

    # 🔹 CatBoost Model
    try:
        # Assume the model file is available in the deployment environment
        model = joblib.load("predict_catboost_multi.pkl")
        st.write("✅ Clinical Severity Model Loaded.")
    except Exception as e:
        model = None
        st.error(f"❌ predict_catboost_multi.pkl NOT FOUND. (Error: {e}) -> Prediction will not work.")

    # 🔹 Encoders / Features / K-Means / Apriori
    try:
        encoders = joblib.load("encoders_multi.pkl")
    except: encoders = None

    try:
        with open("features_multi.json", "r") as f: features = json.load(f)
    except: features = ['age', 'sex', 'is_night', 'head_injury', 'mass_casualty', 'risk1', 'risk2', 'risk3', 'risk4', 'risk5', 'cannabis', 'amphetamine', 'drugs', 'activity', 'aplace', 'prov']

    try:
        kmeans = joblib.load("kmeans_cluster_model.pkl")
        scaler = joblib.load("scaler_cluster.pkl")
    except: kmeans, scaler = None, None

    try:
        rules_minor = joblib.load("apriori_rules_minor.pkl")
        rules_severe = joblib.load("apriori_rules_severe.pkl")
        rules_fatal = joblib.load("apriori_rules_fatal.pkl")
    except: rules_minor, rules_severe, rules_fatal = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return model, encoders, features, kmeans, scaler, rules_minor, rules_severe, rules_fatal

# เรียกใช้การโหลด
model, encoders, features, kmeans, scaler, rules_minor, rules_severe, rules_fatal = load_all()

# ----------------------------------------------------------
# 🧩 Manual Mappings
# ----------------------------------------------------------
activity_mapping = {
    "0": "เดินเท้า", "1": "โดยสารพาหนะสาธารณะ", "2": "โดยสารพาหนะส่วนบุคคล",
    "3": "ขับขี่พาหนะส่วนบุคคล", "4": "ทำงาน", "5": "เล่นกีฬา", "6": "กิจกรรมอื่น ๆ"
}
aplace_mapping = {
    "10": "บ้านพักอาศัย", "11": "ถนน/ทางหลวง", "12": "สถานที่ทำงาน",
    "13": "โรงเรียน/สถาบันศึกษา", "14": "พื้นที่สาธารณะ", "15": "อื่น ๆ"
}
prov_mapping = {
    "10": "กรุงเทพมหานคร", "20": "เชียงใหม่", "30": "ขอนแก่น",
    "40": "ภูเก็ต", "50": "นครราชสีมา", "60": "สงขลา", "99": "อื่น ๆ"
}
severity_map = {0: "เสี่ยงน้อย", 1: "เสี่ยงปานกลาง", 2: "เสี่ยงมาก"}
advice_map = {
    "เสี่ยงน้อย": "ดูแลอาการทั่วไป เฝ้าระวังซ้ำทุก 15–30 นาที",
    "เสี่ยงปานกลาง": "ส่งตรวจเพิ่มเติม ให้สารน้ำ / ยาแก้ปวด / เฝ้าสัญญาณชีพใกล้ชิด",
    "เสี่ยงมาก": "แจ้งทีมสหสาขา เปิดทางเดินหายใจ เตรียมห้องฉุกเฉินหรือส่งต่อด่วน"
}
triage_color = {
    "เสี่ยงน้อย": "#4CAF50", "เสี่ยงปานกลาง": "#FFC107", "เสี่ยงมาก": "#F44336"
}

# ----------------------------------------------------------
# 🧩 Preprocess Function
# ----------------------------------------------------------
def preprocess_input(data_dict, encoders, features, activity_mapping, aplace_mapping, prov_mapping):
    df = pd.DataFrame([data_dict])
    reverse_activity = {v: k for k, v in activity_mapping.items()}
    reverse_aplace = {v: k for k, v in aplace_mapping.items()}
    reverse_prov = {v: k for k, v in prov_mapping.items()}

    # Mapping
    if df.at[0, "activity"] in reverse_activity:
        df.at[0, "activity"] = reverse_activity[df.at[0, "activity"]]
    if df.at[0, "aplace"] in reverse_aplace:
        df.at[0, "aplace"] = reverse_aplace[df.at[0, "aplace"]]
    if df.at[0, "prov"] in reverse_prov:
        df.at[0, "prov"] = reverse_prov[df.at[0, "prov"]]

    # Type Conversion
    for col in [
        "age", "sex", "is_night", "head_injury", "mass_casualty",
        "risk1", "risk2", "risk3", "risk4", "risk5",
        "cannabis", "amphetamine", "drugs"
    ]:
        df[col] = df[col].astype(float)

    # Encoding (Simplified if encoders are not available)
    for col in ["activity", "aplace", "prov"]:
        val = str(df.at[0, col])
        if encoders and col in encoders:
            le = encoders[col]
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0
        else:
            try:
                df[col] = int(val) # Use original string/int value if no encoder
            except ValueError:
                 df[col] = 0 # Handle cases where conversion to int fails


    # Feature Engineering
    if "age_group_60plus" not in df.columns:
        df["age_group_60plus"] = (df["age"] >= 60).astype(int)
    if "risk_count" not in df.columns:
        df["risk_count"] = df[["risk1","risk2","risk3","risk4","risk5"]].sum(axis=1)
    if "night_flag" not in df.columns:
        df["night_flag"] = df["is_night"].astype(int)

    # Reindex and Fill
    df = df.reindex(columns=features, fill_value=0)
    return df

# ----------------------------------------------------------
# 📄 Streamlit App Layout
# ----------------------------------------------------------
st.set_page_config(layout="wide")

st.title("🏥 Hospital AI Decision Support System")

# ----------------------------------------------------------
# 🧠 TAB 1 — CatBoost Prediction
# ----------------------------------------------------------
st.header("🧠 Clinical Severity Prediction")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("อายุ", 0, 100, 30)
        sex = st.radio("เพศ", ["ชาย", "หญิง"])
        is_night = st.checkbox("เหตุการณ์กลางคืน")
    with col2:
        head_injury = st.checkbox("บาดเจ็บที่ศีรษะ")
        mass_casualty = st.checkbox("เหตุการณ์หมู่ (Mass Casualty)")
        activity = st.selectbox("ลักษณะกิจกรรม", list(activity_mapping.values()))
    with col3:
        aplace = st.selectbox("สถานที่เกิดเหตุ", list(aplace_mapping.values()))
        prov = st.selectbox("จังหวัด", list(prov_mapping.values()))
        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            risk1 = st.checkbox("Risk 1: ไม่สวมหมวกนิรภัย / เข็มขัดนิรภัย")
            risk2 = st.checkbox("Risk 2: ขับรถเร็ว / ประมาท")
            risk3 = st.checkbox("Risk 3: เมา / ดื่มสุรา")
        with risk_col2:
            risk4 = st.checkbox("Risk 4: ผู้สูงอายุ / เด็กเล็ก")
            risk5 = st.checkbox("Risk 5: บาดเจ็บหลายตำแหน่ง")
            # Drugs (Assume separate inputs for simplicity in UI)
            cannabis = st.checkbox("พบกัญชา")
            amphetamine = st.checkbox("พบแอมเฟตามีน")
            drugs = st.checkbox("พบยาเสพติดอื่น ๆ")


    submit_button = st.form_submit_button("ประเมินความเสี่ยง")

if submit_button:
    # 1. จัดรูปแบบ Input Data
    input_data = {
        "age": age,
        "sex": 1 if sex == "ชาย" else 0, # Male=1, Female=0
        "is_night": int(is_night),
        "head_injury": int(head_injury),
        "mass_casualty": int(mass_casualty),
        "risk1": int(risk1), "risk2": int(risk2), "risk3": int(risk3),
        "risk4": int(risk4), "risk5": int(risk5),
        "cannabis": int(cannabis), "amphetamine": int(amphetamine), "drugs": int(drugs),
        "activity": activity, "aplace": aplace, "prov": prov
    }

    # 2. Preprocess
    X_input = preprocess_input(input_data, encoders, features, activity_mapping, aplace_mapping, prov_mapping)

    # 3. Predict
    current_label = "ไม่ทราบ"
    if model is not None:
        try:
            probs = model.predict_proba(X_input)[0]
            pred_class = int(np.argmax(probs))
            current_label = severity_map.get(pred_class, "ไม่ทราบ")
            color = triage_color.get(current_label, "#2196F3")

            # 4. แสดงผล
            st.subheader("🔥 ผลการประเมินความเสี่ยง")
            st.markdown(f"<h3 style='color:{color}'>{current_label}</h3>", unsafe_allow_html=True)
            st.info(f"💡 แนวทางทางการแพทย์เบื้องต้น: {advice_map.get(current_label, 'ไม่ทราบแนวทาง')}")
            st.caption(f"🧠 ความมั่นใจของระบบ: {probs[pred_class]*100:.1f}% (Probabilities: {probs})")

            # 5. บันทึก Prediction Log (For Dashboard)
            log_file = "prediction_log.csv"
            new_row = pd.DataFrame([{
                "timestamp": pd.Timestamp.now(),
                "age": age,
                "sex": sex,
                "predicted_severity": current_label
            }])
            if os.path.exists(log_file):
                new_row.to_csv(log_file, mode="a", index=False, header=False)
            else:
                new_row.to_csv(log_file, index=False)
            st.write("📁 บันทึกผลการประเมินเข้าสู่ระบบ Log แล้ว")

        except Exception as e:
            st.error(f"❌ Error ในการทำนาย: {e}. (อาจเป็นเพราะไฟล์โมเดล CatBoost ไม่สมบูรณ์)")

    else:
        st.warning("⚠️ โมเดลหลัก (CatBoost) ไม่พร้อมใช้งาน ไม่สามารถทำนายได้")

    # ----------------------------------------------------------
    # 👥 TAB 2 — K-Means Cluster Analysis
    # ----------------------------------------------------------
    st.header("👥 Patient Segmentation")

    if kmeans is not None and scaler is not None and model is not None:
        st.write(f"🧾 ข้อมูล: อายุ {age} ปี, เพศ {sex}, ระดับความเสี่ยง: {current_label}")

        # 1. เตรียมข้อมูลสำหรับ Clustering
        if hasattr(scaler, "feature_names_in_"):
            valid_cols = scaler.feature_names_in_
            # พยายามเลือกเฉพาะ features ที่ scaler ต้องการ
            X_cluster = X_input[[c for c in valid_cols if c in X_input.columns]]
        else:
            # หากไม่มี feature names ให้เลือกทุกคอลัมน์ที่เป็นตัวเลข
             X_cluster = X_input.select_dtypes(include=[np.number])


        if not X_cluster.empty:
            # 2. Scaling และ Predict Cluster
            try:
                X_scaled = scaler.transform(X_cluster)
                cluster_label = int(kmeans.predict(X_scaled)[0])

                # 3. แสดงผล
                cluster_desc = {
                    0: "👵 กลุ่มผู้สูงอายุ / ลื่นล้มในบ้าน → ความเสี่ยงต่ำ",
                    1: "🚗 กลุ่มวัยทำงาน / เมา / ขับรถเร็ว → ความเสี่ยงสูง",
                    2: "⚽ กลุ่มเด็กและวัยรุ่น / เล่นกีฬา → ความเสี่ยงปานกลาง",
                    3: "👷 กลุ่มแรงงาน / ก่อสร้าง → ความเสี่ยงสูง",
                    4: "🙂 กลุ่มทั่วไป / ไม่มีปัจจัยเด่น → ความเสี่ยงต่ำ"
                }

                st.subheader(f"📊 ผลการจัดกลุ่มผู้บาดเจ็บ: Cluster {cluster_label}")
                st.info(f"💡 คำอธิบายกลุ่ม: {cluster_desc.get(cluster_label, 'ยังไม่มีคำอธิบายกลุ่มนี้')}")
            except Exception as e:
                st.error(f"❌ Error in Clustering: {e}")
        else:
            st.warning("⚠️ ข้อมูลสำหรับ Clustering ไม่เพียงพอ หรือรูปแบบไม่ถูกต้อง")
    else:
        st.warning("⚠️ โมเดล K-Means / Scaler ไม่พร้อมใช้งาน หรือยังไม่ได้ทำนาย")


    # ----------------------------------------------------------
    # 🧩 TAB 3 — Apriori Risk Association
    # ----------------------------------------------------------
    st.header("🧩 Risk Association Analysis")

    # 1. ฟังก์ชันแปลง frozenset → คำอ่านง่าย
    def decode_set(x):
        if isinstance(x, (frozenset, set)):
            replacements = {
                "risk1": "ไม่สวมหมวกนิรภัย/เข็มขัดนิรภัย", "risk2": "ขับรถเร็ว/ประมาท", "risk3": "เมาแล้วขับ",
                "risk4": "ผู้สูงอายุ/เด็กเล็ก", "risk5": "บาดเจ็บหลายตำแหน่ง", "head_injury": "บาดเจ็บที่ศีรษะ",
                "mass_casualty": "เหตุการณ์หมู่", "cannabis": "พบกัญชาในร่างกาย", "amphetamine": "พบแอมเฟตามีนในร่างกาย",
                "drugs": "พบยาอื่น ๆ ในร่างกาย", "sex": "เพศชาย", "age60plus": "อายุมากกว่า 60 ปี"
            }
            readable = [replacements.get(str(i), str(i)) for i in list(x)]
            return ", ".join(readable)
        return str(x)

    # 2. เลือกชุดกฎตามผลการทำนาย
    df_rules = pd.DataFrame()

    if current_label == "เสี่ยงน้อย":
        df_rules = rules_minor.copy()
    elif current_label == "เสี่ยงปานกลาง":
        df_rules = rules_severe.copy()
    elif current_label == "เสี่ยงมาก":
        df_rules = rules_fatal.copy()

    if not df_rules.empty:
        df_rules = df_rules.head(5) # เลือก 5 อันดับแรก
        df_rules["antecedents"] = df_rules["antecedents"].apply(decode_set)
        df_rules["consequents"] = df_rules["consequents"].apply(decode_set)

        # 3. แสดงผล Insight หลัก
        if not df_rules.empty:
            top_rule = df_rules.iloc[0]
            st.subheader("💡 Insight (อิงจากผลการทำนาย)")
            st.write(f"ผู้ที่มี **{top_rule['antecedents']}** มักมีแนวโน้ม **{top_rule['consequents']}**")
            st.caption(f"(Confidence: {top_rule['confidence']*100:.1f}%, Lift: {top_rule['lift']:.2f})")

        # 4. แสดงตารางกฎ
        st.subheader("📚 ตารางกฎ Risk Association Top 5:")
        st.dataframe(df_rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    else:
        st.info(f"📭 ยังไม่มีกฎ Apriori สำหรับความเสี่ยงระดับ '{current_label}'")


    # ----------------------------------------------------------
    # 📊 TAB 4 — Clinical Summary & Insights Dashboard
    # ----------------------------------------------------------
    st.header("📊 Clinical Summary & Insights Dashboard")

    log_file = "prediction_log.csv"

    # 1. โหลดข้อมูล Log
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        st.write(f"📁 โหลดข้อมูลสำเร็จ: {len(df_log):,} รายการ\n")
    else:
        st.warning("⚠️ ยังไม่มีข้อมูลจากการทำนาย (prediction_log.csv) ให้ทำการประเมินความเสี่ยงก่อน")
        df_log = pd.DataFrame(columns=["timestamp", "age", "sex", "predicted_severity"])

    total_cases = len(df_log)

    if total_cases > 0:
        # 2. KPI Overview
        st.subheader("💡 ภาพรวมสถานการณ์ (KPI Overview)")
        severe_ratio = df_log["predicted_severity"].eq("เสี่ยงมาก").mean() * 100
        male_ratio = (df_log["sex"] == "ชาย").mean() * 100
        female_ratio = (df_log["sex"] == "หญิง").mean() * 100

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("จำนวนเคสทั้งหมด", f"{total_cases:,}")
        col_kpi2.metric("สัดส่วนผู้บาดเจ็บรุนแรง", f"{severe_ratio:.1f}%")
        col_kpi3.metric("สัดส่วน เพศชาย : หญิง", f"{male_ratio:.1f}% : {female_ratio:.1f}%")


        # 3. Distribution by Severity (Pie Chart)
        st.subheader("🩸 สัดส่วนผู้บาดเจ็บตามระดับความเสี่ยง (Pie Chart)")

        # Use matplotlib/seaborn to create the plot
        fig, ax = plt.subplots()
        # Need to ensure all 3 severity levels are present for consistent colors
        severity_counts = df_log['predicted_severity'].value_counts().reindex(severity_map.values(), fill_value=0)

        ax.pie(
            severity_counts,
            labels=severity_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=[triage_color.get(l) for l in severity_counts.index],
            textprops={'color': 'black', 'fontsize': 10}
        )
        ax.set_title("ระดับความรุนแรงของผู้บาดเจ็บ")
        st.pyplot(fig)


        # 4. Insight Summary
        st.subheader("🩺 Insight ทางคลินิก & ข้อเสนอเชิงกลยุทธ์")
        top_severity = df_log["predicted_severity"].value_counts().idxmax()

        if top_severity == "เสี่ยงมาก":
            msg = "มีแนวโน้มผู้บาดเจ็บรุนแรงสูง ควรจัดสรรทีมฉุกเฉินและทรัพยากรเพิ่มในช่วงเวลาที่พบเคสสูงสุด"
        elif top_severity == "เสี่ยงปานกลาง":
            msg = "กลุ่มผู้บาดเจ็บส่วนใหญ่มีความเสี่ยงปานกลาง ควรเน้นการติดตามอาการและประเมินซ้ำ"
        else:
            msg = "ส่วนใหญ่เป็นกลุ่มความเสี่ยงต่ำ สามารถใช้แนวทางป้องกันและให้ความรู้ประชาชน"

        st.write(f"📊 **สรุปสถานการณ์ปัจจุบัน:** {msg}")
        st.write("💡 ใช้ข้อมูลนี้เพื่อสนับสนุนการจัดลำดับความสำคัญและบริหารทรัพยากรโรงพยาบาล")

    else:
        st.info("📭 ไม่มีข้อมูลเพียงพอสำหรับการแสดง Dashboard")
