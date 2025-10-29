# -*- coding: utf-8 -*-
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier # Needed even when loading model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # Needed for preprocessing

# ----------------------------------------------------------
# ⚙️ Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Road Accident AI Decision Support", page_icon="🏥", layout="wide")
st.title("🏥 Road Accident AI for Clinical Decision Support")
st.caption("ระบบสนับสนุนการตัดสินใจทางการแพทย์และการบริหารทรัพยากรโรงพยาบาล")

# --- Define File Paths ---
MODEL_PATH = "predict_catboost_multi.pkl"
ENCODERS_PATH = "encoders_multi.pkl"
FEATURES_PATH = "features_multi.json"
KMEANS_PATH = "kmeans_cluster_model.pkl"
SCALER_PATH = "scaler_cluster.pkl"
RULES_MINOR_PATH = "apriori_rules_minor.pkl"
RULES_SEVERE_PATH = "apriori_rules_severe.pkl"
RULES_FATAL_PATH = "apriori_rules_fatal.pkl"
LOG_FILE = "prediction_log.csv"

# ----------------------------------------------------------
# 📦 Load Models + Show in Expander (Improved Error Handling)
# ----------------------------------------------------------
@st.cache_resource
def load_all():
    """Loads all necessary model files and returns them."""
    loaded_items = {}
    msgs = []

    files_to_load = {
        "model": (MODEL_PATH, "Clinical Severity Model"),
        "encoders": (ENCODERS_PATH, "Encoders for Clinical Data"),
        "features": (FEATURES_PATH, "Model Features Configuration"),
        "kmeans": (KMEANS_PATH, "KMeans Clustering Model"),
        "scaler": (SCALER_PATH, "Scaler for Clustering"),
        "rules_minor": (RULES_MINOR_PATH, "Apriori Rules (Minor)"),
        "rules_severe": (RULES_SEVERE_PATH, "Apriori Rules (Severe)"),
        "rules_fatal": (RULES_FATAL_PATH, "Apriori Rules (Fatal)"),
    }

    for key, (path, description) in files_to_load.items():
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    loaded_items[key] = json.load(f)
            else: # .pkl files
                loaded_items[key] = joblib.load(path)
            msgs.append(f"✅ {os.path.basename(path)} — {description}")
        except FileNotFoundError:
            loaded_items[key] = None # Or [] for features
            if key == 'features': loaded_items[key] = []
            if key == 'encoders': loaded_items[key] = {}
            msgs.append(f"❌ File not found: {os.path.basename(path)} — {description}")
        except Exception as e:
            loaded_items[key] = None # Or [] for features
            if key == 'features': loaded_items[key] = []
            if key == 'encoders': loaded_items[key] = {}
            msgs.append(f"⚠️ Error loading {os.path.basename(path)}: {e}")

    # Display loading status
    with st.expander("📂 File Loading Status", expanded=False):
        all_loaded = all(item is not None and item != [] and item != {} for key, item in loaded_items.items() if key != 'encoders') # Encoders can be {} initially
        if all_loaded:
            st.success("All required files loaded successfully.")
        else:
            st.warning("Some files could not be loaded. Please check file paths and availability.")

        for m in msgs:
            if "✅" in m:
                st.caption(m)
            else:
                st.caption(f":warning: {m}") # Highlight errors/warnings

    return (
        loaded_items.get("model"),
        loaded_items.get("encoders"),
        loaded_items.get("features"),
        loaded_items.get("kmeans"),
        loaded_items.get("scaler"),
        loaded_items.get("rules_minor"),
        loaded_items.get("rules_severe"),
        loaded_items.get("rules_fatal"),
    )

# ----------------------------------------------------------
# ✅ Load resources using the function
# ----------------------------------------------------------
model, encoders, features, kmeans, scaler, rules_minor, rules_severe, rules_fatal = load_all()

# --- Check if essential components loaded ---
essentials_loaded = model and features and encoders is not None # Check essential components

# ----------------------------------------------------------
# 🧩 Manual Mappings & Helper Functions
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
    "10": "กรุงเทพมหานคร", "20": "เชียงใหม่", "30": "ขอนแก่น", "40": "ภูเก็ต",
    "50": "นครราชสีมา", "60": "สงขลา", "99": "อื่น ๆ"
}
severity_map = {0: "เสี่ยงน้อย", 1: "เสี่ยงปานกลาง", 2: "เสี่ยงมาก"}
advice_map = {
    "เสี่ยงน้อย": "ดูแลอาการทั่วไป เฝ้าระวังซ้ำทุก 15–30 นาที",
    "เสี่ยงปานกลาง": "ส่งตรวจเพิ่มเติม ให้สารน้ำ / ยาแก้ปวด / เฝ้าสัญญาณชีพใกล้ชิด",
    "เสี่ยงมาก": "แจ้งทีมสหสาขา เปิดทางเดินหายใจ เตรียมห้องฉุกเฉินหรือส่งต่อด่วน"
}
triage_color = {"เสี่ยงน้อย": "#4CAF50", "เสี่ยงปานกลาง": "#FFC107", "เสี่ยงมาก": "#F44336"}

# --- Preprocessing Function (Modified to use loaded features/encoders) ---
def preprocess_input(data_dict, loaded_encoders, expected_features):
    """Preprocesses user input dictionary based on loaded encoders and features."""
    if not expected_features:
        st.error("Feature list is empty. Cannot preprocess.")
        return None

    df = pd.DataFrame([data_dict])

    # Map display values back to codes before encoding
    reverse_activity = {v: k for k, v in activity_mapping.items()}
    reverse_aplace = {v: k for k, v in aplace_mapping.items()}
    reverse_prov = {v: k for k, v in prov_mapping.items()}

    df['activity'] = df['activity'].map(reverse_activity).fillna("Unknown") # Map or default
    df['aplace'] = df['aplace'].map(reverse_aplace).fillna("Unknown")
    df['prov'] = df['prov'].map(reverse_prov).fillna("Unknown")

    # Reindex first to ensure all expected columns are present
    df = df.reindex(columns=expected_features, fill_value=0)

    # Clean potential string issues and fill NaNs *after* reindexing
    clean_values = ["Unknown", "None", "N/A", "ไม่ระบุ", "N", "nan", "NaN", "", " "]
    df = df.replace(clean_values, np.nan).fillna(0)

    # Encode categorical features using loaded encoders
    if loaded_encoders:
      for col, le in loaded_encoders.items():
          if col in df.columns:
              # Convert column to string before applying transform
              df[col] = df[col].astype(str)
              # Handle unseen labels by mapping them to a default index (e.g., 0)
              df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    else:
        # If encoders didn't load, attempt basic numeric conversion for non-numeric looking columns
        for col in df.columns:
             if df[col].dtype == 'object':
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    # --- Create Engineered Features (must match training - check feature list) ---
    # Ensure necessary base columns exist before creating engineered ones
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
        if "age_group_60plus" in expected_features:
             df["age_group_60plus"] = (df["age"] >= 60).astype(int)

    risk_cols_in_features = [f"risk{i}" for i in range(1, 6) if f"risk{i}" in expected_features]
    if "risk_count" in expected_features and risk_cols_in_features:
        df["risk_count"] = df[risk_cols_in_features].sum(axis=1)
    elif "risk_count" in expected_features: # If risk_count expected but base risks aren't
         df["risk_count"] = 0


    if 'is_night' in df.columns:
        df['is_night'] = pd.to_numeric(df['is_night'], errors='coerce').fillna(0)
        if "night_flag" in expected_features:
            df["night_flag"] = df["is_night"].astype(int)
    elif "night_flag" in expected_features: # If night_flag expected but is_night isn't
         df["night_flag"] = 0


    # Ensure all columns are numeric and fill any remaining NaNs
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    # Final check: Ensure column order matches exactly and return
    try:
        df = df[expected_features]
    except KeyError as e:
        st.error(f"Mismatch in expected features during preprocessing. Missing: {e}")
        return None

    return df


# --- Function to decode Apriori rules ---
def decode_set(item_set):
    """Converts a frozenset of coded items to readable strings."""
    if not isinstance(item_set, (frozenset, set)):
        return str(item_set)

    # Define mappings (expand as needed based on your Apriori basket items)
    replacements = {
        "Male": "เพศชาย", "Female": "เพศหญิง", "Age_60+": "อายุ 60+",
        "Nighttime": "กลางคืน", "Head_Injury": "บาดเจ็บศีรษะ",
        "Risk_Alcohol": "ดื่มแอลกอฮอล์", "Risk_Drugs_General": "ใช้ยาเสพติด(ทั่วไป)",
        "Risk_No_Belt": "ไม่คาดเข็มขัด", "Risk_No_Helmet": "ไม่สวมหมวก",
        "Risk_Phone_Use": "ใช้โทรศัพท์ขณะขับ",
        "Drug_Kratom": "กระท่อม", "Drug_Cannabis": "กัญชา", "Drug_Amphetamine": "ยาบ้า",
        "Severity_Minor": "บาดเจ็บเล็กน้อย", "Severity_Severe": "บาดเจ็บรุนแรง",
        "Severity_Fatal": "เสียชีวิต"
        # Add more mappings if needed
    }
    readable = [replacements.get(str(item), str(item)) for item in sorted(list(item_set))]
    return ", ".join(readable)


# ==========================================================
# 🩺 TAB SYSTEM
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Clinical Risk Prediction",
    "👥 Cluster Insight",
    "🧩 Risk Association",
    "📊 Clinical Summary Dashboard"
])

# Initialize session state for submit button if it doesn't exist
if 'submit_pressed' not in st.session_state:
    st.session_state.submit_pressed = False
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = "ยังไม่ได้ประเมิน"
if 'processed_input_for_tabs' not in st.session_state:
    st.session_state.processed_input_for_tabs = None
if 'raw_input_for_tabs' not in st.session_state:
    st.session_state.raw_input_for_tabs = {}


# ----------------------------------------------------------
# 🧠 TAB 1 — CatBoost Prediction
# ----------------------------------------------------------
with tab1:
    st.subheader("🧠 Clinical Severity Prediction")
    st.caption("ระบบประเมินระดับความรุนแรงของผู้บาดเจ็บแบบเรียลไทม์")

    if not essentials_loaded:
        st.error("Cannot proceed with prediction. Essential model files (model, features, encoders) are missing.")
    else:
        st.subheader("📋 ข้อมูลผู้บาดเจ็บ")
        with st.form("input_form"):
            # --- Input fields ---
            age = st.number_input("อายุ (ปี)", min_value=0, max_value=120, value=35, key="tab1_age")
            sex = st.radio("เพศ", ["หญิง", "ชาย"], horizontal=True, key="tab1_sex") # Display text
            is_night = st.checkbox("เกิดเหตุเวลากลางคืน (18:00-06:00)", value=False, key="tab1_is_night")
            head_injury = st.checkbox("บาดเจ็บที่ศีรษะ", value=False, key="tab1_head_injury")
            mass_casualty = st.checkbox("เหตุการณ์ผู้บาดเจ็บจำนวนมาก", value=False, key="tab1_mass_casualty")

            st.markdown("**ปัจจัยเสี่ยง (Risk Factors - เลือกหากเกี่ยวข้อง)**")
            c1, c2, c3, c4, c5 = st.columns(5)
            # Use keys consistent with training data if possible, map later if needed
            with c1: risk1 = st.checkbox("ดื่มแอลกอฮอล์ (risk1)", key="tab1_risk1")
            with c2: risk2 = st.checkbox("ใช้ยาเสพติด(ทั่วไป) (risk2)", key="tab1_risk2")
            with c3: risk3 = st.checkbox("ไม่คาดเข็มขัด (risk3)", key="tab1_risk3")
            with c4: risk4 = st.checkbox("ไม่สวมหมวกนิรภัย (risk4)", key="tab1_risk4")
            with c5: risk5 = st.checkbox("ใช้โทรศัพท์ขณะขับ (risk5)", key="tab1_risk5")

            st.markdown("**สารเสพติด/ยาในร่างกาย (เลือกหากเกี่ยวข้อง)**")
            d1, d2, d3 = st.columns(3)
            with d1: cannabis = st.checkbox("กัญชา", key="tab1_cannabis")
            with d2: amphetamine = st.checkbox("ยาบ้า / แอมเฟตามีน", key="tab1_amphetamine")
            with d3: drugs = st.checkbox("ยาอื่น ๆ (drugs)", key="tab1_drugs") # 'drugs' from notebook

            st.markdown("**บริบทของเหตุการณ์**")
            # Using text input for flexibility, assuming encoding handles variations/unknowns
            activity_display = st.selectbox("กิจกรรมขณะเกิดเหตุ", list(activity_mapping.values()), key="tab1_activity")
            aplace_display = st.selectbox("สถานที่เกิดเหตุ", list(aplace_mapping.values()), key="tab1_aplace")
            prov_display = st.selectbox("จังหวัดที่เกิดเหตุ", list(prov_mapping.values()), key="tab1_prov")

            submit_button_tab1 = st.form_submit_button("🔎 ประเมินระดับความเสี่ยง")

        if submit_button_tab1:
            st.session_state.submit_pressed = True # Mark that form was submitted

            # --- Prepare input dictionary ---
            input_data = {
                "age": age,
                "sex": '1' if sex == "ชาย" else '2', # Map back to '1'/'2' code used in notebook
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
                "activity": activity_display, # Keep display value for now, map in preprocess
                "aplace": aplace_display,
                "prov": prov_display
            }
             # Store raw input for other tabs
            st.session_state.raw_input_for_tabs = input_data.copy()


            # --- Preprocess and Predict ---
            X_input = preprocess_input(input_data, encoders, features)
            st.session_state.processed_input_for_tabs = X_input # Store processed input


            if X_input is not None:
                try:
                    probs = model.predict_proba(X_input)[0]
                    pred_class = int(np.argmax(probs))
                    st.session_state.prediction_label = severity_map.get(pred_class, "ไม่ทราบ") # Store label
                    color = triage_color.get(st.session_state.prediction_label, "#2196F3") # Default blue

                    # Display Triage Color Box
                    st.markdown(
                        f"<div style='background-color:{color};padding:12px;border-radius:10px;margin-top: 15px;'>"
                        f"<h3 style='color:white;text-align:center;'>ระดับความเสี่ยงที่คาดการณ์: {st.session_state.prediction_label}</h3></div>",
                        unsafe_allow_html=True
                    )
                    # Display Advice and Confidence
                    st.info(f"💡 แนวทางทางการแพทย์เบื้องต้น: {advice_map.get(st.session_state.prediction_label, 'N/A')}")
                    st.caption(f"🧠 ความมั่นใจของระบบ: {probs[pred_class]*100:.1f}%")


                    # --- Save prediction log ---
                    try:
                        log_entry = {
                            "timestamp": pd.Timestamp.now(tz='Asia/Bangkok'), # Add timezone
                            "age": age,
                            "sex": sex,
                            "predicted_severity": st.session_state.prediction_label,
                            "prov": prov_display,
                             "is_night": int(is_night),
                             "risk1": int(risk1),
                             "risk2": int(risk2),
                             "risk3": int(risk3),
                             "risk4": int(risk4),
                             "risk5": int(risk5),
                             "head_injury": int(head_injury) # Add head injury to log
                        }
                        new_row = pd.DataFrame([log_entry])

                        if os.path.exists(LOG_FILE):
                            new_row.to_csv(LOG_FILE, mode="a", index=False, header=False, encoding='utf-8-sig')
                        else:
                            new_row.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
                        st.success("📁 บันทึกผลการประเมินเข้าสู่ระบบ Dashboard แล้ว")
                    except Exception as log_e:
                        st.warning(f"Could not save prediction log: {log_e}")

                except Exception as pred_e:
                    st.error(f"An error occurred during prediction: {pred_e}")
                    st.error("Please check the input data and model files.")
                    # st.dataframe(X_input) # Optional: show processed data for debugging
            else:
                 st.error("Input preprocessing failed. Cannot make prediction.")


# ----------------------------------------------------------
# 👥 TAB 2 — K-Means Cluster Analysis
# ----------------------------------------------------------
with tab2:
    st.subheader("👥 Patient Segmentation (K-Means)")
    st.caption("วิเคราะห์กลุ่มผู้บาดเจ็บ เพื่อใช้ในการจัดสรรทรัพยากรและการป้องกันเชิงรุก")

    if not st.session_state.submit_pressed:
        st.info("กรุณากรอกข้อมูลและกด 'ประเมินระดับความเสี่ยง' ในแท็บแรกก่อน")
    elif kmeans is None or scaler is None:
        st.warning("⚠️ ไม่พบไฟล์โมเดล K-Means หรือ Scaler ที่โหลดไว้")
    elif st.session_state.processed_input_for_tabs is None:
         st.warning("⚠️ ไม่สามารถประมวลผลข้อมูล Input สำหรับ Clustering ได้")
    else:
        # --- Display Patient Summary ---
        st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
        raw_input = st.session_state.raw_input_for_tabs
        summary_cols = st.columns(3)
        summary_cols[0].metric("อายุ", f"{raw_input.get('age', 'N/A')} ปี")
        summary_cols[1].metric("เพศ", 'ชาย' if raw_input.get('sex') == '1' else 'หญิง' if raw_input.get('sex') == '2' else 'N/A')
        summary_cols[2].metric("ระดับความเสี่ยง", st.session_state.prediction_label)

        risk_summary = [
            "ดื่มแอลกอฮอล์" if raw_input.get('risk1') else None,
            "ใช้ยาเสพติด" if raw_input.get('risk2') else None,
            "ไม่คาดเข็มขัด" if raw_input.get('risk3') else None,
            "ไม่สวมหมวก" if raw_input.get('risk4') else None,
            "ใช้โทรศัพท์" if raw_input.get('risk5') else None,
            "บาดเจ็บศีรษะ" if raw_input.get('head_injury') else None,
        ]
        risk_summary = [r for r in risk_summary if r]
        st.markdown(f"**ปัจจัยเด่น:** {', '.join(risk_summary) if risk_summary else '-'}")
        st.markdown("---")

        # --- Perform Clustering ---
        try:
            # Select columns used by the scaler during training
            if hasattr(scaler, "feature_names_in_"):
                cluster_features_used = scaler.feature_names_in_
                # Ensure the processed input has these columns
                X_input_cluster = st.session_state.processed_input_for_tabs.copy()
                # Add missing columns expected by scaler, fill with 0
                for col in cluster_features_used:
                    if col not in X_input_cluster.columns:
                        X_input_cluster[col] = 0
                X_input_cluster = X_input_cluster[cluster_features_used] # Select and order

            else:
                # Fallback if scaler doesn't have feature names (less reliable)
                st.warning("Scaler object does not contain feature names. Using numeric columns.")
                X_input_cluster = st.session_state.processed_input_for_tabs.select_dtypes(include=np.number)


            if not X_input_cluster.empty:
                X_scaled = scaler.transform(X_input_cluster)
                cluster_label = int(kmeans.predict(X_scaled)[0])

                # Example Cluster Descriptions (Adjust based on your actual cluster analysis)
                cluster_desc = {
                    0: "กลุ่มทั่วไป/ความเสี่ยงต่ำ: มักเป็นเพศชาย ไม่พบบาดเจ็บศีรษะหรือปัจจัยเสี่ยงเด่นชัด",
                    1: "กลุ่มความเสี่ยงปานกลาง/หญิง: มักเป็นเพศหญิง อาจไม่พบบาดเจ็บศีรษะ",
                    2: "กลุ่มความเสี่ยงสูง/ยาเสพติด: พบการใช้ยาเสพติด อาจมีบาดเจ็บศีรษะร่วม"
                    # Add descriptions for all clusters found during analysis
                }

                st.markdown(f"### 📊 ผลการจัดกลุ่ม: **Cluster {cluster_label}**")
                st.info(f"**ลักษณะกลุ่ม:** {cluster_desc.get(cluster_label, 'ยังไม่มีคำอธิบายสำหรับกลุ่มนี้')}")
                st.caption("💡 ใช้ข้อมูลนี้เพื่อทำความเข้าใจลักษณะผู้ป่วยที่คล้ายกัน และวางแผนทรัพยากรหรือแคมเปญป้องกันที่ตรงกลุ่มเป้าหมาย")

            else:
                 st.error("Could not extract valid features for clustering from the input.")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการจัดกลุ่ม K-Means: {e}")
            # st.dataframe(X_input_cluster) # Optional debugging


# ----------------------------------------------------------
# 🧩 TAB 3 — Apriori Risk Association
# ----------------------------------------------------------
with tab3:
    st.subheader("🧩 Risk Association Analysis (Apriori)")
    st.caption("วิเคราะห์ความสัมพันธ์ของปัจจัยเสี่ยง เพื่อวางแผนป้องกันและสนับสนุนการตัดสินใจเชิงนโยบาย")

    if not st.session_state.submit_pressed:
        st.info("กรุณากรอกข้อมูลและกด 'ประเมินระดับความเสี่ยง' ในแท็บแรกก่อน")
    elif rules_minor is None and rules_severe is None and rules_fatal is None:
        st.warning("⚠️ ไม่พบไฟล์กฎ Apriori ที่โหลดไว้")
    else:
        # --- Display Patient Summary ---
        st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
        raw_input = st.session_state.raw_input_for_tabs
        summary_cols = st.columns(3)
        summary_cols[0].metric("อายุ", f"{raw_input.get('age', 'N/A')} ปี")
        summary_cols[1].metric("เพศ", 'ชาย' if raw_input.get('sex') == '1' else 'หญิง' if raw_input.get('sex') == '2' else 'N/A')
        summary_cols[2].metric("ระดับความเสี่ยง", st.session_state.prediction_label)

        risk_tags = [
            "ดื่มแอลกอฮอล์" if raw_input.get('risk1') else None,
            "ใช้ยาเสพติด" if raw_input.get('risk2') else None,
            "ไม่คาดเข็มขัด" if raw_input.get('risk3') else None,
            "ไม่สวมหมวก" if raw_input.get('risk4') else None,
            "ใช้โทรศัพท์" if raw_input.get('risk5') else None,
            "บาดเจ็บศีรษะ" if raw_input.get('head_injury') else None,
        ]
        risk_tags = [r for r in risk_tags if r]
        st.markdown(f"**ปัจจัยเด่น:** {', '.join(risk_tags) if risk_tags else '-'}")
        st.markdown("---")

        # --- Select and Display Relevant Rules ---
        st.markdown("### 🔗 กฎความสัมพันธ์ที่เกี่ยวข้อง (Top 5 by Lift)")

        target_label = st.session_state.prediction_label
        rules_df_to_show = None
        rules_title = f"กฎที่นำไปสู่: {target_label}"

        if target_label == "เสี่ยงน้อย" and rules_minor is not None and not rules_minor.empty:
            rules_df_to_show = rules_minor
        elif target_label == "เสี่ยงปานกลาง" and rules_severe is not None and not rules_severe.empty:
            rules_df_to_show = rules_severe
        elif target_label == "เสี่ยงมาก" and rules_fatal is not None and not rules_fatal.empty:
            rules_df_to_show = rules_fatal

        if rules_df_to_show is not None:
            st.markdown(f"**{rules_title}**")
            # Decode antecedents and consequents for display
            display_df = rules_df_to_show.head(5).copy()
            if 'antecedents' in display_df.columns:
                 display_df["ปัจจัยนำ (Antecedents)"] = display_df["antecedents"].apply(decode_set)
            if 'consequents' in display_df.columns:
                 display_df["ผลลัพธ์ (Consequents)"] = display_df["consequents"].apply(decode_set)

            # Select and rename columns for clarity
            display_df = display_df.rename(columns={
                "support": "Support", "confidence": "Confidence", "lift": "Lift"
            })
            st.dataframe(
                display_df[["ปัจจัยนำ (Antecedents)", "ผลลัพธ์ (Consequents)", "Support", "Confidence", "Lift"]],
                use_container_width=True,
                hide_index=True
            )

            # Display interpretation guide
            st.markdown("📘 **การตีความ:**")
            st.markdown("- **Support:** สัดส่วนของเคสทั้งหมดที่พบทั้งปัจจัยนำและผลลัพธ์นี้ร่วมกัน")
            st.markdown("- **Confidence:** ความน่าจะเป็นที่จะเกิด 'ผลลัพธ์' เมื่อพบ 'ปัจจัยนำ' เหล่านี้")
            st.markdown("- **Lift > 1:** บ่งชี้ว่าการเกิดร่วมกันนี้มีนัยสำคัญ (ไม่ใช่เรื่องบังเอิญ) ยิ่งค่าสูงยิ่งสัมพันธ์กันมาก")

        else:
            st.info(f"ไม่พบกฎ Apriori ที่เกี่ยวข้องสำหรับระดับความเสี่ยง '{target_label}' ในขณะนี้")


# ----------------------------------------------------------
# 📊 TAB 4 — Clinical Summary Dashboard
# ----------------------------------------------------------
with tab4:
    st.subheader("📊 Clinical Summary & Insights Dashboard")
    st.caption("สรุปแนวโน้มผู้บาดเจ็บจากระบบ AI เพื่อใช้วางแผนเชิงกลยุทธ์และบริหารทรัพยากรโรงพยาบาล")

    # --- Load Log File ---
    df_log = None
    if os.path.exists(LOG_FILE):
        try:
            df_log = pd.read_csv(LOG_FILE, parse_dates=['timestamp'])
            df_log['timestamp'] = pd.to_datetime(df_log['timestamp']).dt.tz_localize(None) # Remove timezone for consistency if needed
            if not df_log.empty:
                 st.success(f"📁 โหลดข้อมูล Log สำเร็จ: {len(df_log):,} รายการ")
            else:
                 st.info("ไฟล์ Log ว่างเปล่า")
                 df_log = None # Treat empty log as if it doesn't exist for checks below
        except Exception as e:
            st.error(f"ไม่สามารถโหลดไฟล์ Log ({LOG_FILE}): {e}")
            df_log = None
    else:
        st.info("ยังไม่มีข้อมูล Log การทำนาย")

    # --- Reset Button ---
    if st.button("🧹 ล้างข้อมูล Log ทั้งหมด"):
        if os.path.exists(LOG_FILE):
            try:
                os.remove(LOG_FILE)
                st.success("✅ ล้างข้อมูล Log เรียบร้อยแล้ว")
                df_log = None # Update state after deletion
                st.rerun() # Rerun to reflect the change immediately
            except Exception as e:
                st.error(f"ไม่สามารถลบไฟล์ Log: {e}")
        else:
            st.info("ไม่มีไฟล์ Log ให้ลบ")

    st.markdown("---")

    # --- Dashboard Content (only if log exists and is not empty) ---
    if df_log is not None and not df_log.empty:
        total_cases = len(df_log)

        # --- KPI Overview ---
        st.markdown("### 💡 ภาพรวมสถานการณ์ (KPI Overview)")
        col1_kpi, col2_kpi, col3_kpi = st.columns(3)

        fatal_ratio = df_log["predicted_severity"].eq("เสี่ยงมาก").mean() * 100
        avg_age = df_log["age"].mean() if 'age' in df_log.columns else 'N/A'
        male_ratio = df_log["sex"].eq("ชาย").mean() * 100 if 'sex' in df_log.columns else 'N/A'

        col1_kpi.metric("จำนวนเคสทั้งหมด", f"{total_cases:,}")
        col2_kpi.metric("สัดส่วนเคสเสี่ยงมาก (Fatal)", f"{fatal_ratio:.1f}%")
        col3_kpi.metric("อายุเฉลี่ย", f"{avg_age:.1f}" if isinstance(avg_age, (int,float)) else avg_age)
        # col4_kpi.metric("สัดส่วนเพศชาย", f"{male_ratio:.1f}%" if isinstance(male_ratio, (int,float)) else male_ratio)


        # --- Charts ---
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        # 1. Severity Distribution (Pie Chart)
        with chart_col1:
            st.markdown("##### 🩸 สัดส่วนตามความรุนแรง")
            severity_counts = df_log['predicted_severity'].value_counts()
            # Ensure correct order and colors
            severity_order = ["เสี่ยงน้อย", "เสี่ยงปานกลาง", "เสี่ยงมาก"]
            severity_counts = severity_counts.reindex(severity_order, fill_value=0)
            colors_ordered = [triage_color.get(level, "#CCCCCC") for level in severity_order]

            if not severity_counts.empty:
                fig1, ax1 = plt.subplots(figsize=(4, 4))
                ax1.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
                        startangle=90, colors=colors_ordered,
                        textprops={'fontsize': 9})
                ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)
            else:
                st.caption("ไม่มีข้อมูล")


        # 2. Cases over Time (Line Chart) - Requires 'timestamp'
        with chart_col2:
            st.markdown("##### 📈 จำนวนเคสตามช่วงเวลา")
            if 'timestamp' in df_log.columns:
                 df_log['date'] = df_log['timestamp'].dt.date
                 cases_over_time = df_log.groupby('date').size()
                 if not cases_over_time.empty:
                      st.line_chart(cases_over_time)
                 else:
                      st.caption("ไม่มีข้อมูล")
            else:
                 st.caption("ไม่มีข้อมูลเวลา (timestamp)")


        # --- Risk Factor Analysis ---
        st.markdown("---")
        st.markdown("### ❗ ปัจจัยเสี่ยงที่พบบ่อยในเคส 'เสี่ยงมาก'")
        risk_cols_log = [f"risk{i}" for i in range(1, 6) if f"risk{i}" in df_log.columns]
        risk_cols_log.append('head_injury') # Include head injury
        if 'is_night' in df_log.columns: risk_cols_log.append('is_night')


        fatal_cases = df_log[df_log["predicted_severity"] == "เสี่ยงมาก"]
        if not fatal_cases.empty and risk_cols_log:
             risk_counts_fatal = fatal_cases[risk_cols_log].sum().sort_values(ascending=False)
             # Map coded risk names to readable names if possible
             risk_display_names = {
                 "risk1": "ดื่มแอลกอฮอล์", "risk2": "ใช้ยาเสพติด", "risk3": "ไม่คาดเข็มขัด",
                 "risk4": "ไม่สวมหมวก", "risk5": "ใช้โทรศัพท์",
                 "head_injury": "บาดเจ็บศีรษะ", "is_night": "เกิดกลางคืน"
             }
             risk_counts_fatal.index = risk_counts_fatal.index.map(risk_display_names).fillna(risk_counts_fatal.index)

             if not risk_counts_fatal.empty:
                 fig3, ax3 = plt.subplots(figsize=(6, 3))
                 sns.barplot(x=risk_counts_fatal.values, y=risk_counts_fatal.index, ax=ax3, palette="viridis")
                 ax3.set_xlabel("จำนวนเคส")
                 ax3.set_ylabel("ปัจจัยเสี่ยง")
                 st.pyplot(fig3)
             else:
                  st.caption("ไม่พบข้อมูลปัจจัยเสี่ยงในเคสเสี่ยงมาก")

        else:
             st.caption("ไม่มีเคส 'เสี่ยงมาก' หรือไม่มีข้อมูลปัจจัยเสี่ยง")


         # --- Insights & Recommendations ---
        st.markdown("---")
        st.markdown("### 💡 ข้อเสนอแนะเชิงกลยุทธ์")
        # Example insights based on data
        if fatal_ratio > 10: # Example threshold
             st.warning("🚨 สัดส่วนเคส 'เสี่ยงมาก' ค่อนข้างสูง: ควรทบทวนเกณฑ์ Triage หรือจัดสรรทรัพยากรห้องฉุกเฉินเพิ่มเติม")
        if not risk_counts_fatal.empty:
             top_risk = risk_counts_fatal.index[0]
             st.info(f"📌 ปัจจัยเสี่ยงที่พบบ่อยสุดในเคส 'เสี่ยงมาก' คือ '{top_risk}': พิจารณารณรงค์หรือออกมาตรการป้องกันที่เกี่ยวข้องกับปัจจัยนี้")

    else: # If df_log is None or empty
        st.info("เริ่มทำการประเมินในแท็บ 'Clinical Risk Prediction' เพื่อดูข้อมูลสรุปที่นี่")

st.markdown("---")
st.markdown("Developed by AI for Road Safety | Data Source: Injury Surveillance (IS) - MOPH")
