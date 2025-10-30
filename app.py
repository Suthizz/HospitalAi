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
import matplotlib.font_manager as fm

# ----------------------------------------------------------
# ⚙️ Page Setup (FIXED: Must be the FIRST Streamlit command)
# ----------------------------------------------------------
st.set_page_config(layout="wide", page_title="Road Accident AI Decision Support", page_icon="🏥")
st.title("🏥 Hospital AI Decision Support System")


# ตั้งค่าให้ matplotlib แสดงผลได้เหมาะสม
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (6, 4)

# Configure matplotlib to support Thai font
# Use font_manager to find a font that supports Thai characters
thai_font_name = None
try:
    # Rebuild font cache if necessary, but can be slow
    # fm._load_fontmanager(try_read_cache=False) # Use this if fonts still not found after packages.txt
    
    # Find the best available Thai font
    thai_fonts = ['Loma', 'Sarabun', 'TH Sarabun New', 'Kinnari', 'Garuda', 'Leelawadee', 'Tahoma']
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    for font in thai_fonts:
        if font in font_list:
            thai_font_name = font
            break

    if thai_font_name:
        plt.rcParams['font.family'] = thai_font_name
        # st.caption(f"✅ Using Thai font: {plt.rcParams['font.family']}") # Uncomment for debugging
    else:
        st.warning("⚠️ No suitable Thai font found (e.g., Loma, Sarabun). Thai characters in graphs might not display correctly.")
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['font.sans-serif'] = [thai_font_name if thai_font_name else 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False # Allow minus sign

except Exception as e:
     st.warning(f"⚠️ Error setting Thai font: {e}. Graphs might show garbled text.")


# ----------------------------------------------------------
# 📦 Load Models + Configs (FIXED: Removed st.write from cache)
# ----------------------------------------------------------

@st.cache_resource
def load_all():
    """Loads models and returns them along with status messages."""
    loaded_items = {}
    msgs = [] # List to store status messages
    
    st.write("⏳ Attempting to Load Models and Configurations...") # This one is ok as it's informational before loading

    # 🔹 CatBoost Model
    try:
        model = joblib.load("predict_catboost_multi.pkl")
        loaded_items["model"] = model
        msgs.append("✅ Clinical Severity Model (predict_catboost_multi.pkl) Loaded.")
    except Exception as e:
        loaded_items["model"] = None
        msgs.append(f"❌ Clinical Severity Model (predict_catboost_multi.pkl) NOT FOUND. (Error: {e}) -> Prediction will not work.")

    # 🔹 Encoders / Features / K-Means / Apriori
    try:
        encoders = joblib.load("encoders_multi.pkl")
        loaded_items["encoders"] = encoders
        msgs.append("✅ Encoders (encoders_multi.pkl) Loaded.")
    except Exception as e:
        loaded_items["encoders"] = None
        msgs.append(f"⚠️ Encoders (encoders_multi.pkl) NOT FOUND. (Error: {e}) -> Preprocessing might fail.")

    try:
        with open("features_multi.json", "r", encoding="utf-8") as f: 
            features = json.load(f)
        loaded_items["features"] = features
        msgs.append("✅ Features List (features_multi.json) Loaded.")
    except Exception as e:
        loaded_items["features"] = ['age', 'sex', 'is_night', 'head_injury', 'mass_casualty', 'risk1', 'risk2', 'risk3', 'risk4', 'risk5', 'cannabis', 'amphetamine', 'drugs', 'activity', 'aplace', 'prov']
        msgs.append(f"⚠️ Features List (features_multi.json) NOT FOUND. (Error: {e}) -> Using default list.")

    try:
        kmeans = joblib.load("kmeans_cluster_model.pkl")
        scaler = joblib.load("scaler_cluster.pkl")
        loaded_items["kmeans"] = kmeans
        loaded_items["scaler"] = scaler
        msgs.append("✅ K-Means Cluster Model and Scaler Loaded.")
    except Exception as e:
        loaded_items["kmeans"], loaded_items["scaler"] = None, None
        msgs.append(f"⚠️ K-Means Cluster Model or Scaler NOT FOUND. (Error: {e}) -> Clustering unavailable.")

    try:
        rules_minor = joblib.load("apriori_rules_minor.pkl")
        rules_severe = joblib.load("apriori_rules_severe.pkl")
        rules_fatal = joblib.load("apriori_rules_fatal.pkl")
        loaded_items["rules_minor"] = rules_minor
        loaded_items["rules_severe"] = rules_severe
        loaded_items["rules_fatal"] = rules_fatal
        msgs.append("✅ Apriori Association Rules Loaded.")
    except Exception as e:
        loaded_items["rules_minor"], loaded_items["rules_severe"], loaded_items["rules_fatal"] = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        msgs.append(f"⚠️ Apriori Association Rules NOT FOUND. (Error: {e}) -> Risk association unavailable.")

    return loaded_items, msgs

# เรียกใช้การโหลด
loaded_items, load_messages = load_all()

# --- Display Loading Status (FIXED: Moved expander outside cache) ---
with st.expander("📂 สถานะการโหลดไฟล์โมเดล", expanded=False):
    for msg in load_messages:
        if "✅" in msg:
            st.caption(msg)
        else:
            st.caption(f":warning: {msg}")

# Assign loaded items to variables
model = loaded_items.get("model")
encoders = loaded_items.get("encoders")
features = loaded_items.get("features")
kmeans = loaded_items.get("kmeans")
scaler = loaded_items.get("scaler")
rules_minor = loaded_items.get("rules_minor")
rules_severe = loaded_items.get("rules_severe")
rules_fatal = loaded_items.get("rules_fatal")


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
# Add default mapping for potential 'Unknown' or missing keys
activity_mapping.setdefault("Unknown", "ไม่ระบุ")
aplace_mapping.setdefault("Unknown", "ไม่ระบุ")
prov_mapping.setdefault("Unknown", "ไม่ระบุ")

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
def preprocess_input(data_dict, loaded_encoders, expected_features, activity_map, aplace_map, prov_map):
    """Preprocesses user input dictionary based on loaded encoders and features."""
    if not expected_features:
        st.error("Feature list is empty. Cannot preprocess.")
        return None
    if loaded_encoders is None:
         st.warning("Encoders not loaded. Categorical features might not be processed correctly.")
         loaded_encoders = {} # Use an empty dict to avoid errors

    df = pd.DataFrame([data_dict])
    
    # Map display values back to codes before encoding
    reverse_activity = {v: k for k, v in activity_map.items()}
    reverse_aplace = {v: k for k, v in aplace_map.items()}
    reverse_prov = {v: k for k, v in prov_map.items()}

    # Use .get() with a default 'Unknown' code if the display value isn't found
    df['activity'] = df['activity'].apply(lambda x: reverse_activity.get(x, "Unknown"))
    df['aplace'] = df['aplace'].apply(lambda x: reverse_aplace.get(x, "Unknown"))
    df['prov'] = df['prov'].apply(lambda x: reverse_prov.get(x, "Unknown"))

    # Reindex first to ensure all expected columns are present
    df = df.reindex(columns=expected_features, fill_value=0) # Use 0 as default fill

    # Clean potential string issues and fill NaNs *after* reindexing
    clean_values = ["None", "N/A", "ไม่ระบุ", "N", "nan", "NaN", "", " "]
    df = df.replace(clean_values, "Unknown") # Replace various missing representations with 'Unknown'
    df = df.fillna("Unknown") # Fill any actual NaNs

    # Encode categorical features using loaded encoders
    for col, le in loaded_encoders.items():
        if col in df.columns and isinstance(le, LabelEncoder): # Check if encoder is valid
            df[col] = df[col].astype(str) # Ensure string type before transform
            
            unknown_class_index = 0 # Default index if 'Unknown' not in classes
            if "Unknown" in le.classes_:
                try:
                  unknown_class_index = int(le.transform(["Unknown"])[0])
                except ValueError:
                  pass # Should not happen

            df[col] = df[col].apply(lambda x: int(le.transform([x])[0]) if x in le.classes_ else unknown_class_index)
        elif col in df.columns: # Fallback if encoder invalid or missing
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    # --- Create Engineered Features (must match training - check feature list) ---
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
        if "age_group_60plus" in expected_features:
             df["age_group_60plus"] = (df["age"] >= 60).astype(int)

    risk_cols_in_features = [f"risk{i}" for i in range(1, 6) if f"risk{i}" in expected_features]
    if "risk_count" in expected_features:
        for r_col in risk_cols_in_features:
            if r_col in df.columns:
                 df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(0)
            else:
                 df[r_col] = 0
        df["risk_count"] = df[risk_cols_in_features].sum(axis=1)

    if 'is_night' in df.columns:
        df['is_night'] = pd.to_numeric(df['is_night'], errors='coerce').fillna(0)
        if "night_flag" in expected_features:
            df["night_flag"] = df["is_night"].astype(int)
    elif "night_flag" in expected_features:
         df["night_flag"] = 0

    # Ensure all columns are numeric at the end
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Final check: Ensure column order matches exactly
    try:
        df = df[expected_features]
    except KeyError as e:
        missing_keys = set(expected_features) - set(df.columns)
        st.error(f"Mismatch in expected features during final step. Missing: {missing_keys}. Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred ensuring final feature order: {e}")
        return None

    return df

# ----------------------------------------------------------
# 📄 Streamlit App Layout
# ----------------------------------------------------------

# Initialize session state for form inputs if not already done
# (Using keys based on the input variable name for simplicity)
defaults = {
    'age': 30, 'sex': "ชาย", 'is_night': False, 'head_injury': False,
    'mass_casualty': False, 
    'activity': list(activity_mapping.values())[0], # Default to "เดินเท้า"
    'aplace': list(aplace_mapping.values())[0], # Default to "บ้านพักอาศัย"
    'prov': list(prov_mapping.values())[0], # Default to "กรุงเทพมหานคร"
    'risk1': False, 'risk2': False, 'risk3': False, 'risk4': False, 'risk5': False,
    'cannabis': False, 'amphetamine': False, 'drugs': False,
    'predicted_severity': "ยังไม่มีข้อมูล"
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Function to reset form inputs and clear log
def reset_all_data():
    # Reset form inputs in session state
    for key, value in defaults.items():
        st.session_state[key] = value
    
    st.session_state.predicted_severity = "ยังไม่มีข้อมูล" # Reset prediction explicitly

    # Clear the prediction log file
    log_file = "prediction_log.csv"
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            st.success("✅ ข้อมูลทั้งหมดและ Log ถูกล้างเรียบร้อยแล้ว")
        except Exception as e:
            st.error(f"⚠️ ไม่สามารถล้างไฟล์ Log: {e}")
    else:
        st.info("⚠️ ไม่มีข้อมูล Log ให้ล้าง")
    
    # Clear processed data for tabs
    if 'processed_input_for_tabs' in st.session_state:
        st.session_state.processed_input_for_tabs = None
    if 'raw_input_for_tabs' in st.session_state:
        st.session_state.raw_input_for_tabs = {}
    if 'submit_pressed' in st.session_state:
        st.session_state.submit_pressed = False


# ----------------------------------------------------------
# 🧠 TAB 1 — Clinical Risk Prediction
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Clinical Risk Prediction",
    "👥 Cluster Insight",
    "🧩 Risk Association",
    "📊 Clinical Summary Dashboard"
])

with tab1:
    st.subheader("🧠 Clinical Severity Prediction")
    st.caption("ระบบประเมินระดับความรุนแรงของผู้บาดเจ็บแบบเรียลไทม์")
    
    if model is None or features is None or encoders is None:
        st.error("❌ โมเดลหลัก (CatBoost), Encoders หรือ Features list ไม่พร้อมใช้งาน ไม่สามารถทำนายได้ กรุณาตรวจสอบไฟล์")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                # Use session state keys for inputs
                age = st.slider("อายุ", 0, 100, key='age')
                sex = st.radio("เพศ", ["ชาย", "หญิง"], key='sex')
                is_night = st.checkbox("เหตุการณ์กลางคืน", key='is_night')
            with col2:
                head_injury = st.checkbox("บาดเจ็บที่ศีรษะ", key='head_injury')
                mass_casualty = st.checkbox("เหตุการณ์หมู่ (Mass Casualty)", key='mass_casualty')
                activity = st.selectbox("ลักษณะกิจกรรม", list(activity_mapping.values()), key='activity')
            with col3:
                aplace = st.selectbox("สถานที่เกิดเหตุ", list(aplace_mapping.values()), key='aplace')
                prov = st.selectbox("จังหวัด", list(prov_mapping.values()), key='prov')
                
                st.markdown("**ปัจจัยเสี่ยง:**")
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                with risk_col1:
                    risk1 = st.checkbox("Risk 1: ไม่สวมหมวก/เข็มขัด", key='risk1', help="ไม่สวมหมวกนิรภัย / เข็มขัดนิรภัย")
                    risk2 = st.checkbox("Risk 2: ขับเร็ว/ประมาท", key='risk2')
                    risk3 = st.checkbox("Risk 3: เมา / ดื่มสุรา", key='risk3')
                with risk_col2:
                    risk4 = st.checkbox("Risk 4: ผู้สูงอายุ / เด็กเล็ก", key='risk4')
                    risk5 = st.checkbox("Risk 5: บาดเจ็บหลายตำแหน่ง", key='risk5')
                
                st.markdown("**สารเสพติด:**")
                drug_col1, drug_col2, drug_col3 = st.columns(3)
                with drug_col1:
                    cannabis = st.checkbox("กัญชา", key='cannabis')
                with drug_col2:
                    amphetamine = st.checkbox("แอมเฟตามีน", key='amphetamine')
                with drug_col3:
                    drugs = st.checkbox("ยาอื่น ๆ", key='drugs')

            col_buttons = st.columns(2)
            with col_buttons[0]:
                submit_button = st.form_submit_button("ประเมินความเสี่ยง")
            with col_buttons[1]:
                # Use the reset function here
                clear_button = st.form_submit_button("เคลียร์ข้อมูลทั้งหมด", on_click=reset_all_data) 

        if submit_button:
            st.session_state.submit_pressed = True # Mark submission
            
            # 1. จัดรูปแบบ Input Data (using session state values)
            input_data = {
                "age": st.session_state.age,
                "sex": '1' if st.session_state.sex == "ชาย" else '2', # Map back to '1'/'2' code
                "is_night": int(st.session_state.is_night),
                "head_injury": int(st.session_state.head_injury),
                "mass_casualty": int(st.session_state.mass_casualty),
                "risk1": int(st.session_state.risk1), "risk2": int(st.session_state.risk2), "risk3": int(st.session_state.risk3),
                "risk4": int(st.session_state.risk4), "risk5": int(st.session_state.risk5),
                "cannabis": int(st.session_state.cannabis), "amphetamine": int(st.session_state.amphetamine), "drugs": int(st.session_state.drugs),
                "activity": st.session_state.activity, # Pass display value
                "aplace": st.session_state.aplace,     # Pass display value
                "prov": st.session_state.prov          # Pass display value
            }
            
            # Store raw input (using display values) for other tabs
            st.session_state.raw_input_for_tabs = {
                "age": st.session_state.age, "sex": st.session_state.sex, "is_night": st.session_state.is_night, 
                "head_injury": st.session_state.head_injury, "mass_casualty": st.session_state.mass_casualty, 
                "risk1": st.session_state.risk1, "risk2": st.session_state.risk2, "risk3": st.session_state.risk3, 
                "risk4": st.session_state.risk4, "risk5": st.session_state.risk5, "cannabis": st.session_state.cannabis, 
                "amphetamine": st.session_state.amphetamine, "drugs": st.session_state.drugs, 
                "activity": st.session_state.activity, "aplace": st.session_state.aplace, "prov": st.session_state.prov
            }

            # 2. Preprocess
            X_input = preprocess_input(input_data, encoders, features, activity_mapping, aplace_mapping, prov_mapping)
            st.session_state.processed_input_for_tabs = X_input # Store processed input

            # 3. Predict
            if X_input is not None:
                try:
                    probs = model.predict_proba(X_input)[0]
                    pred_class = int(np.argmax(probs))
                    current_label = severity_map.get(pred_class, "ไม่ทราบ")
                    color = triage_color.get(current_label, "#2196F3")

                    # Update session state with the prediction result
                    st.session_state.predicted_severity = current_label

                    # 4. แสดงผล
                    st.subheader("🔥 ผลการประเมินความเสี่ยง")
                    st.markdown(f"<h3 style='color:{color}'>{current_label}</h3>", unsafe_allow_html=True)
                    st.info(f"💡 แนวทางทางการแพทย์เบื้องต้น: {advice_map.get(current_label, 'ไม่ทราบแนวทาง')}")
                    st.caption(f"🧠 ความมั่นใจของระบบ: {probs[pred_class]*100:.1f}%")
                    # Optional: Show probabilities for all classes
                    with st.expander("ดูความน่าจะเป็นของแต่ละระดับ"):
                        prob_df = pd.DataFrame({
                            "ระดับความเสี่ยง": [severity_map.get(i, f"Class {i}") for i in range(len(probs))],
                            "ความน่าจะเป็น": [f"{p*100:.1f}%" for p in probs]
                        })
                        st.dataframe(prob_df, hide_index=True)


                    # 5. บันทึก Prediction Log (For Dashboard)
                    log_file = "prediction_log.csv"
                    try:
                        log_entry = {
                            "timestamp": pd.Timestamp.now(tz='Asia/Bangkok').strftime('%Y-%m-%d %H:%M:%S %Z'),
                            "age": st.session_state.age,
                            "sex": st.session_state.sex, # Log display value
                            "predicted_severity": current_label,
                            "prov": st.session_state.prov, # Log display value
                            "is_night": int(st.session_state.is_night),
                            "risk1": int(st.session_state.risk1),
                            "risk2": int(st.session_state.risk2),
                            "risk3": int(st.session_state.risk3),
                            "risk4": int(st.session_state.risk4),
                            "risk5": int(st.session_state.risk5),
                            "head_injury": int(st.session_state.head_injury),
                            "cannabis": int(st.session_state.cannabis),
                            "amphetamine": int(st.session_state.amphetamine),
                            "drugs": int(st.session_state.drugs)
                        }
                        new_row = pd.DataFrame([log_entry])
                        
                        write_header = not os.path.exists(log_file)
                        new_row.to_csv(log_file, mode="a", index=False, header=write_header, encoding='utf-8-sig')
                        st.success("📁 บันทึกผลการประเมินเข้าสู่ระบบ Log แล้ว")
                    
                    except Exception as log_e:
                        st.warning(f"⚠️ ไม่สามารถบันทึก Log: {log_e}")

                except Exception as e:
                    st.error(f"❌ Error ในการทำนาย: {e}")
            else:
                st.error("❌ ไม่สามารถประมวลผลข้อมูล Input สำหรับการทำนายได้")

# ----------------------------------------------------------
# 👥 TAB 2 — K-Means Cluster Analysis
# ----------------------------------------------------------
with tab2:
    st.subheader("👥 Patient Segmentation (K-Means)")
    st.caption("วิเคราะห์กลุ่มผู้บาดเจ็บ เพื่อใช้ในการจัดสรรทรัพยากรและการป้องกันเชิงรุก")

    if not st.session_state.get('submit_pressed', False):
        st.info("🕐 กรุณาประเมินความเสี่ยงในแท็บ 'Clinical Risk Prediction' ก่อน")
    elif kmeans is None or scaler is None:
        st.warning("⚠️ โมเดล K-Means / Scaler ไม่พร้อมใช้งาน")
    elif st.session_state.get('processed_input_for_tabs') is None:
        st.warning("⚠️ ไม่พบข้อมูลที่ประมวลผลแล้วจากแท็บแรก")
    else:
        # --- Display Patient Summary (using raw_input_for_tabs) ---
        st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
        raw_input = st.session_state.get('raw_input_for_tabs', {})
        summary_cols = st.columns(3)
        summary_cols[0].metric("อายุ", f"{raw_input.get('age', 'N/A')} ปี")
        summary_cols[1].metric("เพศ", raw_input.get('sex', 'N/A'))
        summary_cols[2].metric("ระดับความเสี่ยง", st.session_state.get('prediction_label', 'N/A'))

        risk_summary = []
        if raw_input.get('risk1'): risk_summary.append("ดื่มแอลกอฮอล์")
        if raw_input.get('risk2'): risk_summary.append("ใช้ยาเสพติด(ทั่วไป)")
        if raw_input.get('risk3'): risk_summary.append("ไม่คาดเข็มขัด")
        if raw_input.get('risk4'): risk_summary.append("ไม่สวมหมวกนิรภัย")
        if raw_input.get('risk5'): risk_summary.append("ใช้โทรศัพท์ขณะขับ")
        if raw_input.get('head_injury'): risk_summary.append("บาดเจ็บศีรษะ")
        if raw_input.get('cannabis'): risk_summary.append("กัญชา")
        if raw_input.get('amphetamine'): risk_summary.append("ยาบ้า")
        if raw_input.get('drugs'): risk_summary.append("ยาอื่น ๆ")
        if raw_input.get('is_night'): risk_summary.append("เกิดกลางคืน")
        st.markdown(f"**ปัจจัยเด่น:** {', '.join(risk_summary) if risk_summary else '-'}")
        st.markdown("---")

        # --- Perform Clustering (using processed_input_for_tabs) ---
        try:
            X_input_processed = st.session_state.processed_input_for_tabs
            
            cluster_features_used = None
            if hasattr(scaler, "feature_names_in_"):
                cluster_features_used = scaler.feature_names_in_
            elif hasattr(scaler, 'n_features_in_'):
                 st.warning("Scaler object lacks explicit feature names. Attempting to infer feature list.")
                 # Infer features based on notebook's cluster_cols
                 cluster_cols_base = ["age", "sex", "is_night", "head_injury", "risk1", "risk2", "risk3", "risk4", "risk5", "cannabis", "amphetamine", "drugs"]
                 # Check if these columns exist in the processed input
                 if all(c in X_input_processed.columns for c in cluster_cols_base):
                      cluster_features_used = cluster_cols_base
                 else:
                      st.error(f"Cannot infer clustering features. Scaler expected {scaler.n_features_in_}, but base columns not found in processed data.")
            else:
                 st.error("Cannot determine features expected by the scaler. Clustering aborted.")

            if cluster_features_used:
                X_cluster_input = X_input_processed.copy()
                # Ensure all required columns exist, fill missing with 0
                for col in cluster_features_used:
                    if col not in X_cluster_input.columns:
                        X_cluster_input[col] = 0
                
                # Select and reorder columns
                try:
                    X_cluster_input = X_cluster_input[cluster_features_used]
                    
                    X_scaled = scaler.transform(X_cluster_input)
                    cluster_label = int(kmeans.predict(X_scaled)[0])

                    # Cluster Descriptions (Based on Notebook Analysis)
                    cluster_desc = {
                        0: "กลุ่มทั่วไป/เพศชาย (Cluster 0): พบบ่อยสุด, มักเป็นชาย, ความเสี่ยงและบาดเจ็บศีรษะปานกลาง",
                        1: "กลุ่มเพศหญิง/ความเสี่ยงต่ำ (Cluster 1): พบบ่อยรองลงมา, มักเป็นหญิง, ความเสี่ยงต่ำกว่ากลุ่ม 0",
                        2: "กลุ่มเสี่ยงยาเสพติด (Cluster 2): พบน้อย, มีสัดส่วนการใช้ยา (risk2, cannabis, amphetamine) และบาดเจ็บศีรษะสูงที่สุด"
                    }

                    st.markdown(f"### 📊 ผลการจัดกลุ่ม: **Cluster {cluster_label}**")
                    st.info(f"**ลักษณะกลุ่ม (โดยประมาณ):** {cluster_desc.get(cluster_label, 'ยังไม่มีคำอธิบายสำหรับกลุ่มนี้')}")
                    st.caption("💡 ใช้ข้อมูลนี้เพื่อทำความเข้าใจลักษณะผู้ป่วยที่คล้ายกัน และวางแผนทรัพยากรหรือแคมเปญป้องกันที่ตรงกลุ่มเป้าหมาย")

                except KeyError as e:
                     st.error(f"Error selecting/ordering columns for scaler: {e}")
                except Exception as e:
                     st.error(f"เกิดข้อผิดพลาดระหว่างการจัดกลุ่ม K-Means: {e}")
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการเตรียมข้อมูลสำหรับ Clustering: {e}")


# ----------------------------------------------------------
# 🧩 TAB 3 — Apriori Risk Association
# ----------------------------------------------------------
with tab3:
    st.subheader("🧩 Risk Association Analysis (Apriori)")
    st.caption("วิเคราะห์ความสัมพันธ์ของปัจจัยเสี่ยง เพื่อวางแผนป้องกันและสนับสนุนการตัดสินใจเชิงนโยบาย")

    if not st.session_state.get('submit_pressed', False):
        st.info("🕐 กรุณาประเมินความเสี่ยงในแท็บ 'Clinical Risk Prediction' ก่อน")
    elif rules_minor is None and rules_severe is None and rules_fatal is None:
        st.warning("⚠️ ไม่พบไฟล์กฎ Apriori ที่โหลดไว้")
    else:
        # --- Display Patient Summary ---
        st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
        raw_input = st.session_state.get('raw_input_for_tabs', {})
        summary_cols = st.columns(3)
        summary_cols[0].metric("อายุ", f"{raw_input.get('age', 'N/A')} ปี")
        summary_cols[1].metric("เพศ", raw_input.get('sex', 'N/A'))
        summary_cols[2].metric("ระดับความเสี่ยง", st.session_state.get('prediction_label', 'N/A'))

        risk_tags = []
        if raw_input.get('risk1'): risk_tags.append("ดื่มแอลกอฮอล์")
        if raw_input.get('risk2'): risk_tags.append("ใช้ยาเสพติด(ทั่วไป)")
        if raw_input.get('risk3'): risk_tags.append("ไม่คาดเข็มขัด")
        if raw_input.get('risk4'): risk_summary.append("ไม่สวมหมวกนิรภัย")
        if raw_input.get('risk5'): risk_summary.append("ใช้โทรศัพท์ขณะขับ")
        if raw_input.get('head_injury'): risk_tags.append("บาดเจ็บศีรษะ")
        if raw_input.get('cannabis'): risk_tags.append("กัญชา")
        if raw_input.get('amphetamine'): risk_tags.append("ยาบ้า")
        if raw_input.get('drugs'): risk_tags.append("ยาอื่น ๆ")
        if raw_input.get('is_night'): risk_tags.append("เกิดกลางคืน")
        st.markdown(f"**ปัจจัยเด่น:** {', '.join(risk_tags) if risk_tags else '-'}")
        st.markdown("---")

        # --- Select and Display Relevant Rules ---
        st.markdown("### 🔗 กฎความสัมพันธ์ที่เกี่ยวข้อง (Top 5 by Lift)")

        target_label = st.session_state.get('prediction_label', 'N/A')
        rules_df_to_show = None
        rules_title = f"กฎที่นำไปสู่: {target_label}"

        is_valid_df = lambda df: isinstance(df, pd.DataFrame) and not df.empty

        if target_label == "เสี่ยงน้อย" and is_valid_df(rules_minor):
            rules_df_to_show = rules_minor
        elif target_label == "เสี่ยงปานกลาง" and is_valid_df(rules_severe):
            rules_df_to_show = rules_severe
        elif target_label == "เสี่ยงมาก" and is_valid_df(rules_fatal):
            rules_df_to_show = rules_fatal

        if rules_df_to_show is not None:
            st.markdown(f"**{rules_title}**")
            display_df = rules_df_to_show.head(5).copy()

            if 'antecedents' in display_df.columns:
                 display_df["ปัจจัยนำ (Antecedents)"] = display_df["antecedents"].apply(decode_set)
            else:
                 display_df["ปัจจัยนำ (Antecedents)"] = "N/A"

            if 'consequents' in display_df.columns:
                 display_df["ผลลัพธ์ (Consequents)"] = display_df["consequents"].apply(decode_set)
            else:
                 display_df["ผลลัพธ์ (Consequents)"] = "N/A"

            cols_to_display = ["ปัจจัยนำ (Antecedents)", "ผลลัพธ์ (Consequents)"]
            rename_map = {"support": "Support", "confidence": "Confidence", "lift": "Lift"}
            for col, new_name in rename_map.items():
                if col in display_df.columns:
                    cols_to_display.append(new_name)
                    display_df = display_df.rename(columns={col: new_name})
            
            if "ปัจจัยนำ (Antecedents)" in display_df.columns and "ผลลัพธ์ (Consequents)" in display_df.columns:
                # Display Top Rule Insight
                top_rule = display_df.iloc[0]
                st.markdown(
                    f"""
                    <div style='background-color:#262730;border-radius:10px;padding:12px;margin-bottom:10px; border: 1px solid #444;'>
                    💡 <b>Insight ที่พบบ่อยสุด (Lift สูงสุด):</b> 
                    <br>
                    พบว่าผู้ที่มี <b>{top_rule['ปัจจัยนำ (Antecedents)']}</b>
                    <br>
                    มีแนวโน้ม <b>{top_rule['ผลลัพธ์ (Consequents)']}</b>
                    <br>
                    <small>(Confidence: {top_rule.get('Confidence', 0)*100:.1f}%, Lift = {top_rule.get('Lift', 0):.2f})</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display Table
                st.dataframe(
                    display_df[cols_to_display],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Display interpretation
                st.markdown("📘 **การตีความ:**")
                st.markdown("- **Support:** สัดส่วนของเคสทั้งหมดที่พบทั้งปัจจัยนำและผลลัพธ์นี้ร่วมกัน")
                st.markdown("- **Confidence:** ความน่าจะเป็นที่จะเกิด 'ผลลัพธ์' เมื่อพบ 'ปัจจัยนำ' เหล่านี้")
                st.markdown("- **Lift > 1:** บ่งชี้ว่าการเกิดร่วมกันนี้มีนัยสำคัญ (ไม่ใช่เรื่องบังเอิญ) ยิ่งค่าสูงยิ่งสัมพันธ์กันมาก")
            else:
                 st.error("ไม่สามารถแสดงกฎ Apriori ได้ (คอลัมน์ antecedents/consequents ผิดพลาด)")

        else:
            st.info(f"📭 ยังไม่มีกฎ Apriori ที่เกี่ยวข้องสำหรับความเสี่ยงระดับ '{target_label}' หรือไฟล์กฎว่างเปล่า")


# ----------------------------------------------------------
# 📊 TAB 4 — Clinical Summary Dashboard
# ----------------------------------------------------------
with tab4:
    st.subheader("📊 Clinical Summary & Insights Dashboard")
    st.caption("สรุปแนวโน้มผู้บาดเจ็บจากระบบ AI เพื่อใช้วางแผนเชิงกลยุทธ์และบริหารทรัพยากรโรงพยาบาล")

    log_file = "prediction_log.csv"
    df_log = None
    log_load_error = None

    # 1. โหลดข้อมูล Log
    if os.path.exists(log_file):
        try:
            df_log = pd.read_csv(log_file, parse_dates=['timestamp'], encoding='utf-8-sig')
            df_log['timestamp'] = pd.to_datetime(df_log['timestamp'], errors='coerce').dt.tz_localize(None)
            df_log = df_log.dropna(subset=['timestamp'])

            if not df_log.empty:
                 st.success(f"📁 โหลดข้อมูล Log สำเร็จ: {len(df_log):,} รายการ")
            else:
                 st.info("ไฟล์ Log ว่างเปล่า หรือไม่มีข้อมูลเวลาที่ถูกต้อง")
                 df_log = None
        except pd.errors.EmptyDataError:
             st.info("ไฟล์ Log ว่างเปล่า")
             df_log = None
        except Exception as e:
            log_load_error = f"ไม่สามารถโหลดไฟล์ Log ({LOG_FILE}): {e}"
            st.error(log_load_error)
            df_log = None
    else:
        st.info("ยังไม่มีข้อมูล Log การทำนาย (เริ่มประเมินในแท็บแรก)")

    # 2. ปุ่ม Reset (ย้ายมาไว้ใต้การโหลด)
    if st.button("🧹 ล้างข้อมูล Log ทั้งหมด (Reset Dashboard)"):
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                st.success("✅ ล้างข้อมูล Log เรียบร้อยแล้ว")
                df_log = None
                st.rerun()
            except Exception as e:
                st.error(f"ไม่สามารถลบไฟล์ Log: {e}")
        else:
            st.info("ไม่มีไฟล์ Log ให้ลบ")

    st.markdown("---")

    # 3. Dashboard Content
    if df_log is not None and not df_log.empty:
        total_cases = len(df_log)

        # 3.1 KPI Overview
        st.markdown("### 💡 ภาพรวมสถานการณ์ (KPI Overview)")
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

        fatal_ratio = df_log["predicted_severity"].eq("เสี่ยงมาก").mean() * 100 if "predicted_severity" in df_log.columns else 0
        avg_age = df_log["age"].mean() if 'age' in df_log.columns and pd.api.types.is_numeric_dtype(df_log['age']) else 'N/A'
        male_ratio = df_log["sex"].eq("ชาย").mean() * 100 if 'sex' in df_log.columns else 'N/A'

        col_kpi1.metric("จำนวนเคสทั้งหมด", f"{total_cases:,}")
        col_kpi2.metric("สัดส่วนเคสเสี่ยงมาก (Fatal)", f"{fatal_ratio:.1f}%")
        col_kpi3.metric("อายุเฉลี่ย", f"{avg_age:.1f}" if isinstance(avg_age, (int,float)) else avg_age)
        
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        # 3.2 Severity Distribution (Pie Chart)
        with chart_col1:
            st.markdown("##### 🩸 สัดส่วนตามความรุนแรง")
            if 'predicted_severity' in df_log.columns:
                severity_counts = df_log['predicted_severity'].value_counts()
                severity_order = ["เสี่ยงน้อย", "เสี่ยงปานกลาง", "เสี่ยงมาก"]
                severity_counts = severity_counts.reindex(severity_order, fill_value=0)
                colors_ordered = [triage_color.get(level, "#CCCCCC") for level in severity_order]

                if not severity_counts.empty and severity_counts.sum() > 0:
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    ax1.pie(
                        severity_counts, 
                        labels=severity_counts.index, 
                        autopct='%1.1f%%',
                        startangle=90, 
                        colors=colors_ordered,
                        textprops={'fontsize': 9, 'color': 'white' if sum(severity_counts) > 10 else 'black'}, # Adjust text color based on size
                        wedgeprops={'edgecolor': '#333', 'linewidth': 0.5} # Add edge color
                    )
                    ax1.axis('equal')
                    st.pyplot(fig1)
                else:
                    st.caption("ไม่มีข้อมูลความรุนแรง")
            else:
                st.caption("ไม่มีข้อมูลคอลัมน์ 'predicted_severity'")

        # 3.3 Cases over Time (Line Chart)
        with chart_col2:
            st.markdown("##### 📈 จำนวนเคสตามช่วงเวลา")
            if 'timestamp' in df_log.columns:
                 df_log['date'] = df_log['timestamp'].dt.date
                 cases_over_time = df_log.groupby('date').size().rename("จำนวนเคส")
                 if not cases_over_time.empty:
                      st.line_chart(cases_over_time)
                 else:
                      st.caption("ไม่มีข้อมูลเคสในช่วงเวลาที่บันทึก")
            else:
                 st.caption("ไม่มีข้อมูลเวลา (timestamp) ใน Log")
        
        st.markdown("---")

        # 3.4 Risk Factor Analysis (Bar Chart)
        st.markdown("### ❗ ปัจจัยเสี่ยงที่พบบ่อยในเคส 'เสี่ยงมาก'")
        
        risk_cols_in_log = [
            'risk1', 'risk2', 'risk3', 'risk4', 'risk5',
            'head_injury', 'is_night', 'cannabis', 'amphetamine', 'drugs'
        ]
        # Filter only columns that actually exist in the log
        risk_cols_available = [col for col in risk_cols_in_log if col in df_log.columns]

        if "predicted_severity" in df_log.columns:
            fatal_cases = df_log[df_log["predicted_severity"] == "เสี่ยงมาก"]
            
            if not fatal_cases.empty and risk_cols_available:
                # Ensure risk columns are numeric (convert bools/objects if necessary)
                fatal_cases_numeric = fatal_cases.copy()
                for r_col in risk_cols_available:
                    fatal_cases_numeric[r_col] = pd.to_numeric(fatal_cases_numeric[r_col], errors='coerce').fillna(0)

                # Sum only columns that are binary (0 or 1)
                binary_risk_cols = [col for col in risk_cols_available if fatal_cases_numeric[col].isin([0, 1]).all()]
                
                if binary_risk_cols:
                    risk_counts_fatal = fatal_cases_numeric[binary_risk_cols].sum().sort_values(ascending=False)
                    risk_counts_fatal = risk_counts_fatal[risk_counts_fatal > 0] # Filter out 0 counts

                    risk_display_names = {
                        "risk1": "ดื่มแอลกอฮอล์", "risk2": "ใช้ยาเสพติด", "risk3": "ไม่คาดเข็มขัด",
                        "risk4": "ไม่สวมหมวก", "risk5": "ใช้โทรศัพท์",
                        "head_injury": "บาดเจ็บศีรษะ", "is_night": "เกิดกลางคืน",
                        "cannabis": "กัญชา", "amphetamine": "ยาบ้า", "drugs": "ยาอื่น ๆ"
                    }
                    risk_counts_fatal.index = risk_counts_fatal.index.map(risk_display_names).fillna(risk_counts_fatal.index)

                    if not risk_counts_fatal.empty:
                        fig3, ax3 = plt.subplots(figsize=(7, max(3, len(risk_counts_fatal)*0.4)))
                        sns.barplot(x=risk_counts_fatal.values, y=risk_counts_fatal.index, ax=ax3, palette="viridis")
                        ax3.set_xlabel("จำนวนเคส 'เสี่ยงมาก' ที่พบปัจจัยนี้")
                        ax3.set_ylabel("ปัจจัยเสี่ยง")
                        plt.tight_layout()
                        st.pyplot(fig3)
                    else:
                        st.caption("ไม่พบปัจจัยเสี่ยง (0/1) ที่ระบุในเคสเสี่ยงมาก")
                else:
                     st.caption("ไม่พบคอลัมน์ปัจจัยเสี่ยงที่เป็นค่า 0/1 ในข้อมูล Log")
            elif fatal_cases.empty:
                st.caption("ไม่มีเคส 'เสี่ยงมาก' ที่บันทึกไว้")
            else:
                st.caption("ไม่มีข้อมูลปัจจัยเสี่ยงใน Log")
        else:
             st.caption("ไม่มีข้อมูลคอลัมน์ 'predicted_severity' ใน Log")
             
        # 3.5 Insights & Recommendations
        st.markdown("---")
        st.markdown("### 💡 ข้อเสนอแนะเชิงกลยุทธ์")
        insights_generated = False
        if fatal_ratio > 10: # Example threshold
             st.warning("🚨 สัดส่วนเคส 'เสี่ยงมาก' ค่อนข้างสูง: ควรทบทวนเกณฑ์ Triage หรือจัดสรรทรัพยากรห้องฉุกเฉินเพิ่มเติม")
             insights_generated = True
        if 'risk_counts_fatal' in locals() and not risk_counts_fatal.empty:
             top_risk = risk_counts_fatal.index[0]
             st.info(f"📌 ปัจจัยเสี่ยงที่พบบ่อยสุดในเคส 'เสี่ยงมาก' คือ '{top_risk}': พิจารณารณรงค์หรือออกมาตรการป้องกันที่เกี่ยวข้องกับปัจจัยนี้")
             insights_generated = True
        if not insights_generated and total_cases > 5: # Only show if some data exists
             st.info("ข้อมูล Log ยังไม่แสดงแนวโน้มที่ชัดเจนสำหรับข้อเสนอแนะอัตโนมัติ")

    else: # If df_log is None or empty
        st.info("เริ่มทำการประเมินในแท็บ 'Clinical Risk Prediction' เพื่อดูข้อมูลสรุปที่นี่")
        if log_load_error:
            st.error(f"การโหลดข้อมูล Log ล้มเหลว: {log_load_error}")

st.markdown("---")
st.markdown("Developed by AI for Road Safety | Data Source: Injury Surveillance (IS) - MOPH")
"
I'm not asking for any changes, I'm just updating the file.

