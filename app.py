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
            loaded_items[key] = None # Default for failed load
            if key == 'features': loaded_items[key] = [] # Specific default for features
            if key == 'encoders': loaded_items[key] = {} # Specific default for encoders
            msgs.append(f"❌ File not found: {os.path.basename(path)} — {description}")
        except Exception as e:
            loaded_items[key] = None # Default for failed load
            if key == 'features': loaded_items[key] = [] # Specific default for features
            if key == 'encoders': loaded_items[key] = {} # Specific default for encoders
            msgs.append(f"⚠️ Error loading {os.path.basename(path)}: {e}")

    # Display loading status
    with st.expander("📂 File Loading Status", expanded=False):
        # --- FIXED CHECK: Check for None, empty list/dict appropriately ---
        all_loaded_check = []
        for key, item in loaded_items.items():
            if key == 'features':
                all_loaded_check.append(isinstance(item, list) and len(item) > 0)
            elif key == 'encoders':
                 # Encoders can be an empty dict if no categorical features were encoded originally
                 all_loaded_check.append(isinstance(item, dict)) # Check if it's a dict (even empty)
            else:
                all_loaded_check.append(item is not None)

        all_loaded = all(all_loaded_check)
        # --- END OF FIX ---

        if all_loaded:
            st.success("All required files loaded successfully.")
        else:
            st.warning("Some files could not be loaded or are empty. Please check file paths and availability.")

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
# Updated check for encoders (can be empty dict {})
essentials_loaded = model and features and encoders is not None

# ----------------------------------------------------------
# 🧩 Manual Mappings & Helper Functions
# ----------------------------------------------------------
activity_mapping = {
    "0": "เดินเท้า", "1": "โดยสารพาหนะสาธารณะ", "2": "โดยสารพาหนะส่วนบุคคล",
    "3": "ขับขี่พาหนะส่วนบุคคล", "4": "ทำงาน", "5": "เล่นกีฬา", "6": "กิจกรรมอื่น ๆ", "Unknown": "ไม่ระบุ"
}
aplace_mapping = {
    "10": "บ้านพักอาศัย", "11": "ถนน/ทางหลวง", "12": "สถานที่ทำงาน",
    "13": "โรงเรียน/สถาบันศึกษา", "14": "พื้นที่สาธารณะ", "15": "อื่น ๆ", "Unknown": "ไม่ระบุ"
}
prov_mapping = {
    "10": "กรุงเทพมหานคร", "20": "เชียงใหม่", "30": "ขอนแก่น", "40": "ภูเก็ต",
    "50": "นครราชสีมา", "60": "สงขลา", "99": "อื่น ๆ", "Unknown": "ไม่ระบุ"
}
# Add default mapping for unknown values generated during preprocessing fillna('Unknown')
activity_mapping.setdefault("Unknown", "ไม่ระบุ")
aplace_mapping.setdefault("Unknown", "ไม่ระบุ")
prov_mapping.setdefault("Unknown", "ไม่ระบุ")

severity_map = {0: "เสี่ยงน้อย", 1: "เสี่ยงปานกลาง", 2: "เสี่ยงมาก"}
advice_map = {
    "เสี่ยงน้อย": "ดูแลอาการทั่วไป เฝ้าระวังซ้ำทุก 15–30 นาที",
    "เสี่ยงปานกลาง": "ส่งตรวจเพิ่มเติม ให้สารน้ำ / ยาแก้ปวด / เฝ้าสัญญาณชีพใกล้ชิด",
    "เสี่ยงมาก": "แจ้งทีมสหสาขา เปิดทางเดินหายใจ เตรียมห้องฉุกเฉินหรือส่งต่อด่วน"
}
triage_color = {"เสี่ยงน้อย": "#4CAF50", "เสี่ยงปานกลาง": "#FFC107", "เสี่ยงมาก": "#F44336"}

# --- Preprocessing Function (Modified to use loaded features/encoders) ---
#@st.cache_data # Cache this function if inputs are likely to repeat
def preprocess_input(data_dict, loaded_encoders, expected_features):
    """Preprocesses user input dictionary based on loaded encoders and features."""
    if not expected_features:
        st.error("Feature list is empty. Cannot preprocess.")
        return None
    if loaded_encoders is None:
         st.warning("Encoders not loaded. Categorical features might not be processed correctly.")
         loaded_encoders = {} # Use an empty dict to avoid errors


    df = pd.DataFrame([data_dict])

    # Map display values back to codes before encoding
    reverse_activity = {v: k for k, v in activity_mapping.items()}
    reverse_aplace = {v: k for k, v in aplace_mapping.items()}
    reverse_prov = {v: k for k, v in prov_mapping.items()}

    # Use .get() with a default 'Unknown' code if the display value isn't found
    df['activity'] = df['activity'].apply(lambda x: reverse_activity.get(x, "Unknown"))
    df['aplace'] = df['aplace'].apply(lambda x: reverse_aplace.get(x, "Unknown"))
    df['prov'] = df['prov'].apply(lambda x: reverse_prov.get(x, "Unknown"))

    # Reindex first to ensure all expected columns are present
    df = df.reindex(columns=expected_features, fill_value=0) # Use 0 as default fill

    # Clean potential string issues and fill NaNs *after* reindexing
    # Important: Ensure 'Unknown' is handled consistently with how missing/cleaned values were treated before encoding
    clean_values = ["None", "N/A", "ไม่ระบุ", "N", "nan", "NaN", "", " "]
    df = df.replace(clean_values, "Unknown") # Replace various missing representations with 'Unknown'
    df = df.fillna("Unknown") # Fill any actual NaNs with 'Unknown' string for categorical, 0 for potential numerics later


    # Encode categorical features using loaded encoders
    for col, le in loaded_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str) # Ensure string type before transform
            # Handle unseen labels by mapping them to the 'Unknown' class index if it exists, otherwise default to 0
            unknown_class_index = 0 # Default index if 'Unknown' not in classes
            if "Unknown" in le.classes_:
                try:
                  unknown_class_index = le.transform(["Unknown"])[0]
                except ValueError:
                  pass # Should not happen if 'Unknown' is in classes_ but handle defensively

            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else unknown_class_index)


    # --- Create Engineered Features (must match training - check feature list) ---
    # Ensure necessary base columns exist before creating engineered ones
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0) # Convert age, fill errors with 0
        if "age_group_60plus" in expected_features:
             df["age_group_60plus"] = (df["age"] >= 60).astype(int)

    risk_cols_in_features = [f"risk{i}" for i in range(1, 6) if f"risk{i}" in expected_features]
    if "risk_count" in expected_features:
        # Convert risk columns to numeric before summing, coercing errors to 0
        for r_col in risk_cols_in_features:
            if r_col in df.columns:
                 df[r_col] = pd.to_numeric(df[r_col], errors='coerce').fillna(0)
            else:
                 df[r_col] = 0 # Ensure column exists if expected but missing in input_dict

        if risk_cols_in_features: # Only sum if relevant risk columns exist
             df["risk_count"] = df[risk_cols_in_features].sum(axis=1)
        else:
             df["risk_count"] = 0 # If risk_count expected but base risks aren't features


    if 'is_night' in df.columns:
        df['is_night'] = pd.to_numeric(df['is_night'], errors='coerce').fillna(0) # Convert, fill errors with 0
        if "night_flag" in expected_features:
            df["night_flag"] = df["is_night"].astype(int)
    elif "night_flag" in expected_features: # If night_flag expected but is_night isn't
         df["night_flag"] = 0


    # Ensure all columns are numeric at the end and fill any remaining NaNs (e.g., from failed numeric conversions)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    # Final check: Ensure column order matches exactly and return
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


# --- Function to decode Apriori rules ---
#@st.cache_data # Cache decoding if rules don't change often
def decode_set(item_set):
    """Converts a frozenset of coded items to readable strings."""
    if not isinstance(item_set, (frozenset, set)):
        return str(item_set)

    # Define mappings (expand as needed based on your Apriori basket items from notebook)
    replacements = {
        "Male": "เพศชาย", "Female": "เพศหญิง", "Age_60+": "อายุ 60+",
        "Nighttime": "กลางคืน", "Head_Injury": "บาดเจ็บศีรษะ",
        "Risk_Alcohol": "ดื่มแอลกอฮอล์", "Risk_Drugs_General": "ใช้ยาเสพติด(ทั่วไป)",
        "Risk_No_Belt": "ไม่คาดเข็มขัด", "Risk_No_Helmet": "ไม่สวมหมวก",
        "Risk_Phone_Use": "ใช้โทรศัพท์ขณะขับ",
        "Drug_Kratom": "กระท่อม", "Drug_Cannabis": "กัญชา", "Drug_Amphetamine": "ยาบ้า",
        "Severity_Minor": "บาดเจ็บเล็กน้อย", "Severity_Severe": "บาดเจ็บรุนแรง",
        "Severity_Fatal": "เสียชีวิต"
        # Add more mappings corresponding to df_basket column names in the notebook
    }
    # Sort for consistency before joining
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
        st.error("Cannot proceed with prediction. Essential model files (model, features, encoders) are missing or empty.")
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
            with c1: risk1 = st.checkbox("ดื่มแอลกอฮอล์ (risk1)", key="tab1_risk1") # Matches notebook variable name
            with c2: risk2 = st.checkbox("ใช้ยาเสพติด(ทั่วไป) (risk2)", key="tab1_risk2") # Matches notebook variable name
            with c3: risk3 = st.checkbox("ไม่คาดเข็มขัด (risk3)", key="tab1_risk3") # Matches notebook variable name
            with c4: risk4 = st.checkbox("ไม่สวมหมวกนิรภัย (risk4)", key="tab1_risk4") # Matches notebook variable name
            with c5: risk5 = st.checkbox("ใช้โทรศัพท์ขณะขับ (risk5)", key="tab1_risk5") # Matches notebook variable name


            st.markdown("**สารเสพติด/ยาในร่างกาย (เลือกหากเกี่ยวข้อง)**")
            d1, d2, d3 = st.columns(3)
            with d1: cannabis = st.checkbox("กัญชา", key="tab1_cannabis")
            with d2: amphetamine = st.checkbox("ยาบ้า / แอมเฟตามีน", key="tab1_amphetamine")
            with d3: drugs = st.checkbox("ยาอื่น ๆ (drugs)", key="tab1_drugs") # 'drugs' from notebook


            st.markdown("**บริบทของเหตุการณ์**")
            # Get display values from selectbox
            activity_display = st.selectbox("กิจกรรมขณะเกิดเหตุ", list(activity_mapping.values()), key="tab1_activity_display", index=list(activity_mapping.values()).index("ขับขี่พาหนะส่วนบุคคล")) # Default example
            aplace_display = st.selectbox("สถานที่เกิดเหตุ", list(aplace_mapping.values()), key="tab1_aplace_display", index=list(aplace_mapping.values()).index("ถนน/ทางหลวง")) # Default example
            prov_display = st.selectbox("จังหวัดที่เกิดเหตุ", list(prov_mapping.values()), key="tab1_prov_display", index=list(prov_mapping.values()).index("กรุงเทพมหานคร")) # Default example


            submit_button_tab1 = st.form_submit_button("🔎 ประเมินระดับความเสี่ยง")

        if submit_button_tab1:
            st.session_state.submit_pressed = True # Mark that form was submitted

            # --- Prepare input dictionary using display values for categorical ---
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
                "activity": activity_display, # Pass display value to preprocess
                "aplace": aplace_display,      # Pass display value to preprocess
                "prov": prov_display          # Pass display value to preprocess
            }
             # Store raw input for other tabs (using display values)
            st.session_state.raw_input_for_tabs = {
                "age": age, "sex": sex, "is_night": is_night, "head_injury": head_injury,
                "mass_casualty": mass_casualty, "risk1": risk1, "risk2": risk2,
                "risk3": risk3, "risk4": risk4, "risk5": risk5, "cannabis": cannabis,
                "amphetamine": amphetamine, "drugs": drugs, "activity": activity_display,
                "aplace": aplace_display, "prov": prov_display
            }


            # --- Preprocess and Predict ---
            # Make sure 'features' list is loaded correctly
            if not features:
                 st.error("Feature list could not be loaded. Cannot preprocess input.")
            else:
                X_input = preprocess_input(input_data, encoders, features)
                st.session_state.processed_input_for_tabs = X_input # Store processed input for other tabs


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
                                "timestamp": pd.Timestamp.now(tz='Asia/Bangkok').strftime('%Y-%m-%d %H:%M:%S %Z'), # Format timestamp
                                "age": age,
                                "sex": sex, # Log display value 'ชาย'/'หญิง'
                                "predicted_severity": st.session_state.prediction_label,
                                "prov": prov_display, # Log display value
                                "is_night": int(is_night),
                                "risk1": int(risk1),
                                "risk2": int(risk2),
                                "risk3": int(risk3),
                                "risk4": int(risk4),
                                "risk5": int(risk5),
                                "head_injury": int(head_injury) # Add head injury to log
                            }
                            new_row = pd.DataFrame([log_entry])

                            # Ensure header exists or write it
                            write_header = not os.path.exists(LOG_FILE)
                            new_row.to_csv(LOG_FILE, mode="a", index=False, header=write_header, encoding='utf-8-sig')

                            st.success("📁 บันทึกผลการประเมินเข้าสู่ระบบ Dashboard แล้ว")
                        except Exception as log_e:
                            st.warning(f"Could not save prediction log: {log_e}")

                    except AttributeError as ae:
                         st.error(f"Prediction error: Model object might not be loaded correctly or is not a CatBoostClassifier. Details: {ae}")
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

    # Check if form was submitted in Tab 1
    if not st.session_state.get('submit_pressed', False): # Use .get() for safety
        st.info("กรุณากรอกข้อมูลและกด 'ประเมินระดับความเสี่ยง' ในแท็บแรกก่อน")
    elif kmeans is None or scaler is None:
        st.warning("⚠️ ไม่พบไฟล์โมเดล K-Means หรือ Scaler ที่โหลดไว้")
    elif st.session_state.get('processed_input_for_tabs') is None:
         st.warning("⚠️ ไม่สามารถประมวลผลข้อมูล Input สำหรับ Clustering ได้")
    else:
        # --- Display Patient Summary ---
        st.markdown("### 🧾 ข้อมูลผู้บาดเจ็บ")
        raw_input = st.session_state.get('raw_input_for_tabs', {}) # Use .get()
        summary_cols = st.columns(3)
        summary_cols[0].metric("อายุ", f"{raw_input.get('age', 'N/A')} ปี")
        summary_cols[1].metric("เพศ", raw_input.get('sex', 'N/A')) # Use stored display value
        summary_cols[2].metric("ระดับความเสี่ยง", st.session_state.get('prediction_label', 'N/A')) # Use .get()

        # Generate risk summary from stored raw boolean values
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

        # --- Perform Clustering ---
        try:
            # Prepare data specifically for clustering - using the *processed* input
            X_input_processed = st.session_state.processed_input_for_tabs

            # Identify features used by the scaler during training
            if hasattr(scaler, "feature_names_in_"):
                cluster_features_used = scaler.feature_names_in_
            else:
                 # Fallback: try to infer from scaler attributes if possible, else error
                 if hasattr(scaler, 'n_features_in_'):
                      if scaler.n_features_in_ == X_input_processed.shape[1]:
                           cluster_features_used = X_input_processed.columns.tolist() # Assume same features/order
                           st.warning("Scaler object lacks explicit feature names. Assuming features match processed input order.")
                      else:
                           st.error(f"Scaler expects {scaler.n_features_in_} features, but processed input has {X_input_processed.shape[1]}. Cannot proceed with clustering.")
                           cluster_features_used = None
                 else:
                    st.error("Cannot determine features expected by the scaler. Clustering aborted.")
                    cluster_features_used = None


            if cluster_features_used:
                # Ensure the processed input DataFrame has exactly these columns in the correct order
                X_cluster_input = X_input_processed.copy()
                missing_cols = set(cluster_features_used) - set(X_cluster_input.columns)
                if missing_cols:
                    st.warning(f"Processed input is missing columns expected by scaler: {missing_cols}. Filling with 0.")
                    for col in missing_cols:
                        X_cluster_input[col] = 0

                # Select and reorder columns
                try:
                    X_cluster_input = X_cluster_input[cluster_features_used]
                except KeyError as e:
                     st.error(f"Error selecting/ordering columns for scaler: {e}")
                     X_cluster_input = None # Prevent further processing


                if X_cluster_input is not None and not X_cluster_input.empty:
                    X_scaled = scaler.transform(X_cluster_input)
                    cluster_label = int(kmeans.predict(X_scaled)[0])

                    # Example Cluster Descriptions (Adjust based on your actual cluster analysis from notebook)
                    cluster_desc = {
                        0: "กลุ่มทั่วไป/ความเสี่ยงต่ำ: มักเป็นเพศชาย ไม่พบบาดเจ็บศีรษะหรือปัจจัยเสี่ยงเด่นชัด (Cluster 0 Profile)",
                        1: "กลุ่มความเสี่ยงปานกลาง/หญิง: มักเป็นเพศหญิง อาจไม่พบบาดเจ็บศีรษะ (Cluster 1 Profile)",
                        2: "กลุ่มความเสี่ยงสูง/ยาเสพติด: พบการใช้ยาเสพติด (risk2=1) สูง อาจมีบาดเจ็บศีรษะร่วม (Cluster 2 Profile)"
                        # Add descriptions reflecting the means from your notebook analysis
                    }

                    st.markdown(f"### 📊 ผลการจัดกลุ่ม: **Cluster {cluster_label}**")
                    st.info(f"**ลักษณะกลุ่ม (โดยประมาณ):** {cluster_desc.get(cluster_label, 'ยังไม่มีคำอธิบายสำหรับกลุ่มนี้')}")
                    st.caption("💡 ใช้ข้อมูลนี้เพื่อทำความเข้าใจลักษณะผู้ป่วยที่คล้ายกัน และวางแผนทรัพยากรหรือแคมเปญป้องกันที่ตรงกลุ่มเป้าหมาย")

                else:
                    st.error("Could not prepare valid input for clustering after feature selection.")
            # else: Error already shown about scaler features

        except AttributeError as ae:
             st.error(f"Clustering error: Model object (kmeans or scaler) might not be loaded correctly. Details: {ae}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการจัดกลุ่ม K-Means: {e}")
            # st.dataframe(X_input_processed) # Optional debugging


# ----------------------------------------------------------
# 🧩 TAB 3 — Apriori Risk Association
# ----------------------------------------------------------
with tab3:
    st.subheader("🧩 Risk Association Analysis (Apriori)")
    st.caption("วิเคราะห์ความสัมพันธ์ของปัจจัยเสี่ยง เพื่อวางแผนป้องกันและสนับสนุนการตัดสินใจเชิงนโยบาย")

    # Check if form was submitted in Tab 1
    if not st.session_state.get('submit_pressed', False):
        st.info("กรุณากรอกข้อมูลและกด 'ประเมินระดับความเสี่ยง' ในแท็บแรกก่อน")
    elif rules_minor is None and rules_severe is None and rules_fatal is None:
        st.warning("⚠️ ไม่พบไฟล์กฎ Apriori ที่โหลดไว้ หรือไฟล์ว่างเปล่า")
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
        if raw_input.get('risk4'): risk_tags.append("ไม่สวมหมวกนิรภัย")
        if raw_input.get('risk5'): risk_tags.append("ใช้โทรศัพท์ขณะขับ")
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

        # Check if the loaded rules object is a non-empty DataFrame
        is_valid_df = lambda df: isinstance(df, pd.DataFrame) and not df.empty

        if target_label == "เสี่ยงน้อย" and is_valid_df(rules_minor):
            rules_df_to_show = rules_minor
        elif target_label == "เสี่ยงปานกลาง" and is_valid_df(rules_severe):
            rules_df_to_show = rules_severe
        elif target_label == "เสี่ยงมาก" and is_valid_df(rules_fatal):
            rules_df_to_show = rules_fatal

        if rules_df_to_show is not None:
            st.markdown(f"**{rules_title}**")
            # Decode antecedents and consequents for display
            display_df = rules_df_to_show.head(5).copy() # Work with a copy

            # Check if columns exist before applying decode_set
            if 'antecedents' in display_df.columns:
                 display_df["ปัจจัยนำ (Antecedents)"] = display_df["antecedents"].apply(decode_set)
            else:
                 display_df["ปัจจัยนำ (Antecedents)"] = "N/A"

            if 'consequents' in display_df.columns:
                 display_df["ผลลัพธ์ (Consequents)"] = display_df["consequents"].apply(decode_set)
            else:
                 display_df["ผลลัพธ์ (Consequents)"] = "N/A"


            # Select and rename columns for clarity, handle missing columns gracefully
            cols_to_display = ["ปัจจัยนำ (Antecedents)", "ผลลัพธ์ (Consequents)"]
            rename_map = {"support": "Support", "confidence": "Confidence", "lift": "Lift"}
            for col, new_name in rename_map.items():
                if col in display_df.columns:
                    cols_to_display.append(new_name)
                    display_df = display_df.rename(columns={col: new_name})

            # Check if essential columns exist before displaying
            if "ปัจจัยนำ (Antecedents)" in display_df.columns and "ผลลัพธ์ (Consequents)" in display_df.columns:
                st.dataframe(
                    display_df[cols_to_display],
                    use_container_width=True,
                    hide_index=True
                )

                # Display interpretation guide
                st.markdown("📘 **การตีความ:**")
                st.markdown("- **Support:** สัดส่วนของเคสทั้งหมดที่พบทั้งปัจจัยนำและผลลัพธ์นี้ร่วมกัน")
                st.markdown("- **Confidence:** ความน่าจะเป็นที่จะเกิด 'ผลลัพธ์' เมื่อพบ 'ปัจจัยนำ' เหล่านี้")
                st.markdown("- **Lift > 1:** บ่งชี้ว่าการเกิดร่วมกันนี้มีนัยสำคัญ (ไม่ใช่เรื่องบังเอิญ) ยิ่งค่าสูงยิ่งสัมพันธ์กันมาก")
            else:
                 st.error("ไม่สามารถแสดงกฎ Apriori ได้ เนื่องจากคอลัมน์ที่จำเป็น (antecedents/consequents) หายไปจากไฟล์ .pkl")

        else:
            st.info(f"ไม่พบกฎ Apriori ที่เกี่ยวข้องสำหรับระดับความเสี่ยง '{target_label}' ในขณะนี้ หรือไฟล์กฎว่างเปล่า")


# ----------------------------------------------------------
# 📊 TAB 4 — Clinical Summary Dashboard
# ----------------------------------------------------------
with tab4:
    st.subheader("📊 Clinical Summary & Insights Dashboard")
    st.caption("สรุปแนวโน้มผู้บาดเจ็บจากระบบ AI เพื่อใช้วางแผนเชิงกลยุทธ์และบริหารทรัพยากรโรงพยาบาล")

    # --- Load Log File ---
    df_log = None
    log_load_error = None
    if os.path.exists(LOG_FILE):
        try:
            df_log = pd.read_csv(LOG_FILE, parse_dates=['timestamp'], encoding='utf-8-sig') # Specify encoding
            # Attempt to convert timestamp, handle potential errors
            df_log['timestamp'] = pd.to_datetime(df_log['timestamp'], errors='coerce').dt.tz_localize(None) # Remove timezone for consistency
            df_log = df_log.dropna(subset=['timestamp']) # Drop rows where timestamp conversion failed

            if not df_log.empty:
                 st.success(f"📁 โหลดข้อมูล Log สำเร็จ: {len(df_log):,} รายการ")
            else:
                 st.info("ไฟล์ Log ว่างเปล่า หรือไม่มีข้อมูลเวลาที่ถูกต้อง")
                 df_log = None # Treat empty/invalid log as if it doesn't exist
        except Exception as e:
            log_load_error = f"ไม่สามารถโหลดไฟล์ Log ({LOG_FILE}): {e}"
            st.error(log_load_error)
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

        fatal_ratio = df_log["predicted_severity"].eq("เสี่ยงมาก").mean() * 100 if "predicted_severity" in df_log.columns else 0
        avg_age = df_log["age"].mean() if 'age' in df_log.columns and pd.api.types.is_numeric_dtype(df_log['age']) else 'N/A'
        male_ratio = df_log["sex"].eq("ชาย").mean() * 100 if 'sex' in df_log.columns else 'N/A' # Assuming 'sex' logs 'ชาย'/'หญิง'

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
            if 'predicted_severity' in df_log.columns:
                severity_counts = df_log['predicted_severity'].value_counts()
                # Ensure correct order and colors
                severity_order = ["เสี่ยงน้อย", "เสี่ยงปานกลาง", "เสี่ยงมาก"]
                severity_counts = severity_counts.reindex(severity_order, fill_value=0)
                colors_ordered = [triage_color.get(level, "#CCCCCC") for level in severity_order]

                if not severity_counts.empty and severity_counts.sum() > 0: # Check sum > 0 for pie chart
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    # Use white text only if background colors are dark enough
                    textprops={'fontsize': 9} # Default props
                    ax1.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
                            startangle=90, colors=colors_ordered,
                            textprops=textprops, wedgeprops={'edgecolor': 'white'}) # Add edge color
                    ax1.axis('equal') # Equal aspect ratio
                    # ax1.set_title("Severity Distribution", fontsize=11) # Optional title
                    st.pyplot(fig1)
                else:
                    st.caption("ไม่มีข้อมูลความรุนแรง")
            else:
                st.caption("ไม่มีข้อมูลคอลัมน์ 'predicted_severity'")


        # 2. Cases over Time (Line Chart) - Requires 'timestamp'
        with chart_col2:
            st.markdown("##### 📈 จำนวนเคสตามช่วงเวลา")
            if 'timestamp' in df_log.columns:
                 df_log['date'] = df_log['timestamp'].dt.date
                 cases_over_time = df_log.groupby('date').size()
                 if not cases_over_time.empty:
                      st.line_chart(cases_over_time)
                 else:
                      st.caption("ไม่มีข้อมูลเคสในช่วงเวลาที่บันทึก")
            else:
                 st.caption("ไม่มีข้อมูลเวลา (timestamp) ใน Log")


        # --- Risk Factor Analysis ---
        st.markdown("---")
        st.markdown("### ❗ ปัจจัยเสี่ยงที่พบบ่อยในเคส 'เสี่ยงมาก'")

        # Define which columns represent risks in the log file
        risk_cols_in_log = [f"risk{i}" for i in range(1, 6) if f"risk{i}" in df_log.columns]
        if 'head_injury' in df_log.columns: risk_cols_in_log.append('head_injury')
        if 'is_night' in df_log.columns: risk_cols_in_log.append('is_night')
        # Add others like cannabis, amphetamine if they are logged as 0/1
        if 'cannabis' in df_log.columns: risk_cols_in_log.append('cannabis')
        if 'amphetamine' in df_log.columns: risk_cols_in_log.append('amphetamine')
        if 'drugs' in df_log.columns: risk_cols_in_log.append('drugs')


        if "predicted_severity" in df_log.columns:
            fatal_cases = df_log[df_log["predicted_severity"] == "เสี่ยงมาก"]
            if not fatal_cases.empty and risk_cols_in_log:
                # Ensure risk columns are numeric (convert bools/objects if necessary)
                for r_col in risk_cols_in_log:
                    fatal_cases[r_col] = pd.to_numeric(fatal_cases[r_col], errors='coerce').fillna(0)

                # Sum only the numeric risk columns that are truly binary (0 or 1)
                binary_risk_cols = [col for col in risk_cols_in_log if fatal_cases[col].isin([0, 1]).all()]
                if binary_risk_cols:
                    risk_counts_fatal = fatal_cases[binary_risk_cols].sum().sort_values(ascending=False)
                    # Filter out risks with zero counts
                    risk_counts_fatal = risk_counts_fatal[risk_counts_fatal > 0]

                    # Map coded risk names to readable names
                    risk_display_names = {
                        "risk1": "ดื่มแอลกอฮอล์", "risk2": "ใช้ยาเสพติด", "risk3": "ไม่คาดเข็มขัด",
                        "risk4": "ไม่สวมหมวก", "risk5": "ใช้โทรศัพท์",
                        "head_injury": "บาดเจ็บศีรษะ", "is_night": "เกิดกลางคืน",
                        "cannabis": "กัญชา", "amphetamine": "ยาบ้า", "drugs": "ยาอื่น ๆ"
                    }
                    risk_counts_fatal.index = risk_counts_fatal.index.map(risk_display_names).fillna(risk_counts_fatal.index)

                    if not risk_counts_fatal.empty:
                        fig3, ax3 = plt.subplots(figsize=(7, max(3, len(risk_counts_fatal)*0.4))) # Adjust height
                        sns.barplot(x=risk_counts_fatal.values, y=risk_counts_fatal.index, ax=ax3, palette="viridis")
                        ax3.set_xlabel("จำนวนเคส 'เสี่ยงมาก' ที่พบปัจจัยนี้")
                        ax3.set_ylabel("ปัจจัยเสี่ยง")
                        plt.tight_layout()
                        st.pyplot(fig3)
                    else:
                        st.caption("ไม่พบปัจจัยเสี่ยงที่ระบุในเคสเสี่ยงมาก หรือปัจจัยเสี่ยงไม่ใช่ค่า 0/1")
                else:
                     st.caption("ไม่พบคอลัมน์ปัจจัยเสี่ยงที่เป็นค่า 0/1 ในข้อมูล Log")

            else:
                st.caption("ไม่มีเคส 'เสี่ยงมาก' หรือไม่มีข้อมูลปัจจัยเสี่ยงใน Log")
        else:
             st.caption("ไม่มีข้อมูลคอลัมน์ 'predicted_severity' ใน Log")


         # --- Insights & Recommendations ---
        st.markdown("---")
        st.markdown("### 💡 ข้อเสนอแนะเชิงกลยุทธ์")
        # Example insights based on data
        insights_generated = False
        if fatal_ratio > 10: # Example threshold
             st.warning("🚨 สัดส่วนเคส 'เสี่ยงมาก' ค่อนข้างสูง: ควรทบทวนเกณฑ์ Triage หรือจัดสรรทรัพยากรห้องฉุกเฉินเพิ่มเติม")
             insights_generated = True
        if 'risk_counts_fatal' in locals() and not risk_counts_fatal.empty:
             top_risk = risk_counts_fatal.index[0]
             st.info(f"📌 ปัจจัยเสี่ยงที่พบบ่อยสุดในเคส 'เสี่ยงมาก' คือ '{top_risk}': พิจารณารณรงค์หรือออกมาตรการป้องกันที่เกี่ยวข้องกับปัจจัยนี้")
             insights_generated = True

        if not insights_generated:
             st.info("ข้อมูล Log ยังไม่แสดงแนวโน้มที่ชัดเจนสำหรับข้อเสนอแนะอัตโนมัติ")


    else: # If df_log is None or empty or failed to load
        st.info("เริ่มทำการประเมินในแท็บ 'Clinical Risk Prediction' เพื่อดูข้อมูลสรุปที่นี่")
        if log_load_error:
            st.error(f"การโหลดข้อมูล Log ล้มเหลว: {log_load_error}")


st.markdown("---")
st.markdown("Developed by AI for Road Safety | Data Source: Injury Surveillance (IS) - MOPH")
