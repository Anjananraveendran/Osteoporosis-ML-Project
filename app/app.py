"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Osteoporosis Risk Assessor",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Asset Loader Function (Handles both dict and direct model)
# ---------------------------------------------------------
import os
import joblib
import streamlit as st

@st.cache_resource
def load_pipeline():
    # Dynamically locate the directory where app.py resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path= "models/gradient_boosting_osteoporosis_model.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"Missing file at path: {model_path}")
        return None, None
        
    try:
        loaded_object = joblib.load(model_path)
        if isinstance(loaded_object, dict):
            return loaded_object.get('model'), loaded_object.get('feature_names')
        return loaded_object, getattr(loaded_object, 'feature_names_in_', None)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None, None

model, expected_features = load_pipeline()

# ---------------------------------------------------------
# 3. Header & Sidebar UI
# ---------------------------------------------------------
st.title("🦴 Osteoporosis Risk Prediction System")
st.markdown(""" """
This clinical decision-support tool utilizes a **Gradient Boosting Classifier** trained on patient demographic and medical profile data to estimate osteoporosis risk.
""" """)

st.sidebar.header("📋 Patient Clinical Profile")

if model is not None:
    # Input controls
    age = st.sidebar.slider("Age", min_value=18, max_value=95, value=50, step=1)
    gender = st.sidebar.selectbox("Gender", options=["Female", "Male"])
    hormonal_changes = st.sidebar.selectbox("Hormonal Changes", options=["Normal", "Postmenopausal"])
    family_history = st.sidebar.selectbox("Family History of Osteoporosis", options=["No", "Yes"])
    race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", options=["Caucasian", "Asian", "African American"])
    body_weight = st.sidebar.selectbox("Body Weight", options=["Normal", "Underweight"])
    calcium_intake = st.sidebar.selectbox("Calcium Intake", options=["Low", "Adequate"])
    vitamin_d_intake = st.sidebar.selectbox("Vitamin D Intake", options=["Sufficient", "Insufficient"])
    physical_activity = st.sidebar.selectbox("Physical Activity", options=["Active", "Sedentary"])
    smoking = st.sidebar.selectbox("Smoking Status", options=["No", "Yes"])
    alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption", options=["None", "Moderate"])
    medical_conditions = st.sidebar.selectbox("Medical Conditions", options=["None", "Hyperthyroidism", "Rheumatoid Arthritis"])
    medications = st.sidebar.selectbox("Medications", options=["None", "Corticosteroids"])
    prior_fractures = st.sidebar.selectbox("Prior Fractures", options=["No", "Yes"])

    raw_input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Hormonal Changes': hormonal_changes,
        'Family History': family_history,
        'Race/Ethnicity': race_ethnicity,
        'Body Weight': body_weight,
        'Calcium Intake': calcium_intake,
        'Vitamin D Intake': vitamin_d_intake,
        'Physical Activity': physical_activity,
        'Smoking': smoking,
        'Alcohol Consumption': alcohol_consumption,
        'Medical Conditions': medical_conditions,
        'Medications': medications,
        'Prior Fractures': prior_fractures
    }])

    # Preprocessing
    def preprocess_input(raw_df, feature_list):
        encoded_df = pd.get_dummies(raw_df)
        if feature_list is not None:
            return encoded_df.reindex(columns=feature_list, fill_value=0)
        return encoded_df

    processed_input = preprocess_input(raw_input_data, expected_features)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Selected Patient Parameters")
        st.dataframe(raw_input_data.T.rename(columns={0: "Value"}), use_container_width=True)

    with col2:
        st.subheader("Prediction Analysis")
        
        # FIX: Robust 2D Indexing [0, 1] for predict_proba
        probabilities = model.predict_proba(processed_input)
        risk_probability = float(probabilities[0, 1])
        
        threshold = st.slider("Clinical Decision Threshold", min_value=0.20, max_value=0.80, value=0.50, step=0.05)
        predicted_class = 1 if risk_probability >= threshold else 0

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_probability * 100,
            number={'suffix': "%"},
            title={'text': "Osteoporosis Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2C3E50"},
                'steps': [
                    {'range': [0, threshold * 100], 'color': "#2ECC71"},
                    {'range': [threshold * 100, 100], 'color': "#E74C3C"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if predicted_class == 1:
            st.error(f"⚠️ **HIGH RISK DETECTED** (Probability: {risk_probability:.1%})")
            st.warning("Recommendation: Diagnostic Bone Mineral Density (BMD) testing advised.")
        else:
            st.success(f"✅ **LOW RISK DETECTED** (Probability: {risk_probability:.1%})")
            st.info("Recommendation: Maintain routine health monitoring.")

else:
    st.error("Model assets file (`osteoporosis_gb_pipeline.pkl`) not found or could not be loaded. Ensure the file exists in the same directory.")

    """
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Osteoporosis Risk Assessor",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. FIND PROJECT DIRECTORIES
# =========================================================

# app.py is inside:
# Osteoporosis-ML-Project/app/app.py
#
# Models are expected inside:
# Osteoporosis-ML-Project/models/

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "gradient_boosting_osteoporosis_model.pkl"
)

SCALER_PATH = os.path.join(
    MODELS_DIR,
    "scaler.pkl"
)


# =========================================================
# 3. LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_assets():

    model = None
    scaler = None
    error_messages = []

    # -----------------------------
    # Load Gradient Boosting model
    # -----------------------------
    if not os.path.exists(MODEL_PATH):
        error_messages.append(
            f"Model not found:\n{MODEL_PATH}"
        )
    else:
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            error_messages.append(
                f"Could not load model:\n{e}"
            )

    # -----------------------------
    # Load Age scaler
    # -----------------------------
    if not os.path.exists(SCALER_PATH):
        error_messages.append(
            f"Scaler not found:\n{SCALER_PATH}"
        )
    else:
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            error_messages.append(
                f"Could not load scaler:\n{e}"
            )

    return model, scaler, error_messages


model, scaler, load_errors = load_assets()


# =========================================================
# 4. HEADER
# =========================================================

st.title("🦴 Osteoporosis Risk Prediction System")

st.markdown(
    """
    This application uses a **Gradient Boosting Classifier** trained on
    demographic and medical-profile data to estimate osteoporosis risk.

    **Important:** This is an ML-based risk prediction and is not a
    medical diagnosis.
    """
)


# =========================================================
# 5. SHOW LOADING ERRORS
# =========================================================

if load_errors:

    st.error("There was a problem loading the model assets.")

    for error in load_errors:
        st.code(error)

    st.stop()


# =========================================================
# 6. SIDEBAR - PATIENT INPUT
# =========================================================

st.sidebar.header("📋 Patient Clinical Profile")


age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=95,
    value=50,
    step=1
)


gender = st.sidebar.selectbox(
    "Gender",
    options=["Female", "Male"]
)


hormonal_changes = st.sidebar.selectbox(
    "Hormonal Changes",
    options=["Normal", "Postmenopausal"]
)


family_history = st.sidebar.selectbox(
    "Family History of Osteoporosis",
    options=["No", "Yes"]
)


race_ethnicity = st.sidebar.selectbox(
    "Race/Ethnicity",
    options=[
        "African American",
        "Asian",
        "Caucasian"
    ]
)


body_weight = st.sidebar.selectbox(
    "Body Weight",
    options=[
        "Normal",
        "Underweight"
    ]
)


calcium_intake = st.sidebar.selectbox(
    "Calcium Intake",
    options=[
        "Low",
        "Adequate"
    ]
)


vitamin_d_intake = st.sidebar.selectbox(
    "Vitamin D Intake",
    options=[
        "Insufficient",
        "Sufficient"
    ]
)


physical_activity = st.sidebar.selectbox(
    "Physical Activity",
    options=[
        "Sedentary",
        "Active"
    ]
)


smoking = st.sidebar.selectbox(
    "Smoking Status",
    options=[
        "No",
        "Yes"
    ]
)


# The notebook maps Moderate -> 1 and Unknown -> 0.
# Keep the same terminology used during training.
alcohol_consumption = st.sidebar.selectbox(
    "Alcohol Consumption",
    options=[
        "Unknown",
        "Moderate"
    ]
)


medical_conditions = st.sidebar.selectbox(
    "Medical Conditions",
    options=[
        "None",
        "Hyperthyroidism",
        "Rheumatoid Arthritis",
        "Unknown"
    ]
)


# The notebook maps Corticosteroids -> 1 and Unknown -> 0.
medications = st.sidebar.selectbox(
    "Medications",
    options=[
        "Unknown",
        "Corticosteroids"
    ]
)


prior_fractures = st.sidebar.selectbox(
    "Prior Fractures",
    options=[
        "No",
        "Yes"
    ]
)


# =========================================================
# 7. RAW PATIENT DATA
# =========================================================

raw_input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Hormonal Changes": hormonal_changes,
    "Family History": family_history,
    "Race/Ethnicity": race_ethnicity,
    "Body Weight": body_weight,
    "Calcium Intake": calcium_intake,
    "Vitamin D Intake": vitamin_d_intake,
    "Physical Activity": physical_activity,
    "Smoking": smoking,
    "Alcohol Consumption": alcohol_consumption,
    "Medical Conditions": medical_conditions,
    "Medications": medications,
    "Prior Fractures": prior_fractures
}])


# =========================================================
# 8. PREPROCESSING
#
# IMPORTANT:
# This reproduces the preprocessing from the notebook.
# =========================================================

def preprocess_input(raw_df, model, scaler):

    data = raw_df.copy()

    # -----------------------------------------------------
    # A. Manual binary mappings
    #
    # Same mappings used in the notebook
    # -----------------------------------------------------

    binary_mappings = {

        "Gender": {
            "Female": 1,
            "Male": 0
        },

        "Hormonal Changes": {
            "Postmenopausal": 1,
            "Normal": 0
        },

        "Family History": {
            "Yes": 1,
            "No": 0
        },

        "Body Weight": {
            "Underweight": 1,
            "Normal": 0
        },

        "Calcium Intake": {
            "Adequate": 1,
            "Low": 0
        },

        "Vitamin D Intake": {
            "Sufficient": 1,
            "Insufficient": 0
        },

        "Physical Activity": {
            "Active": 1,
            "Sedentary": 0
        },

        "Smoking": {
            "Yes": 1,
            "No": 0
        },

        "Alcohol Consumption": {
            "Moderate": 1,
            "Unknown": 0
        },

        "Medications": {
            "Corticosteroids": 1,
            "Unknown": 0
        },

        "Prior Fractures": {
            "Yes": 1,
            "No": 0
        }
    }


    # Apply binary mappings
    for column, mapping in binary_mappings.items():

        if column in data.columns:

            data[column] = data[column].map(mapping)


    # -----------------------------------------------------
    # B. One-hot encoding for the two multi-class columns
    #
    # Training notebook used:
    #
    # pd.get_dummies(
    #     df_encoded,
    #     columns=['Race/Ethnicity', 'Medical Conditions'],
    #     drop_first=True,
    #     dtype=int
    # )
    #
    # We don't call get_dummies directly on the single
    # patient row because that can choose a different
    # "first" category for each individual patient.
    #
    # Instead, we create the exact columns expected by
    # the trained model.
    # -----------------------------------------------------

    expected_features = list(model.feature_names_in_)


    # Start with every expected feature set to 0
    processed = pd.DataFrame(
        0,
        index=[0],
        columns=expected_features
    )


    # -----------------------------------------------------
    # C. Insert Age
    #
    # The notebook standardized Age using StandardScaler
    # before training the models.
    # -----------------------------------------------------

    scaled_age = scaler.transform(
        pd.DataFrame({"Age": [age]})
    )[0, 0]

    if "Age" in processed.columns:
        processed.loc[0, "Age"] = scaled_age


    # -----------------------------------------------------
    # D. Insert binary features
    # -----------------------------------------------------

    for column in binary_mappings.keys():

        if column in processed.columns:
            processed.loc[0, column] = data.loc[0, column]


    # -----------------------------------------------------
    # E. Race/Ethnicity one-hot features
    # -----------------------------------------------------

    race = raw_df.loc[0, "Race/Ethnicity"]

    for feature in expected_features:

        prefix = "Race/Ethnicity_"

        if feature.startswith(prefix):

            category = feature[len(prefix):]

            if race == category:
                processed.loc[0, feature] = 1


    # -----------------------------------------------------
    # F. Medical Conditions one-hot features
    # -----------------------------------------------------

    medical_condition = raw_df.loc[0, "Medical Conditions"]

    for feature in expected_features:

        prefix = "Medical Conditions_"

        if feature.startswith(prefix):

            category = feature[len(prefix):]

            if medical_condition == category:
                processed.loc[0, feature] = 1


    # -----------------------------------------------------
    # G. Ensure exact feature order
    # -----------------------------------------------------

    processed = processed[expected_features]


    # Convert everything to numeric
    processed = processed.apply(
        pd.to_numeric,
        errors="coerce"
    )


    return processed


# =========================================================
# 9. PREPARE MODEL INPUT
# =========================================================

try:

    processed_input = preprocess_input(
        raw_input_data,
        model,
        scaler
    )

except Exception as e:

    st.error(
        f"Error while preprocessing patient data: {e}"
    )

    st.stop()


# =========================================================
# 10. DISPLAY PATIENT INFORMATION
# =========================================================

col1, col2 = st.columns([1, 1])


with col1:

    st.subheader("Selected Patient Parameters")

    display_data = raw_input_data.T.rename(
        columns={0: "Value"}
    )

    st.dataframe(
        display_data,
        use_container_width=True
    )


# =========================================================
# 11. PREDICTION
# =========================================================

with col2:

    st.subheader("Prediction Analysis")

    try:

        # Probability for class 0 and class 1
        probabilities = model.predict_proba(
            processed_input
        )

        # Find the column corresponding to class 1
        class_1_index = list(model.classes_).index(1)

        risk_probability = float(
            probabilities[0, class_1_index]
        )


        # -------------------------------------------------
        # Clinical decision threshold
        # -------------------------------------------------

        threshold = st.slider(
            "Clinical Decision Threshold",
            min_value=0.20,
            max_value=0.80,
            value=0.50,
            step=0.05
        )


        predicted_class = (
            1
            if risk_probability >= threshold
            else 0
        )


        # -------------------------------------------------
        # Gauge
        # -------------------------------------------------

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=risk_probability * 100,

                number={
                    "suffix": "%"
                },

                title={
                    "text": "Predicted Osteoporosis Risk"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#2C3E50"
                    },

                    "steps": [

                        {
                            "range": [
                                0,
                                threshold * 100
                            ],

                            "color": "#2ECC71"
                        },

                        {
                            "range": [
                                threshold * 100,
                                100
                            ],

                            "color": "#E74C3C"
                        }
                    ],

                    "threshold": {

                        "line": {
                            "color": "black",
                            "width": 4
                        },

                        "thickness": 0.75,

                        "value": threshold * 100
                    }
                }
            )
        )


        fig.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        if predicted_class == 1:

            st.error(
                f"⚠️ HIGH PREDICTED RISK "
                f"({risk_probability:.1%})"
            )

            st.warning(
                "The ML model predicts a higher likelihood "
                "of osteoporosis. This is not a clinical "
                "diagnosis."
            )

            st.info(
                "Recommendation: Consider confirmatory "
                "clinical assessment such as BMD/DXA "
                "testing according to medical guidance."
            )

        else:

            st.success(
                f"✅ LOW PREDICTED RISK "
                f"({risk_probability:.1%})"
            )

            st.info(
                "The ML model predicts a lower likelihood "
                "of osteoporosis. This does not rule out "
                "osteoporosis or other bone-health problems."
            )


        # -------------------------------------------------
        # Model class information
        # -------------------------------------------------

        st.caption(
            f"Decision threshold: {threshold:.2f} | "
            f"Class 0 = negative, Class 1 = positive"
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        st.stop()


# =========================================================
# 12. DEBUG INFORMATION
# =========================================================

with st.expander("🔍 View Model Input"):

    st.write(
        "The following values are the exact features "
        "sent to the trained Gradient Boosting model."
    )

    st.dataframe(
        processed_input.T.rename(
            columns={0: "Model Value"}
        ),
        use_container_width=True
    )

    st.write(
        "Model feature order:"
    )

    st.write(
        list(model.feature_names_in_)
    )