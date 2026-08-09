import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)
# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

h1 {
    text-align: center;
    font-size: 42px;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.info-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model Files
# --------------------------------------------------

model = joblib.load("models/student_performance_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<h1>🎓 Student Performance Prediction</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'An AI-powered system for predicting student academic performance '
    'using machine learning.'
    '</p>',
    unsafe_allow_html=True
)

st.info(
    "📌 Enter the student's academic, personal, and educational "
    "information below to generate a performance prediction."
)

st.divider()


# --------------------------------------------------
# Student Information
# --------------------------------------------------

st.subheader("📚 Academic Information")

col1, col2, col3 = st.columns(3)

with col1:
    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0,
        max_value=50,
        value=20
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80
    )

with col3:
    previous_scores = st.number_input(
        "Previous Scores",
        min_value=0,
        max_value=100,
        value=70
    )


col1, col2, col3 = st.columns(3)

with col1:
    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0,
        max_value=24,
        value=7
    )

with col2:
    tutoring_sessions = st.number_input(
        "Tutoring Sessions",
        min_value=0,
        max_value=20,
        value=2
    )

with col3:
    physical_activity = st.number_input(
        "Physical Activity (hours)",
        min_value=0,
        max_value=20,
        value=3
    )


st.subheader("🏠 Student Background")

col1, col2, col3 = st.columns(3)

with col1:
    parental_involvement = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"]
    )

with col2:
    access_to_resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"]
    )

with col3:
    extracurricular_activities = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    motivation_level = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"]
    )

with col2:
    internet_access = st.selectbox(
        "Internet Access",
        ["No", "Yes"]
    )

with col3:
    family_income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )


st.subheader("🏫 School Information")

col1, col2, col3 = st.columns(3)

with col1:
    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"]
    )

with col2:
    school_type = st.selectbox(
        "School Type",
        ["Public", "Private"]
    )

with col3:
    peer_influence = st.selectbox(
        "Peer Influence",
        ["Negative", "Neutral", "Positive"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    learning_disabilities = st.selectbox(
        "Learning Disabilities",
        ["No", "Yes"]
    )

with col2:
    parental_education = st.selectbox(
        "Parental Education Level",
        ["High School", "College", "Postgraduate"]
    )

with col3:
    distance_from_home = st.selectbox(
        "Distance from Home",
        ["Near", "Moderate", "Far"]
    )


gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Student Performance", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Parental_Involvement": [parental_involvement],
        "Access_to_Resources": [access_to_resources],
        "Extracurricular_Activities": [extracurricular_activities],
        "Sleep_Hours": [sleep_hours],
        "Previous_Scores": [previous_scores],
        "Motivation_Level": [motivation_level],
        "Internet_Access": [internet_access],
        "Tutoring_Sessions": [tutoring_sessions],
        "Family_Income": [family_income],
        "Teacher_Quality": [teacher_quality],
        "School_Type": [school_type],
        "Peer_Influence": [peer_influence],
        "Physical_Activity": [physical_activity],
        "Learning_Disabilities": [learning_disabilities],
        "Parental_Education_Level": [parental_education],
        "Distance_from_Home": [distance_from_home],
        "Gender": [gender]
    })


    # Apply same categorical encoding
    input_encoded = pd.get_dummies(input_data)


    # Make sure input has exactly the same columns
    # and order as the training data
    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )


    # Scale using the saved scaler
    input_scaled = scaler.transform(input_encoded)


    # Make prediction
    prediction = model.predict(input_scaled)[0]


    # Display result
    st.subheader("📊 Prediction Result")

    if prediction == "High":

      st.success(
        "🎉 Predicted Performance: HIGH"
      )

      st.write(
        "The student shows characteristics associated "
        "with high academic performance."
      )

    elif prediction == "Average":

      st.warning(
        "📚 Predicted Performance: AVERAGE"
      )

      st.write(
        "The student shows characteristics associated "
        "with average academic performance."
      )

    elif prediction == "Low":

      st.error(
        "⚠️ Predicted Performance: LOW"
      )

      st.write(
        "The student may benefit from additional academic "
        "support and improved study habits."
      )

    else:
        st.info(f"Predicted Performance: {prediction}")


    # Probability if supported
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_scaled)[0]

        classes = model.classes_

        probability_df = pd.DataFrame({
            "Performance Level": classes,
            "Probability": probabilities
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.subheader("Prediction Probabilities")

        st.dataframe(
         probability_df,
         width="stretch"
       )