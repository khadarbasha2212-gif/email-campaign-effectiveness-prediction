
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Email Campaign Prediction",
    page_icon="📧",
    layout="centered"
)

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load(
        "models/weighted_xgboost_pipeline.joblib"
    )

model = load_model()

st.title("📧 Email Campaign Effectiveness Prediction")
st.write(
    "Enter the email characteristics to predict its status class."
)

# Input form
with st.form("email_prediction_form"):
    email_type = st.number_input(
        "Email Type",
        min_value=0,
        step=1,
        value=1
    )

    hotness = st.number_input(
        "Subject Hotness Score",
        value=2.5
    )

    source_type = st.text_input(
        "Email Source Type",
        value="Marketing"
    )

    location = st.text_input(
        "Customer Location",
        value="Unknown"
    )

    campaign_type = st.number_input(
        "Email Campaign Type",
        min_value=0,
        step=1,
        value=2
    )

    past_communications = st.number_input(
        "Total Past Communications",
        min_value=0,
        value=5
    )

    sent_time = st.number_input(
        "Time Email Sent Category",
        min_value=0,
        step=1,
        value=2
    )

    word_count = st.number_input(
        "Word Count",
        min_value=0,
        value=150
    )

    total_links = st.number_input(
        "Total Links",
        min_value=0,
        value=3
    )

    total_images = st.number_input(
        "Total Images",
        min_value=0,
        value=1
    )

    submitted = st.form_submit_button("Predict Email Status")

if submitted:
    new_email = pd.DataFrame([{
        "Email_Type": email_type,
        "Subject_Hotness_Score": hotness,
        "Email_Source_Type": source_type,
        "Customer_Location": location,
        "Email_Campaign_Type": campaign_type,
        "Total_Past_Communications": past_communications,
        "Time_Email_sent_Category": sent_time,
        "Word_Count": word_count,
        "Total_Links": total_links,
        "Total_Images": total_images
    }])

    try:
        prediction = model.predict(new_email)[0]
        probabilities = model.predict_proba(new_email)[0]

        st.success(f"Predicted status class: {prediction}")

        st.subheader("Estimated probability by class")

        for label, probability in enumerate(probabilities):
            st.write(f"Class {label}: {probability:.2%}")
            st.progress(float(probability))

        st.caption(
            "Probabilities are model estimates. "
            "Status class meanings must be verified "
            "from the dataset documentation."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
