import pyAgrum as gum
import pyAgrum.lib.image as gimg
import pandas as pd
import streamlit as st
from PIL import Image
import os

# Streamlit page configuration
st.set_page_config(page_title="Hospital Survey Inference", layout="wide")

# Load data and learn Bayesian Network
try:
    bn = gum.loadBN("tan3max_bg.xdsl")
except Exception as e:
    st.error(f"Failed to load Bayesian network: {e}")
    st.stop()

# Streamlit app layout
st.title("Hospital Survey Bayesian Network Inference")
st.subheader("To use the tool please follow the steps:")
st.write("Enter the average score for each section from a scale of 0 to 10, the score will then automatically be converted into a predetermined state [Bad, Acceptable, Good] based on the studied data set")
st.write("After you input all your values, click the **\"Compute Inference\"** button, an inference with a visual and an interpretation will be generated for you case.")

# Create two columns: inputs on the left, output on the right
st.markdown("<style>div[data-testid='stHorizontalBlock'] {align-items: start;}</style>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Survey Scores")
    # Input fields with state display to the right
    st.markdown("""
        <style>
        /* Ensure uniform input box width */
        div[data-testid="stNumberInput"] {
            width: 200px !important;
            max-width: 200px !important;
            min-width: 200px !important;
        }
        div[data-testid="stNumberInput"] > div > input {
            width: 100% !important;
            box-sizing: border-box;
        }
        /* Vertically align input and state columns */
        div[data-testid="stHorizontalBlock"] > div {
            display: stretch;
            align-items: center;
            justify-content: flex-start;
            padding: 0px;
        }
        /* Position state text close to input */
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
            margin-left: 0px;
            padding-top: 32px;
        }
        </style>
    """, unsafe_allow_html=True)
    inputs = {}
    for label, key in [
        ("S1: Admission to hospital", "S1_Admission_to_hospital"),
        ("S2: The hospital and ward", "S2_The_hospital_and_ward"),
        ("S3: Doctors", "S3_Doctors"),
        ("S4: Nurses", "S4_Nurses"),
        ("S5: Your care and treatment", "S5_Your_care_and_treatment"),
        ("S6: Leaving the hospital", "S6_Leaving_the_hospital"),
        ("S7: Feedback on quality of care", "S7_Feedback_on_quality_of_care"),
        ("S8: Respect and dignity", "S8_Respect_and_dignity")
    ]:
        col_input, col_state = st.columns([0.5, 1])
        with col_input:
            inputs[key] = st.number_input(label, min_value=0.0, max_value=10.0, value=0.0, step=0.5, key=key)
        with col_state:
            if key == "S8_Respect_and_dignity":
                state = "Good" if inputs[key] >= 9.2 else ("Acceptable" if inputs[key] >= 8.9 else "Bad")
            elif key == "S7_Feedback_on_quality_of_care":
                state = "Good" if inputs[key] >= 2.4 else "Bad"
            elif key == "S6_Leaving_the_hospital":
                state = "Good" if inputs[key] >= 7 else "Bad"
            elif key == "S5_Your_care_and_treatment":
                state = "Good" if inputs[key] >= 8 else "Bad"
            elif key == "S4_Nurses":
                state = "Good" if inputs[key] >= 8.7 else ("Acceptable" if inputs[key] >= 8 else "Bad")
            elif key == "S3_Doctors":
                state = "Good" if inputs[key] >= 9 else ("Acceptable" if inputs[key] >= 8.5 else "Bad")
            elif key == "S2_The_hospital_and_ward":
                state = "Good" if inputs[key] >= 8 else "Bad"
            elif key == "S1_Admission_to_hospital":
                state = "Good" if inputs[key] >= 7 else "Bad"
            st.markdown(f"**{state}**")

    # Button to compute inference
    if st.button("Compute Inference"):
        # Compute evidence based on input values
        evs = {}
        S8_val = inputs["S8_Respect_and_dignity"]
        evs["S8_Respect_and_dignity"] = 1 if S8_val >= 9.2 else (2 if S8_val >= 8.9 else 0)

        S7_val = inputs["S7_Feedback_on_quality_of_care"]
        evs["S7_Feedback_on_quality_of_care"] = 1 if S7_val >= 2.4 else 0

        S6_val = inputs["S6_Leaving_the_hospital"]
        evs["S6_Leaving_the_hospital"] = 1 if S6_val >= 7 else 0

        S5_val = inputs["S5_Your_care_and_treatment"]
        evs["S5_Your_care_and_treatment"] = 1 if S5_val >= 8 else 0

        S4_val = inputs["S4_Nurses"]
        evs["S4_Nurses"] = 1 if S4_val >= 8.7 else (2 if S4_val >= 8 else 0)

        S3_val = inputs["S3_Doctors"]
        evs["S3_Doctors"] = 1 if S3_val >= 9 else (2 if S3_val >= 8.5 else 0)

        S2_val = inputs["S2_The_hospital_and_ward"]
        evs["S2_The_hospital_and_ward"] = 1 if S2_val >= 8 else 0

        S1_val = inputs["S1_Admission_to_hospital"]
        evs["S1_Admission_to_hospital"] = 1 if S1_val >= 7 else 0

        # Perform inference
        try:
            # Compute inference for text results
            ie = gum.LazyPropagation(bn)
            for node, state in evs.items():
                ie.addEvidence(node, state)
            ie.makeInference()

            # Display text-based results in the right column
            with col2:
                st.header("Inference Results")
                st.subheader("Probabilities")
                for node in bn.nodes():
                    node_name = bn.variable(node).name()
                    if node_name not in evs:  # Show only non-evidence nodes
                        probs = ie.posterior(node_name)
                        st.write(f"Based on your current inputs, you are predicted to have the following probabilities for Overall Patient Experience:")
                        var = bn.variable(node_name)
                        for i in range(var.domainSize()):
                            label = var.label(i)
                            prob = probs[i]
                            st.write(f"A {100*prob:.2f}% chance of {label} overall experience")

                # Generate and display visualization
                output_file = "inference.png"
                gimg.exportInference(bn, output_file, evs=evs)
                if os.path.exists(output_file):
                    img = Image.open(output_file)
                    st.subheader("Bayesian Network Visualization")
                    st.image(img, caption="Bayesian Network Inference")
                    st.markdown("<style>img {max-width: 700px; max-height: 600px; height: auto;}</style>", unsafe_allow_html=True)
                else:
                    st.error("Failed to generate inference image.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")