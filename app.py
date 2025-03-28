import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
scaler = joblib.load('scaler.pkl')
rf = joblib.load('random_forest_model.pkl')

# Apply custom CSS styling
def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit UI
def main():
    st.title("Breast Cancer Prediction App")
    st.markdown("### Enter the required details to check if the sample is Cancerous or Not.")

    load_css()  # Load custom styles

    # User input fields
    clump_thickness = st.number_input("Clump Thickness", min_value=0.0, step=0.1)
    uniformity_of_cell_shape = st.number_input("Uniformity of Cell Shape", min_value=0.0, step=0.1)
    marginal_adhesion = st.number_input("Marginal Adhesion", min_value=0.0, step=0.1)
    single_epithelial_cell_size = st.number_input("Single Epithelial Cell Size", min_value=0.0, step=0.1)
    bare_nuclei = st.number_input("Bare Nuclei", min_value=0.0, step=0.1)
    bland_chromatin = st.number_input("Bland Chromatin", min_value=0.0, step=0.1)
    normal_nucleoli = st.number_input("Normal Nucleoli", min_value=0.0, step=0.1)

    # Prediction Button
    if st.button("Predict"):
        # Collect inputs into a pandas DataFrame with proper feature names
        input_data = pd.DataFrame([[
            clump_thickness, uniformity_of_cell_shape, marginal_adhesion, 
            single_epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli
        ]], columns=[
            'Clump_thickness', 'Uniformity_of_cell_shape', 'Marginal_adhesion', 
            'Single_epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli'
        ])

        # Scale the input data
        input_scaled = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

        print(input_data)
        print(input_scaled_df)

        # Make prediction
        prediction = rf.predict(input_scaled_df)
        result = "Cancerous" if prediction[0] == 1 else "Not Cancerous"

        # Display result with styling
        st.markdown(f'<div class="result-box"><b>Prediction: {result}</b></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
