import streamlit as st
import pandas as pd
import numpy as np
import pywt
from scipy import stats
from tensorflow.keras.models import load_model

# Load the model
@st.cache(allow_output_mutation=True)
def load_model_file():
    return load_model('D:/IIT_SE/Learning Resources/4th Year/FYP/Model/my_model.h5') 

loaded_model = load_model_file()

# Function for denoising ECG data
def denoise(input_data):
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values.flatten()
    wavelet = pywt.Wavelet('sym4')
    max_levels = pywt.dwt_max_level(len(input_data), wavelet.dec_len)
    threshold = 0.04  
    wavelet_coeffs = pywt.wavedec(input_data, 'sym4', level=max_levels)
    for i in range(1, len(wavelet_coeffs)):
        wavelet_coeffs[i] = pywt.threshold(wavelet_coeffs[i], threshold * max(wavelet_coeffs[i]))
    denoised_data = pywt.waverec(wavelet_coeffs, 'sym4')
    return denoised_data

# Function to preprocess and predict CAD
def predict_cad(ecg_data):
    # Denoise the ECG signals in the actual data
    for index, row in ecg_data.iterrows():
        ecg_data.iloc[index] = denoise(row)
    
    # Perform Z-score normalization on the actual data
    actual_data = stats.zscore(ecg_data)

    # Reshape actual data to match the expected input shape
    actual_data_reshaped = np.expand_dims(actual_data, axis=-1)
    pad_width = ((0, 0), (0, max(0, 360 - actual_data_reshaped.shape[1])), (0, 0))
    actual_data_reshaped = np.pad(actual_data_reshaped, pad_width, mode='constant')  

    # Make predictions on the actual data
    predictions_proba_actual = loaded_model.predict(actual_data_reshaped)
    predictions_actual = np.argmax(predictions_proba_actual, axis=1)

    # Define class labels
    class_labels = ['N', 'L', 'R', 'A', 'V']

    # Convert predictions to class labels
    predicted_labels = [class_labels[prediction] for prediction in predictions_actual]

    # Calculate the count for each class separately
    class_counts = {label: predicted_labels.count(label) for label in class_labels}

    # Determine which class has the highest count
    most_common_class = max(class_counts, key=class_counts.get)

    # Return predictions
    return predicted_labels, class_counts, most_common_class

# Function to display introduction page
def intro_page():
    st.title("ECG Based CAD Prediction System")
    st.markdown("**Hello there!**", unsafe_allow_html=True)
    st.markdown("**Welcome to the CAD Prediction System, where cutting-edge technology meets life-saving potential.**", unsafe_allow_html=True)
    st.write("Our innovative system utilizes ECG data to predict the likelihood of CAD in individuals. By harnessing the power of machine learning and advanced algorithms, we can provide accurate and timely predictions, enabling early intervention and prevention strategies. With our user-friendly interface, healthcare professionals can easily input ECG data and receive instantaneous risk assessments, allowing for swift decision-making and personalized patient care.")
    # Image link
    image_link = "https://media.licdn.com/dms/image/D5612AQFTmcnOkh6_8Q/article-cover_image-shrink_720_1280/0/1663775961726?e=2147483647&v=beta&t=M9mbbxYPGFr47zo4VkNGeWPOuwW_5c737dqp-MnE32k"
    # Display image
    st.image(image_link, use_column_width=True)
    if st.button("Get Started"):
        st.session_state.page = "Data Input"


# Function to display data input page
def data_input_page():
    st.title("Data Input Page")
    st.write("Please choose an ECG file (.csv format):")
    ecg_file = st.file_uploader("Upload CSV", type=['csv'])
    if ecg_file is not None:
        predict_button = st.button("Predict CAD")
        if predict_button:
            with st.spinner("Please wait while we process your data..."):
                ecg_data = pd.read_csv(ecg_file)  
                predictions, class_counts, most_common_class = predict_cad(ecg_data)
                st.session_state.predictions = predictions
                st.session_state.class_counts = class_counts
                st.session_state.most_common_class = most_common_class
                st.session_state.page = "Result" 
    if st.button("Back"):
        st.session_state.page = "Introduction"

# Function to display CAD prediction result page
def result_page():
    st.title("CAD Prediction Result")
    if "predictions" in st.session_state:
        # st.write("Predictions for actual data:", st.session_state.predictions)
        # st.write("Class counts:", st.session_state.class_counts)
        # st.write("Most common class:", st.session_state.most_common_class)
        most_common_class = st.session_state.most_common_class
        if most_common_class == "N":
            st.write("You are Normal.")
            # Image link
            image_link = "https://t3.ftcdn.net/jpg/03/27/29/30/360_F_327293004_GHZxnzImI4VphIN7VFFwfPcjCgk6k3eq.jpg"
            # Display image 
            st.image(image_link, use_column_width=True)
        else:
            st.write("CAD Detected. Seek Medical Advice For Further Treatments.")
            # Image link
            image_link = "https://t4.ftcdn.net/jpg/01/84/13/19/360_F_184131905_HAGEl2Aov4XgTAh2yIgK8wzgDrFrfeqm.jpg"
            # Display image 
            st.image(image_link, use_column_width=True)
    if st.button("Back"):
        st.session_state.page = "Data Input"

# Main function to run the application
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Introduction"
    
    if st.session_state.page == "Introduction":
        intro_page()
    elif st.session_state.page == "Data Input":
        data_input_page()
    elif st.session_state.page == "Result":
        result_page()

if __name__ == "__main__":
    main()