import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import pandas as pd
import plotly.express as px
from io import StringIO
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
import streamlit.components.v1 as components

import fitz  # PyMuPDF
import google.generativeai as genai
import re
import json


def extract_json_from_response(response_text):
    """
    Extracts and parses a JSON object from a string that may contain other text.
    """
    try:
        # Match first valid JSON object in the string using regex
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response.")
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return None


load_dotenv()

from codebase.dashboard_graphs import MaternalHealthDashboard

maternal_model = pickle.load(open("model/finalized_maternal_model_v1_faster.sav",'rb'))
fetal_model = pickle.load(open("model/fetal_health_classifier_v1_faster.sav",'rb'))

# Initialize session state for clearing inputs
if 'clear_pregnancy' not in st.session_state:
    st.session_state.clear_pregnancy = False
if 'clear_fetal' not in st.session_state:
    st.session_state.clear_fetal = False

# sidebar for navigation
with st.sidebar:
    st.title("NavJeevanAI")
    st.write("Welcome to the NavJeevanAI")
    st.write(" Choose an option from the menu below to get started:")

    selected = option_menu('NavJeevanAI',
                          
                          ['About us',
                            'Pregnancy Risk Prediction',
                           'Fetal Health Prediction',
                           'Dashboard',
                           'AI Assistant'],
                          icons=['chat-square-text','hospital','capsule-pill','clipboard-data', 'robot'],
                          menu_icon='bi bi-cpu', # activity
                          default_index=0)
    
if (selected == 'About us'):
    
    st.title("Welcome to NavJeevanAI")
    st.write("At NavJeevanAI, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. "
         "Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate "
         "predictions and proactive risk management.")
    
    col1, col2= st.columns(2)
    with col1:
        # Section 1: Pregnancy Risk Prediction
        st.header("1. Pregnancy Risk Prediction")
        st.write("Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various parameters, including age, "
                "body sugar levels, blood pressure, and more. By processing this information, we provide accurate predictions of "
                "potential risks during pregnancy.")
        # Add an image for Pregnancy Risk Prediction
        st.image("graphics/pregnancy_risk_image.jpg", caption="Pregnancy Risk Prediction", use_column_width=True)
    with col2:
        # Section 2: Fetal Health Prediction
        st.header("2. Fetal Health Prediction")
        st.write("Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge technology to assess the "
                "health status of the fetus. Through a comprehensive analysis of factors such as ultrasound data, maternal health, "
                "and genetic factors, we deliver insights into the well-being of the unborn child.")
        # Add an image for Fetal Health Prediction
        st.image("graphics/fetal_health_image.jpg", caption="Fetal Health Prediction", use_column_width=True)

    # Section 3: Dashboard
    st.header("3. Dashboard")
    st.write("Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic "
            "view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard "
            "is designed for ease of use and accessibility.")
    
    # Closing note
    st.write("Thank you for choosing E-Doctor. We are committed to advancing healthcare through technology and predictive analytics. "
            "Feel free to explore our features and take advantage of the insights we provide.")

if selected == "Pregnancy Risk Prediction":
    st.title("Pregnancy Risk Prediction")
    
    # PDF Upload Section
    st.subheader("Upload Prescription Report")
    uploaded_pdf = st.file_uploader("Drag and drop or browse a PDF", type="pdf")

    if uploaded_pdf is not None:
        # Save temporarily and extract text
        pdf_doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        extracted_text = ""
        for page in pdf_doc:
            extracted_text += page.get_text()

        st.success("Text extracted from PDF")

        # Gemini API configuration
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Gemini prompt
        prompt = f"""
        You are a medical assistant. From the prescription report text below, extract and return the following fields in JSON:

        - age
        - diastolicBP
        - BS
        - bodyTemp
        - heartRate

        Respond only with a valid JSON object.

        Prescription Text:
        {extracted_text}
        """

        if st.button("Generate Report Analysis"):
            with st.spinner("Analyzing using Gemini..."):
                response = model.generate_content(prompt)

                # Use the parser
                parsed_json = extract_json_from_response(response.text)

                if parsed_json:
                    st.session_state['pregnancy_json'] = parsed_json
                    st.success("Report successfully analyzed!")
                    st.json(parsed_json)
                else:
                    st.error("❌ Failed to parse Gemini response as JSON.")
                    st.code(response.text)

    # Single "Fill from Report" button - only show if we have parsed JSON data
    if 'pregnancy_json' in st.session_state:
        if st.button("📄 Fill from Report", key="fill_from_pregnancy_report"):
            json_data = st.session_state['pregnancy_json']
            
            # Set values in session_state for auto-fill
            st.session_state.pregnancy_age = str(json_data.get("age", ""))
            st.session_state.pregnancy_diastolicBP = str(json_data.get("diastolicBP", ""))
            st.session_state.pregnancy_BS = str(json_data.get("BS", ""))
            st.session_state.pregnancy_bodyTemp = str(json_data.get("bodyTemp", ""))
            st.session_state.pregnancy_heartRate = str(json_data.get("heartRate", ""))
            
            st.rerun()

    # Page description
    content = "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age of the Person', key="pregnancy_age", value="" if st.session_state.clear_pregnancy else st.session_state.get('pregnancy_age', ""))
        
    with col2:
        diastolicBP = st.text_input('diastolicBP in mmHg', key="pregnancy_diastolicBP", value="" if st.session_state.clear_pregnancy else st.session_state.get('pregnancy_diastolicBP', ""))
    
    with col3:
        BS = st.text_input('Blood glucose in mmol/L', key="pregnancy_BS", value="" if st.session_state.clear_pregnancy else st.session_state.get('pregnancy_BS', ""))
    
    with col1:
        bodyTemp = st.text_input('Body Temperature in Celsius', key="pregnancy_bodyTemp", value="" if st.session_state.clear_pregnancy else st.session_state.get('pregnancy_bodyTemp', ""))

    with col2:
        heartRate = st.text_input('Heart rate in beats per minute', key="pregnancy_heartRate", value="" if st.session_state.clear_pregnancy else st.session_state.get('pregnancy_heartRate', ""))
    
    # Reset clear flag after inputs are rendered
    if st.session_state.clear_pregnancy:
        st.session_state.clear_pregnancy = False
    
    # Load scaler and prediction logic
    scale_X = pickle.load(open('model/scaler_maternal_model.sav', 'rb'))
    
    # Prediction and Clear buttons
    with col1:
        if st.button('Predict Pregnancy Risk'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_data = np.array([[age, diastolicBP, BS, bodyTemp, heartRate]])
                input_scaled = scale_X.transform(input_data)
                predicted_risk = maternal_model.predict(input_scaled)
            
            st.subheader("Risk Level:")
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p></bold>', unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Medium Risk</p></Bold>', unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p><bold>', unsafe_allow_html=True)
    
    with col2:
        if st.button("Clear", key="pregnancy_clear"): 
            st.session_state.clear_pregnancy = True
            st.rerun()
            
if (selected == 'Fetal Health Prediction'):

        # --- PDF Upload Section for Fetal Prediction ---
    st.subheader("Upload Fetal Health Report")
    uploaded_pdf_fetal = st.file_uploader("Upload PDF for Fetal Health Analysis", type="pdf", key="fetal_pdf_upload")

    if uploaded_pdf_fetal is not None:
        pdf_doc = fitz.open(stream=uploaded_pdf_fetal.read(), filetype="pdf")
        extracted_text_fetal = ""
        for page in pdf_doc:
            extracted_text_fetal += page.get_text()

        st.success("Text extracted from PDF")

        # Gemini API Config
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        fetal_prompt = f"""
        You are a medical assistant. Extract the following fetal health report fields from the given prescription and return only valid JSON:
        - BaselineValue
        - Accelerations
        - fetal_movement
        - uterine_contractions
        - light_decelerations
        - severe_decelerations
        - prolongued_decelerations
        - abnormal_short_term_variability
        - mean_value_of_short_term_variability
        - percentage_of_time_with_abnormal_long_term_variability
        - mean_value_of_long_term_variability
        - histogram_width
        - histogram_min
        - histogram_max
        - histogram_number_of_peaks
        - histogram_number_of_zeroes
        - histogram_mode
        - histogram_mean
        - histogram_median
        - histogram_variance
        - histogram_tendency

        Prescription Text:
        {extracted_text_fetal}
        """

        if st.button("Generate Fetal Report Analysis"):
            with st.spinner("Analyzing fetal report with Gemini..."):
                response = model.generate_content(fetal_prompt)
                parsed_fetal_json = extract_json_from_response(response.text)

                if parsed_fetal_json:
                    st.session_state['fetal_json'] = parsed_fetal_json
                    st.success("Fetal report analyzed successfully!")
                    st.json(parsed_fetal_json)
                else:
                    st.error("❌ Failed to parse Gemini response as JSON.")
                    st.code(response.text)
    
    # page title
    st.title('Fetal Health Prediction')
    
    if 'fetal_json' in st.session_state:
        if st.button("📄 Fill from Report", key="fill_from_fetal_report"):
            json_data = st.session_state['fetal_json']
            for key, value in json_data.items():
                st.session_state[f"fetal_{key}"] = str(value)
            st.rerun()
    
    content = "Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        BaselineValue = st.text_input('Baseline Value', key="fetal_BaselineValue", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_BaselineValue', ""))
        
    with col2:
        Accelerations = st.text_input('Accelerations', key="fetal_Accelerations", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_Accelerations', ""))
    
    with col3:
        fetal_movement = st.text_input('Fetal Movement', key="fetal_movement", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_movement', ""))
    
    with col1:
        uterine_contractions = st.text_input('Uterine Contractions', key="fetal_uterine_contractions", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_uterine_contractions', ""))

    with col2:
        light_decelerations = st.text_input('Light Decelerations', key="fetal_light_decelerations", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_light_decelerations', ""))
    
    with col3:
        severe_decelerations = st.text_input('Severe Decelerations', key="fetal_severe_decelerations", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_severe_decelerations', ""))

    with col1:
        prolongued_decelerations = st.text_input('Prolongued Decelerations', key="fetal_prolongued_decelerations", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_prolongued_decelerations', ""))
        
    with col2:
        abnormal_short_term_variability = st.text_input('Abnormal Short Term Variability', key="fetal_abnormal_short_term_variability", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_abnormal_short_term_variability', ""))
    
    with col3:
        mean_value_of_short_term_variability = st.text_input('Mean Value Of Short Term Variability', key="fetal_mean_value_of_short_term_variability", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_mean_value_of_short_term_variability', ""))
    
    with col1:
        percentage_of_time_with_abnormal_long_term_variability = st.text_input('Percentage Of Time With ALTV', key="fetal_percentage_of_time_with_abnormal_long_term_variability", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_percentage_of_time_with_abnormal_long_term_variability', ""))

    with col2:
        mean_value_of_long_term_variability = st.text_input('Mean Value Long Term Variability', key="fetal_mean_value_of_long_term_variability", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_mean_value_of_long_term_variability', ""))
    
    with col3:
        histogram_width = st.text_input('Histogram Width', key="fetal_histogram_width", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_width', ""))

    with col1:
        histogram_min = st.text_input('Histogram Min', key="fetal_histogram_min", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_min', ""))
        
    with col2:
        histogram_max = st.text_input('Histogram Max', key="fetal_histogram_max", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_max', ""))
    
    with col3:
        histogram_number_of_peaks = st.text_input('Histogram Number Of Peaks', key="fetal_histogram_number_of_peaks", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_number_of_peaks', ""))
    
    with col1:
        histogram_number_of_zeroes = st.text_input('Histogram Number Of Zeroes', key="fetal_histogram_number_of_zeroes", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_number_of_zeroes', ""))

    with col2:
        histogram_mode = st.text_input('Histogram Mode', key="fetal_histogram_mode", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_mode', ""))
    
    with col3:
        histogram_mean = st.text_input('Histogram Mean', key="fetal_histogram_mean", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_mean', ""))
    
    with col1:
        histogram_median = st.text_input('Histogram Median', key="fetal_histogram_median", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_median', ""))

    with col2:
        histogram_variance = st.text_input('Histogram Variance', key="fetal_histogram_variance", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_variance', ""))
    
    with col3:
        histogram_tendency = st.text_input('Histogram Tendency', key="fetal_histogram_tendency", value="" if st.session_state.clear_fetal else st.session_state.get('fetal_histogram_tendency', ""))
    
    # Reset clear flag after inputs are rendered
    if st.session_state.clear_fetal:
        st.session_state.clear_fetal = False
    
    # creating a button for Prediction
    st.markdown('</br>', unsafe_allow_html=True)
    with col1:
        if st.button('Predict Fetal Health'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = fetal_model.predict([[BaselineValue, Accelerations, fetal_movement,
       uterine_contractions, light_decelerations, severe_decelerations,
       prolongued_decelerations, abnormal_short_term_variability,
       mean_value_of_short_term_variability,
       percentage_of_time_with_abnormal_long_term_variability,
       mean_value_of_long_term_variability, histogram_width,
       histogram_min, histogram_max, histogram_number_of_peaks,
       histogram_number_of_zeroes, histogram_mode, histogram_mean,
       histogram_median, histogram_variance, histogram_tendency]])
            # st.subheader("Risk Level:")
            st.markdown('</br>', unsafe_allow_html=True)
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Result  Comes to be  Normal</p></bold>', unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Result  Comes to be  Suspect</p></Bold>', unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">Result  Comes to be  Pathological</p><bold>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear", key="fetal_clear"): 
            st.session_state.clear_fetal = True
            st.rerun()

if (selected == "Dashboard"):
    api_key = os.getenv("DASHBOARD_API_KEY")
    api_endpoint = api_endpoint= f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"
    st.header("Dashboard")
    content = "Our interactive dashboard offers a comprehensive visual representation of maternal health achievements across diverse regions. The featured chart provides insights into the performance of each region concerning institutional deliveries compared to their assessed needs. It serves as a dynamic tool for assessing healthcare effectiveness, allowing users to quickly gauge the success of maternal health initiatives."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    dashboard = MaternalHealthDashboard(api_endpoint)
    dashboard.create_bubble_chart()
    with st.expander("Show More"):
    # Display a portion of the data
        content = dashboard.get_bubble_chart_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)

    dashboard.create_pie_chart()
    with st.expander("Show More"):
    # Display a portion of the data
        content = dashboard.get_pie_graph_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)



if selected == 'AI Assistant':
    st.markdown("<h1 style='text-align: center;'>🧠 AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; font-size: 18px;'>
            Our AI Assistant is here to help! Whether you're navigating the platform, have questions about pregnancy risk,
            fetal health, or using our dashboard — just ask.
        </div>
        <br><br>
    """, unsafe_allow_html=True)

    # Embed Convai AI widget in center
    components.html(
        """
        <div style="display: flex; justify-content: center;">
            <div style="width: 350px; height: 220px;">
                <elevenlabs-convai agent-id="agent_01jx740krne1ra7ynz6bt6na85"></elevenlabs-convai>
            </div>
        </div>
        <script src="https://unpkg.com/@elevenlabs/convai-widget-embed" async type="text/javascript"></script>
        """,
        height=500
    )