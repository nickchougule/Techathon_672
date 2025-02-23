import os
import time
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from firebase_config import register_user, login_user

# Load API keys from .env
load_dotenv("chatbot.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load the detection model
@st.cache_resource
def load_detection_model():
    return keras.models.load_model(r"E:\Techathon_672\trained_cancer_model.h5")

# Function to generate Gemini AI response
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Function for model prediction
def model_prediction(test_image, model):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Blood Cell Recommendations
def get_recommendations(cell_type):
    recommendations = {
        "EOSINOPHIL": {
            "Info": "Eosinophils help in allergic reactions and fighting parasites.",
            "Diet": "Vitamin C-rich foods, turmeric, ginger, omega-3.",
            "Precautions": "Avoid allergens, reduce stress, stay hydrated.",
            "Learn More": "[More on Eosinophils](https://www.ncbi.nlm.nih.gov/books/NBK538361/)"
        },
        "LYMPHOCYTE": {
            "Info": "Lymphocytes are key for immune responses (B-cells, T-cells).",
            "Diet": "High-protein foods, leafy greens, nuts, antioxidants.",
            "Precautions": "Maintain hygiene, get enough sleep.",
            "Learn More": "[More on Lymphocytes](https://www.ncbi.nlm.nih.gov/books/NBK2263/)"
        },
        "MONOCYTE": {
            "Info": "Monocytes become macrophages and help fight infections.",
            "Diet": "Iron-rich foods, probiotics, balanced diet.",
            "Precautions": "Avoid processed foods, exercise regularly.",
            "Learn More": "[More on Monocytes](https://www.ncbi.nlm.nih.gov/books/NBK27128/)"
        },
        "NEUTROPHIL": {
            "Info": "Neutrophils are the first responders against infections.",
            "Diet": "Protein, citrus fruits, hydration.",
            "Precautions": "Avoid crowded places, maintain hygiene.",
            "Learn More": "[More on Neutrophils](https://www.ncbi.nlm.nih.gov/books/NBK562241/)"
        }
    }
    return recommendations[cell_type]

# Streamlit UI
st.sidebar.title("🔬 Cancer Cell Detection & Segmentation")

# Ensure authentication
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Show login/register if not authenticated
if not st.session_state["logged_in"]:
    menu = st.sidebar.selectbox("Menu", ["🔐 Login/Register"])
else:
    menu = st.sidebar.selectbox("Menu", ["🏠 Home", "🔍 Cancer Cell Detection", "💬 Chatbot", "🚪 Logout"])

# Login & Registration
if menu == "🔐 Login/Register":
    st.title("🔐 Cancer Detection Platform - Login/Register")
    auth_mode = st.radio("Select Authentication Mode:", ["Login", "Register"])
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if auth_mode == "Login":
        if st.button("Login"):
            if not email or not password:
                st.error("❌ Please enter both email and password!")
            else:
                user = login_user(email, password)
                if user:
                    st.success("✅ Logged in successfully!")
                    st.session_state["logged_in"] = True
                    st.session_state["messages"] = []  # Clear chat history for new login
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password!")

    elif auth_mode == "Register":
        if st.button("Register"):
            if not email or not password or "@gmail.com" not in email:
                st.error("❌ Please enter a valid Gmail ID and password!")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters long!")
            else:
                user_id = register_user(email, password)
                if user_id:
                    st.success("✅ Registered successfully! Now you can log in.")
                else:
                    st.error("❌ Registration failed.")

# Home Page
elif menu == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🧬 Cancer Cell Detection and Segmentation</h1>", unsafe_allow_html=True)
    st.write("📌 Detect different types of **cancerous blood cells** using deep learning.")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img = Image.open(r'E:\Techathon_672\cancer.jpg')
        st.image(img, caption="Cancer Cells Under Microscope", use_container_width=True)

# Cancer Cell Detection Page
elif menu == "🔍 Cancer Cell Detection":
    st.header('📷 Upload a Microscopic Blood Image for Analysis')

    test_image = st.file_uploader("**Choose an image:**", type=["jpg", "png", "jpeg"])

    # Store results in session state to prevent refresh issues
    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None
        st.session_state["cell_type"] = None

    if test_image:
        st.markdown("### Preview of Uploaded Image:")
        st.image(test_image, use_container_width=True)

        if st.button("🔍 Predict"):
            with st.spinner('⏳ Analyzing Image...'):
                time.sleep(2)  
                model = load_detection_model()
                result_index = model_prediction(test_image, model)
                class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
                detected_cell = class_names[result_index]

                # Store results in session state
                st.session_state["prediction"] = detected_cell
                st.session_state["cell_type"] = get_recommendations(detected_cell)

    # Display results from session state
    if st.session_state["prediction"]:
        st.success(f'✅ **Prediction:** {st.session_state["prediction"]}')
        recs = st.session_state["cell_type"]

        st.markdown("### ℹ️ Information")
        st.info(recs["Info"])

        st.markdown("### 🥗 Diet Plan")
        st.success(recs["Diet"])

        st.markdown("### ⚠️ Precautions")
        st.warning(recs["Precautions"])

        st.markdown(f"### 🔗 Learn More: {recs['Learn More']}")

# Chatbot Page (Gemini AI)
elif menu == "💬 Chatbot":
    st.title("💬 AI Chatbot - Ask About Cancer Cells & Segmentation")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # Ensure messages are unique per user

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me about cancer cell detection, segmentation, or general queries...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            bot_response = get_gemini_response(user_input)

        st.session_state["messages"].append({"role": "assistant", "content": bot_response})

        with st.chat_message("assistant"):
            st.markdown(bot_response)

# Logout
elif menu == "🚪 Logout":
    st.session_state["logged_in"] = False
    st.session_state["messages"] = []  # Clear chat history on logout
    st.success("✅ Logged out successfully!")
    time.sleep(1)
    st.rerun()
