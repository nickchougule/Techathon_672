import os
import time
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from PIL import Image
import mysql.connector
import google.generativeai as genai

# Load API keys
from dotenv import load_dotenv
load_dotenv("chatbot.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# MySQL Connection
def get_db_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="root",
        database="cancer_detection"
    )

# User Authentication with MySQL
def login_user(email, password_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s AND password_hash=%s", (email, password_hash))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def register_user(email, password_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password_hash))
        conn.commit()
        return cursor.lastrowid
    except mysql.connector.Error:
        return None
    finally:
        cursor.close()
        conn.close()

# Load the detection model
@st.cache_resource
def load_detection_model():
    return keras.models.load_model(r"E:\Techathon_672\trained_cancer_model.h5")

# Function to generate Gemini AI response
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
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
        "EOSINOPHIL": {"Info": "Eosinophils help in allergic reactions and fighting parasites.", "Diet": "Vitamin C-rich foods, turmeric, ginger, omega-3.", "Precautions": "Avoid allergens, reduce stress, stay hydrated."},
        "LYMPHOCYTE": {"Info": "Lymphocytes are key for immune responses.", "Diet": "High-protein foods, leafy greens, nuts.", "Precautions": "Maintain hygiene, get enough sleep."},
        "MONOCYTE": {"Info": "Monocytes become macrophages and help fight infections.", "Diet": "Iron-rich foods, probiotics, balanced diet.", "Precautions": "Avoid processed foods, exercise regularly."},
        "NEUTROPHIL": {"Info": "Neutrophils are first responders against infections.", "Diet": "Protein, citrus fruits, hydration.", "Precautions": "Avoid crowded places, maintain hygiene."}
    }
    return recommendations[cell_type]

# Maintain Active Page After Refresh Using Session State
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_email"] = None
    st.session_state["messages"] = {}

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🏠 Home" if st.session_state["logged_in"] else "🔐 Login/Register"

# Sidebar Navigation
st.sidebar.title("🔬 Cancer Cell Detection & Segmentation")
menu_options = ["🔐 Login/Register"] if not st.session_state["logged_in"] else ["🏠 Home", "🔍 Cancer Cell Detection", "💬 Chatbot", "🚪 Logout"]

# Ensure active_page exists in menu_options
if st.session_state["active_page"] not in menu_options:
    st.session_state["active_page"] = "🏠 Home" if st.session_state["logged_in"] else "🔐 Login/Register"

# User selects a page
menu = st.sidebar.radio("Menu", menu_options, index=menu_options.index(st.session_state["active_page"]))

# Update session state when switching pages
st.session_state["active_page"] = menu

# Login & Registration
if menu == "🔐 Login/Register":
    st.title("🔐 Cancer Detection Platform - Login/Register")
    auth_mode = st.radio("Select Authentication Mode:", ["Login", "Register"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if auth_mode == "Login" and st.button("Login"):
        if not email or not password:
            st.error("❌ Please enter both email and password!")
        else:
            user = login_user(email, password)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email
                if email not in st.session_state["messages"]:
                    st.session_state["messages"][email] = []
                st.success("✅ Logged in successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid email or password!")
    elif auth_mode == "Register" and st.button("Register"):
        if not email or not password:
            st.error("❌ Please enter both email and password!")
        else:
            user_id = register_user(email, password)
            if user_id:
                st.success("✅ Registered successfully! Please login.")
            else:
                st.error("❌ Registration failed! Email might be already used.")

# Home Page
elif menu == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🧬 Cancer Cell Detection and Segmentation</h1>", unsafe_allow_html=True)
    img = Image.open(r'E:\Techathon_672\cancer.jpg')
    st.image(img, caption="Cancer Cells Under Microscope", use_container_width=True)

# Cancer Cell Detection Page
elif menu == "🔍 Cancer Cell Detection":
    st.header('📷 Upload a Microscopic Blood Image for Analysis')
    test_image = st.file_uploader("**Choose an image:**", type=["jpg", "png", "jpeg"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Predict"):
            model = load_detection_model()
            result_index = model_prediction(test_image, model)
            class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
            detected_cell = class_names[result_index]
            recs = get_recommendations(detected_cell)
            st.success(f'✅ **Prediction:** {detected_cell}')
            st.subheader("📌 Information")
            st.info(recs["Info"])
            st.subheader("⚠️ Precautions")
            st.warning(recs["Precautions"])
            st.subheader("🥗 Recommended Diet")
            st.success(recs["Diet"])

# Chatbot Page
elif menu == "💬 Chatbot":
    st.title("💬 AI Chatbot - Ask About Cancer Cells & Segmentation")
    user_input = st.chat_input("Ask me anything...")
    email = st.session_state["user_email"]
    if email and email not in st.session_state["messages"]:
        st.session_state["messages"][email] = []
    if user_input:
        bot_response = get_gemini_response(user_input)
        st.session_state["messages"][email].append(("You", user_input))
        st.session_state["messages"][email].append(("Bot", bot_response))
    for sender, msg in st.session_state["messages"].get(email, []):
        st.chat_message(sender).markdown(msg)

# Logout
elif menu == "🚪 Logout":
    st.session_state.clear()
    st.success("✅ Logged out successfully!")
    time.sleep(1)
    st.rerun()