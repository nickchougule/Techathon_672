import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the Model Once (Performance Optimization)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"E:\Techathon_672\trained_cancer_model.h5")

# Prediction Function
def model_prediction(test_image, model):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expanding dimensions to match model input
    
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar Configuration
st.sidebar.title("🔬 Cancer Cell Detection and Segmentation")
app_mode = st.sidebar.radio('Select a Page:', ['🏠 Home', '🔍 Cancer Cell Detection'])

# Display Banner Image
st.markdown("<h1 style='text-align: center;'>🧬 Cancer Cell Detection and Segmentation</h1>", unsafe_allow_html=True)

# Load and display a sample image
col1, col2, col3 = st.columns([1, 3, 1])  # Center alignment
with col2:
    img = Image.open(r'E:\Techathon_672\cancer.jpg')
    st.image(img, caption="Cancer Cells Under Microscope", use_column_width=True)

# Home Page
if app_mode == '🏠 Home':
    st.markdown("<h3 style='text-align: center;'>Welcome to the Cancer Cell Detection Platform</h3>", unsafe_allow_html=True)
    st.write("📌 This tool helps in detecting different types of **cancerous blood cells** using deep learning.")
    st.write("🔹 Upload a **microscopic blood image** to get predictions.")
    st.write("🔹 Works with **Eosinophils, Lymphocytes, Monocytes, and Neutrophils**.")

# Cancer Cell Detection Page
elif app_mode == '🔍 Cancer Cell Detection':
    st.header('📷 Upload a Microscopic Blood Image for Analysis')

    test_image = st.file_uploader("**Choose an image:**", type=["jpg", "png", "jpeg"])
    
    # Show Image Button
    if test_image:
        st.markdown("### Preview of Uploaded Image:")
        st.image(test_image, width=400, use_column_width=True)

    # Predict Button
    if test_image and st.button("🔍 Predict"):
        st.markdown("## ⏳ Analyzing Image...")
        model = load_model()
        result_index = model_prediction(test_image, model)

        # Class Labels
        class_names = ['🩸 EOSINOPHIL', '🦠 LYMPHOCYTE', '🔬 MONOCYTE', '🧪 NEUTROPHIL']
        st.success(f'✅ **Prediction:** {class_names[result_index]}')

        # Additional Information for User
        st.markdown("### 🏥 Medical Insights:")
        if result_index == 0:
            st.info("🔸 **Eosinophils** are involved in allergic reactions and fight parasites.")
        elif result_index == 1:
            st.info("🔸 **Lymphocytes** play a key role in immune responses (T-cells & B-cells).")
        elif result_index == 2:
            st.info("🔸 **Monocytes** develop into macrophages and help in fighting infections.")
        elif result_index == 3:
            st.info("🔸 **Neutrophils** are the first responders to infections.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>© 2025 Cancer Cell Detection AI</h5>", unsafe_allow_html=True)
