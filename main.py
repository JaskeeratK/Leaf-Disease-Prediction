import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import uuid

if "session_id" not in st.session_state:
    st.session_state["session_id"]=str(uuid.uuid4())


try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(r"path/to/firebase-admin-sdk.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

UPLOAD_FOLDER='uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

def save_images_locally(image_file):
    file_path=os.path.join(UPLOAD_FOLDER,image_file.name)
    with open(file_path,"wb") as f:
        f.write(image_file.getbuffer())
    return file_path

def save_prediction(image_path,predicted_class):
    try:
        data = {
            "image_path": image_path,
            "prediction": predicted_class,
            "session_id": st.session_state["session_id"]
        }
        print("Saving to Firestore:", data)  
        db.collection("predictions").add(data)
        print("Prediction saved successfully!")
    except Exception as e:
        print("Error saving prediction:", str(e))

def get_past_prediction():
    predictions = db.collection("predictions").where("session_id", "==", st.session_state["session_id"]).stream()
    past_data = []
    for p in predictions:
        doc = p.to_dict()
        print("Fetched Data:", doc) 
        past_data.append({"image_path": doc["image_path"], "prediction": doc["prediction"]})
    
    return past_data

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/leaf_disease_prediction2.h5"

model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_idx.json"))


def load_preprocess(image_path, target_size=(224, 224)):

    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_class(model, image_file, class_idx):
    preprocessed_img = load_preprocess(image_file)

    prediction = model.predict(preprocessed_img)
    is_diseased=np.max(prediction)>0.5

    if prediction.size == 0:
        return "Prediction Error"

    if is_diseased:
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_idx.get(str(predicted_class_index), "Unknown")
    else:
        predicted_class_name=" ☘️ Healthy Leaf ☘️"

    confidence = np.max(prediction) * 100
    return predicted_class_name, round(confidence, 2)


# Streamlit App
st.title('🌿Leaf Disease Predictor🌿')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    with col1:
        resize_img = image.resize((150, 150))
        st.image(resize_img,caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button('Classify'):
            saved_path = save_images_locally(uploaded_image)
            prediction = predict_class(model, uploaded_image, class_indices)
            save_prediction(saved_path, prediction)
            st.success(f'Prediction: {str(prediction)}')

st.subheader("Past Predictions")

past_predictions=get_past_prediction()

if past_predictions:
    for pred in past_predictions:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if os.path.exists(pred["image_path"]):
                st.image(pred["image_path"], width=100, caption="Past Image")
            else:
                st.warning("Image Not Found!")

        with col2:
            st.write(f"**Prediction:** {pred['prediction']}")
            st.markdown("---")

else:
    st.info("No past predictions yet!")

