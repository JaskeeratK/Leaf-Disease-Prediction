# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
# import json
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import streamlit as st
# import firebase_admin
# from firebase_admin import credentials, firestore
# import pandas as pd
# import uuid

# if "session_id" not in st.session_state:
#     st.session_state["session_id"]=str(uuid.uuid4())


# try:
#     firebase_admin.get_app()
# except ValueError:
#     # cred = credentials.Certificate(r"path/to/firebase-admin-sdk.json")
#     try:
#         firebase_admin.get_app()
#     except ValueError:
#     # Use Streamlit secrets instead of local file
#         firebase_creds = dict(st.secrets["firebase"])
#         cred = credentials.Certificate(firebase_creds)
#     firebase_admin.initialize_app(cred)

# db = firestore.client()

# # UPLOAD_FOLDER='uploads'
# # os.makedirs(UPLOAD_FOLDER,exist_ok=True)

# # def save_images_locally(image_file):
# #     file_path=os.path.join(UPLOAD_FOLDER,image_file.name)
# #     with open(file_path,"wb") as f:
# #         f.write(image_file.getbuffer())
# #     return file_path
# def save_images_locally(image_file):
#     # Return just the filename for reference
#     return image_file.name
    
# def save_prediction(image_path,predicted_class):
#     try:
#         data = {
#             "image_path": image_path,
#             "prediction": predicted_class,
#             "session_id": st.session_state["session_id"]
#         }
#         print("Saving to Firestore:", data)  
#         db.collection("predictions").add(data)
#         print("Prediction saved successfully!")
#     except Exception as e:
#         print("Error saving prediction:", str(e))

# def get_past_prediction():
#     predictions = db.collection("predictions").where("session_id", "==", st.session_state["session_id"]).stream()
#     past_data = []
#     for p in predictions:
#         doc = p.to_dict()
#         print("Fetched Data:", doc) 
#         past_data.append({"image_path": doc["image_path"], "prediction": doc["prediction"]})
    
#     return past_data

# working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = f"{working_dir}/leaf_disease_prediction2.h5"

# model = tf.keras.models.load_model(model_path)

# class_indices = json.load(open(f"{working_dir}/class_idx.json"))


# def load_preprocess(image_path, target_size=(224, 224)):

#     img = Image.open(image_path)
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array.astype('float32') / 255.
#     return img_array

# def predict_class(model, image_file, class_idx):
#     preprocessed_img = load_preprocess(image_file)

#     prediction = model.predict(preprocessed_img)
#     is_diseased=np.max(prediction)>0.5

#     if prediction.size == 0:
#         return "Prediction Error"

#     if is_diseased:
#         predicted_class_index = np.argmax(prediction, axis=1)[0]
#         predicted_class_name = class_idx.get(str(predicted_class_index), "Unknown")
#     else:
#         predicted_class_name=" ☘️ Healthy Leaf ☘️"

#     confidence = np.max(prediction) * 100
#     return predicted_class_name, round(confidence, 2)


# # Streamlit App
# st.title('🌿Leaf Disease Predictor🌿')

# uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     image = Image.open(uploaded_image)
#     col1, col2 = st.columns(2)
#     with col1:
#         resize_img = image.resize((150, 150))
#         st.image(resize_img,caption="Uploaded Image", use_column_width=True)

#     with col2:
#         if st.button('Classify'):
#             saved_path = save_images_locally(uploaded_image)
#             prediction = predict_class(model, uploaded_image, class_indices)
#             save_prediction(saved_path, prediction)
#             st.success(f'Prediction: {str(prediction)}')

# # st.subheader("Past Predictions")

# # past_predictions=get_past_prediction()

# # if past_predictions:
# #     for pred in past_predictions:
# #         col1, col2 = st.columns([1, 2])
        
# #         with col1:
# #             if os.path.exists(pred["image_path"]):
# #                 st.image(pred["image_path"], width=100, caption="Past Image")
# #             else:
# #                 st.warning("Image Not Found!")

# #         with col2:
# #             st.write(f"**Prediction:** {pred['prediction']}")
# #             st.markdown("---")

# # else:
# #     st.info("No past predictions yet!")
# st.subheader("Past Predictions")
# past_predictions = get_past_prediction()
# if past_predictions:
#     df_data = []
#     for pred in past_predictions:
#         df_data.append({
#             "Image Name": pred["image_path"],
#             "Prediction": pred["prediction"]
#         })
#     df = pd.DataFrame(df_data)
#     st.dataframe(df, use_container_width=True)
# else:
#     st.info("No past predictions yet!")
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import uuid

# Initialize session ID
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Initialize Firebase with Streamlit secrets
try:
    firebase_admin.get_app()
except ValueError:
    firebase_creds = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_prediction(image_name, predicted_class):
    """Save prediction to Firestore"""
    try:
        data = {
            "image_name": image_name,
            "prediction": str(predicted_class),
            "session_id": st.session_state["session_id"]
        }
        db.collection("predictions").add(data)
        return True
    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
        return False

def get_past_prediction():
    """Retrieve past predictions from Firestore"""
    try:
        predictions = db.collection("predictions").where(
            "session_id", "==", st.session_state["session_id"]
        ).stream()
        
        past_data = []
        for p in predictions:
            doc = p.to_dict()
            past_data.append({
                "image_name": doc.get("image_name", "Unknown"),
                "prediction": doc.get("prediction", "Unknown")
            })
        return past_data
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return []

@st.cache_resource
def load_model_and_classes():
    """Load model and class indices with caching"""
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/leaf_disease_prediction2.h5"
    model = tf.keras.models.load_model(model_path)
    
    class_indices_path = f"{working_dir}/class_idx.json"
    with open(class_indices_path) as f:
        class_indices = json.load(f)
    
    return model, class_indices

# Load model and classes
model, class_indices = load_model_and_classes()

def load_preprocess(image_file, target_size=(224, 224)):
    """Preprocess uploaded image"""
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_class(model, image_file, class_idx):
    """Predict disease class from image"""
    preprocessed_img = load_preprocess(image_file)
    prediction = model.predict(preprocessed_img)
    
    is_diseased = np.max(prediction) > 0.5
    
    if prediction.size == 0:
        return "Prediction Error", 0.0
    
    if is_diseased:
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_idx.get(str(predicted_class_index), "Unknown")
    else:
        predicted_class_name = "☘️ Healthy Leaf ☘️"
    
    confidence = np.max(prediction) * 100
    return predicted_class_name, round(confidence, 2)

# Streamlit App UI
st.title('🌿 Leaf Disease Predictor 🌿')
st.write("Upload a leaf image to detect diseases using AI-powered classification")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    
    with col1:
        resize_img = image.resize((150, 150))
        st.image(resize_img, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if st.button('🔍 Classify', use_container_width=True):
            with st.spinner('Analyzing leaf...'):
                predicted_class, confidence = predict_class(model, uploaded_image, class_indices)
                
                # Save prediction
                if save_prediction(uploaded_image.name, f"{predicted_class} ({confidence}%)"):
                    st.success(f'**Prediction:** {predicted_class}')
                    st.info(f'**Confidence:** {confidence}%')

# Display past predictions
st.markdown("---")
st.subheader("📊 Past Predictions")

past_predictions = get_past_prediction()

if past_predictions:
    df_data = []
    for pred in past_predictions:
        df_data.append({
            "Image Name": pred["image_name"],
            "Prediction": pred["prediction"]
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No past predictions yet! Upload an image to get started.")



