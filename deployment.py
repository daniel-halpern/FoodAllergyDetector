import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import streamlit as st
from PIL import Image
import ollama

def main(picture = None):
    model = load_model()
    size = 128
    # Draws what always is on screen
    draw()

    # Take picture
    if picture is None:
        picture = st.camera_input("Use camera to take a picture")
        picture = st.file_uploader("Upload an image")

    if picture is not None:
        img = Image.open(picture)
        predictions = predict(model, img, size)

    if st.button("Ask why the sky is blue"):
        response = ask_ollama()
        st.write(response['message']['content'])

def predict(model, img, size):
    img = img.resize((size, size))
    img = np.array(img)
    img = img.reshape((1, size, size, 3))
    predictions = model.predict(img)
    class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10', 'class11', 'class12', 'class13', 'class14', 'class15', 'class16', 'class17', 'class18', 'class19', 'class20', 'class21', 'class22', 'class23', 'class24', 'class25', 'class26', 'class27', 'class28', 'class29']
    top3_class_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_classes = [class_names[i] for i in top3_class_indices]
    print(f'The model predicts that the image is most likely of class: {top3_classes[0]}, second most likely: {top3_classes[1]}, third most likely: {top3_classes[2]}')    
    return predictions

def ask_ollama():
    response = ollama.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue? Respond with 1 sentence.',
    },
    ])
    print(response['message']['content'])
    return response

def draw():
    # Streamlit setup
    st.set_page_config(
        page_title="Food Allergy Detector",
        page_icon="📍",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Show where you want community improvements!"
        }
    )
    st.markdown("## A tool to detect if a food is safe to eat")

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('my_model.keras')

main()