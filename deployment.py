import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import streamlit as st
from PIL import Image
import ollama
from langchain.llms import Ollama
llm = Ollama(model="llama3")

def main(picture = None, food = ""):
    model = load_model()
    size = 128
    # Draws what always is on screen
    draw()

    # Take picture
    if picture is None:
        #picture = st.camera_input("Use camera to take a picture")
        picture = st.file_uploader("Upload an image")

    if picture is not None:
        img = Image.open(picture)
        st.image(img)
        predictions = predict(model, img, size)
        print(predictions, type(predictions))
        selection_list = predictions + ["Other"]
        food = st.selectbox("Which of these are your food?", selection_list)
        if "Other" in food:
            food = st.text_input("Please specify your food:")
            print(food)
    if food != "":
        allergies = ["Peanuts", "Tree nuts", "Milk", "Eggs", "Wheat", "Soy", "Fish", "Shellfish", "Other"]
        selected_allergies = st.multiselect("Select your allergies:", allergies)

        if "Other" in selected_allergies:
            other_allergy = st.text_input("Please specify your other allergies:")
            selected_allergies.remove("Other")
            allergies_string = ', '.join(selected_allergies)
            if other_allergy:
                if selected_allergies == []:
                    allergies_string = other_allergy
                else:
                    allergies_string += ', ' + other_allergy
        else:
            allergies_string = ', '.join(selected_allergies)
        print(allergies_string)
        if st.button("Ask if this food is safe"):
            if allergies_string == '':
                st.write("Safe!")
                # NO PROMPT, just write it is safe
            else:
                prompt = f"""You are an assistant for people with food allergies. Your job is to take a food and determine whether
                has specified allergies in it. Never make assumptions that this food was prepared in a allergy safe space. Assume 
                the food is prepared with normal ingredients. Is the *average* {food} safe to eat if you have a {allergies_string} allergy? Write 1 short sentence."""
                response = llm.predict(prompt)
                #response = ask_ollama(prompt)
                st.write(response)
    

def predict(model, img, size):
    img = img.resize((size, size))
    img = np.array(img)
    img = img.reshape((1, size, size, 3))
    predictions = model.predict(img)
    class_names = ['Japanese tofu and vegetable chowder', 'beef curry', 'beef steak',
                    'boiled chicken and vegetables', 'chinese soup', 'croissant', 
                    'croquette', 'egg roll', 'egg sunny-side up', 'french fries', 
                    'fried chicken', 'fried fish', 'fried noodle', 'fried rice', 
                    'green salad', 'grilled eggplant', 'hamburger', 'hot dog',
                    'macaroni salad', 'miso soup', 'pizza', 'potato salad',
                    'raisin bread', 'rice', 'roast chicken', 'sandwiches', 'sausage',
                    'sauteed spinach']
    top3_class_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_classes = [class_names[i-1] for i in top3_class_indices]
    print(f'The model predicts that the image is most likely of class: {top3_classes[0]}, second most likely: {top3_classes[1]}, third most likely: {top3_classes[2]}')    
    return top3_classes

def ask_ollama(prompt):
    
    response = ollama.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    print(response['message']['content'])
    return response['message']['content']

def draw():
    # Streamlit setup
    st.set_page_config(
        page_title="Food Allergy Detector",
        page_icon="🍕",
        initial_sidebar_state="expanded",
    )
    st.markdown("## A tool to detect if a food is safe to eat")

def load_model():
    return tf.keras.models.load_model('my_model.keras')

main()