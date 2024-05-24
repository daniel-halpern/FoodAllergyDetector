import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('my_model.keras')
size = 128

# Function to preprocess the image
#def preprocess_image(image):
    # Add your image preprocessing code here
    # ...

    #return preprocessed_image

# Function to make predictions
def predict(image):
    # Preprocess the image
    preprocessed_image = image #preprocess_image(image)

    # Make predictions using the model
    predictions = model.predict(preprocessed_image)

    return predictions

# Main function
from tensorflow.keras.applications.vgg16 import preprocess_input

def main():
    # Load the image
    image = tf.keras.preprocessing.image.load_img('croissant.jpg', target_size=(size, size))

    # Convert the image to a numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Reshape the image array
    # Reshape the image array to include a batch dimension
    image_array = image_array.reshape((1, 128, 128, 3))

    # Make predictions using the model
    predictions = model.predict(image_array)

    # Assuming you have a list of class names in the same order as the model's output
    class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10', 'class11', 'class12', 'class13', 'class14', 'class15', 'class16', 'class17', 'class18', 'class19', 'class20', 'class21', 'class22', 'class23', 'class24', 'class25', 'class26', 'class27', 'class28', 'class29']

    # Get the indices of the classes with the three highest probabilities
    top3_class_indices = np.argsort(predictions[0])[-3:][::-1]

    # Get the names of the classes
    top3_classes = [class_names[i] for i in top3_class_indices]

    print(f'The model predicts that the image is most likely of class: {top3_classes[0]}, second most likely: {top3_classes[1]}, third most likely: {top3_classes[2]}')    
    return predictions

# Run the main function
if __name__ == '__main__':
    main()