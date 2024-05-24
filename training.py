import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import numpy as np

size = 128

def load_data(image_dir, json_file):
    with open(os.path.join(image_dir, json_file)) as f:
        labels_data = json.load(f)

    # Convert list of dictionaries to dictionary
    labels = {str(item['image_id']): item['category_id'] for item in labels_data['annotations']}

    imageList = []
    image_labels = []
    names = {category['id']: category['name'] for category in labels_data['categories']}

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images = labels_data["images"]
            for item in images:
                if item["file_name"] in filename:
                    id = item["id"]
                    img = Image.open(os.path.join(image_dir, filename))
                    img = img.resize((size, size)) 
                    imageList.append(np.array(img))
                    # Check if id exists in labels before trying to access it
                    image_labels.append(labels[str(id)])  

    return np.array(imageList), np.array(image_labels), names

test_images, test_labels, names = load_data('test', '_annotations.coco.json')
train_images, train_labels, names = load_data('train', '_annotations.coco.json')
valid_images, valid_data, names = load_data('valid', '_annotations.coco.json')

# Load the VGG16 model but exclude the top layer, which is the classification layer
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(weights=weights_path, include_top=False, input_shape=(size, size, 3))
# Freeze the layers in the base model so they're not trained
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the base model
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(size*2, activation='relu'))
model.add(layers.Dense(len(names), activation='softmax'))  # Use softmax for multi-class classification

# Now you can compile and train your model as before
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))

model.save('my_model.keras')
