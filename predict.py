import csv
from PIL import Image
import tensorflow
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import keras
from keras.models import load_model
import mysql.connector
import cv2
import matplotlib.cm as cm


def get_img_array(img_path, size):
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tensorflow.keras.preprocessing.image.img_to_array(img)
    # We add a batch dimension to transform our array into a "batch" of size 1
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tensorflow.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tensorflow.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tensorflow.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tensorflow.newaxis]
    heatmap = tensorflow.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = tensorflow.keras.preprocessing.image.load_img(img_path)
    img = tensorflow.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tensorflow.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
#     display(Image(cam_path))

    return cam_path

def create_bounding_box(img_path, heatmap, threshold=0.5):
    # Load the original image
    original_img = cv2.imread(img_path)

    # Resize the heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Threshold the heatmap to get the areas with high activation
    thresholded_heatmap = np.where(heatmap_resized > threshold, 1.0, 0.0)

    # Find contours in the thresholded heatmap
    contours, _ = cv2.findContours(np.uint8(thresholded_heatmap), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw bounding boxes
    img_with_boxes = original_img.copy()

    # Iterate through contours and draw bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_with_boxes

def proses(file):
    model_baru = load_model('effNetV2CAM.h5')
    jenis = ['Jagung_Blight', 'Jagung_Common_Rust', 'Jagung_Gray_Leaf_Spot', 'Jagung_Healthy', 'Padi_Bacterialblight', 'Padi_Blast', 'Padi_Brownspot', 'Pisang_Cordana', 'Pisang_Healthy', 'Pisang_Pestalotiopsis', 'Pisang_Sigatoka', 'Singkong_Bacterial_Blight', 'Singkong_Brown_Streak_Disease', 'Singkong_Green_Mottle', 'Singkong_Healthy', 'Singkong_Mosaic_Disease', 'Tebu_Healthy', 'Tebu_Mosaic', 'Tebu_RedRot', 'Tebu_Rust', 'Tebu_Yellow']

    # Open and save the image
    image = Image.open(file.file)
    saved_path = f'saved_pictures/{file.filename}'
    image.save(saved_path)

    # Preprocess the image
    img_array = preprocess_input(get_img_array(saved_path, size=(224, 224)))

    # Make prediction using the model
    p = model_baru.predict(img_array)
    kelas = p.argmax(axis=1)[0]
    label = jenis[kelas]
    conf = float(p[0][kelas])

    last_conv_layer_name = "top_conv"

    # Remove last layer's softmax
    model_baru.layers[-1].activation = None

    heatmap = make_gradcam_heatmap(img_array, model_baru, last_conv_layer_name)

    # Create and save the original image with bounding box
    img_with_boxes = create_bounding_box(saved_path, heatmap)
    img_with_boxes_path = f'saved_pictures_bb/{file.filename}_bb.jpg'
    cv2.imwrite(img_with_boxes_path, img_with_boxes)

    # Write to CSV
    with open('predictions.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['Image', 'Class', 'Probability'])
        writer.writerow([file.filename, label, conf])

    return conf, label