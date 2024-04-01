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
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from io import BytesIO
from fastapi import FastAPI, Request
import mysql.connector
from pydantic import BaseModel
import base64
import mimetypes
from google.cloud import storage

app = FastAPI()
bucket_name = "www.skinnie.my.id"

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="plantleaf"
)


# def get_img_array(img_path, size):
#     img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=size)
#     array = tensorflow.keras.preprocessing.image.img_to_array(img)
#     # We add a batch dimension to transform our array into a "batch" of size 1
#     array = np.expand_dims(array, axis=0)
#     return array

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     # First, we create a model that maps the input image to the activations
#     # of the last conv layer as well as the output predictions
#     grad_model = tensorflow.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     # Then, we compute the gradient of the top predicted class for our input image
#     # with respect to the activations of the last conv layer
#     with tensorflow.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tensorflow.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     # This is the gradient of the output neuron (top predicted or chosen)
#     # with regard to the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)

#     # This is a vector where each entry is the mean intensity of the gradient
#     # over a specific feature map channel
#     pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the top predicted class
#     # then sum all the channels to obtain the heatmap class activation
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tensorflow.newaxis]
#     heatmap = tensorflow.squeeze(heatmap)

#     # For visualization purpose, we will also normalize the heatmap between 0 & 1
#     heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
#     # Load the original image
#     img = tensorflow.keras.preprocessing.image.load_img(img_path)
#     img = tensorflow.keras.preprocessing.image.img_to_array(img)

#     # Rescale heatmap to a range 0-255
#     heatmap = np.uint8(255 * heatmap)

#     # Use jet colormap to colorize heatmap
#     jet = cm.get_cmap("jet")

#     # Use RGB values of the colormap
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap]

#     # Create an image with RGB colorized heatmap
#     jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
#     jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
#     jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)

#     # Superimpose the heatmap on original image
#     superimposed_img = jet_heatmap * alpha + img
#     superimposed_img = tensorflow.keras.preprocessing.image.array_to_img(superimposed_img)

#     # Save the superimposed image
#     superimposed_img.save(cam_path)

#     # Display Grad CAM
# #     display(Image(cam_path))

#     return cam_path

def create_bounding_box(img_path, heatmap, threshold=0.8):
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

def VizGradCAMBBfix(model, image, interpolant=0.5, plot_results=True, threshold=0.8):
    assert (interpolant > 0 and interpolant < 1), "Heatmap Interpolation Must Be Between 0 - 1"

    original_img = np.asarray(image, dtype=np.float32)
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)
    prediction_idx = np.argmax(prediction)

    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tensorflow.keras.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    with tensorflow.GradientTape() as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        loss = prediction[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]
    weights = tensorflow.reduce_mean(gradients[0], axis=(0, 1))
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)

    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    activation_map = cv2.resize(activation_map.numpy(),
                                (original_img.shape[1],
                                 original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = np.uint8(255 * activation_map)

    # Enlarge plot
    # plt.rcParams["figure.dpi"] = 100

    if plot_results:
        overlaid_img = np.uint8(original_img * interpolant)

        # Apply thresholding to the heatmap
        thresholded_heatmap = cv2.threshold(activation_map, 255 * threshold, 255, cv2.THRESH_BINARY)[1]

        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(thresholded_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes on the original image
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlaid_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # plt.imshow(overlaid_img)
        overlaid_img_bgr = cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR)
        return overlaid_img_bgr
    else:
        return activation_map  # Return activation map without heatmap

def proses(file, model_baru, jenis):
    # Read the image into memory
    image_data = BytesIO(file.file.read())
    image = Image.open(image_data)
    
    # Optional: Save the image if necessary
    saved_path = f'saved_pictures/{file.filename}'
    image.save(saved_path)

    # Preprocess the image
    img_array = preprocess_input(np.array(image.resize((224, 224))))

    # Ensure img_array is 4D
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the model
    p = model_baru.predict(img_array)
    kelas = p.argmax(axis=1)[0]
    label = jenis[kelas]
    conf = float(p[0][kelas])
    
    mydb.connect()
    cursor = mydb.cursor()
    query = "SELECT * FROM penyakit WHERE label_penyakit = '" + label + "'"
    
    # # Remove last layer's softmax
    model_baru.layers[-1].activation = None
    
    # # Convert PIL image to numpy array (OpenCV format)
    test_img = np.array(image.convert('RGB'))  # Ensure the image is in RGB format
    
    # # Resize the image to the required input size of the model
    resized_test_img = cv2.resize(test_img, (224, 224))
    
    daun_healthy = ['Jagung_Healthy','Mangga_Healthy','Padi_Healthy','Pisang_Healthy','Kentang__Healthy', ]
    
    if label in daun_healthy:
        _, buffer = cv2.imencode('.jpg', resized_test_img)
        io_buf = BytesIO(buffer)
        # Encode the image buffer to Base64
        base64_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
    else:
        # # Run Grad-CAM and bounding box drawing
        bbpic = VizGradCAMBBfix(model_baru, resized_test_img, plot_results=True)
        # Save the image to a buffer rather than a file
        _, buffer = cv2.imencode('.jpg', bbpic)
        io_buf = BytesIO(buffer)
        # Encode the image buffer to Base64
        base64_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        # # Create and save the original image with bounding box
        if not os.path.exists('saved_pictures_bb'):
            os.makedirs('saved_pictures_bb')
        img_with_boxes_path = f'saved_pictures_bb/{file.filename}_bb.jpg'
        cv2.imwrite(img_with_boxes_path, bbpic)
    
    try:
        cursor.execute(query)
        row = cursor.fetchone()
        return {
                "gambar":  base64_image,
                "conf": conf,
                "label": label,
                "id": row[0],
                "label_penyakit": row[1],
                "tentang_penyakit": row[2],
                "gejala": row[3],
                "penanganan": row[4]
            }
    
    except Exception as e:
        mydb.close()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return response

    # # Write to CSV
    # csv_file = 'predictions.csv'
    # with open(csv_file, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     if f.tell() == 0:
    #         writer.writerow(['Image', 'Class', 'Probability'])
    #     writer.writerow([file.filename, label, conf])

    # # Optionally: Close the file if it's no longer needed
    # file.file.close()
    
def upload_image_to_storage_base64(image_base64, filename):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    directory = "fotoResults"
    blob_path = f"{directory}/{filename}"
    file_blob = bucket.blob(blob_path)

    # Decode base64 image data
    image_data = base64.b64decode(image_base64)

    # Upload to Google Cloud Storage
    file_blob.upload_from_string(image_data, content_type='image/jpeg')
    print(f"Image {filename} uploaded to Google Cloud Storage.")

    image_url = f"https://storage.googleapis.com/{bucket_name}/fotoResults/{filename}"
    return image_url

def proses_upload(file, model_baru, jenis, nik, nama):
    # Read the image into memory
    image_data = BytesIO(file.file.read())
    image = Image.open(image_data)
    
    # Optional: Save the image if necessary
    saved_path = f'saved_pictures/{file.filename}'
    image.save(saved_path)

    # Preprocess the image
    img_array = preprocess_input(np.array(image.resize((224, 224))))

    # Ensure img_array is 4D
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the model
    p = model_baru.predict(img_array)
    kelas = p.argmax(axis=1)[0]
    label = jenis[kelas]
    conf = float(p[0][kelas])
    
    mydb.connect()
    cursor = mydb.cursor()
    query = "SELECT * FROM penyakit WHERE label_penyakit = '" + label + "'"
    
    # # Remove last layer's softmax
    model_baru.layers[-1].activation = None
    
    # # Convert PIL image to numpy array (OpenCV format)
    test_img = np.array(image.convert('RGB'))  # Ensure the image is in RGB format
    
    # # Resize the image to the required input size of the model
    resized_test_img = cv2.resize(test_img, (224, 224))
    
    daun_healthy = ['Jagung_Healthy','Mangga_Healthy','Padi_Healthy','Pisang_Healthy','Kentang__Healthy', ]
    
    if label in daun_healthy:
        _, buffer = cv2.imencode('.jpg', resized_test_img)
        io_buf = BytesIO(buffer)
        # Encode the image buffer to Base64
        base64_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
    else:
        # # Run Grad-CAM and bounding box drawing
        bbpic = VizGradCAMBBfix(model_baru, resized_test_img, plot_results=True)
        # Save the image to a buffer rather than a file
        _, buffer = cv2.imencode('.jpg', bbpic)
        io_buf = BytesIO(buffer)
        # Encode the image buffer to Base64
        base64_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        # # Create and save the original image with bounding box
        if not os.path.exists('saved_pictures_bb'):
            os.makedirs('saved_pictures_bb')
        img_with_boxes_path = f'saved_pictures_bb/{file.filename}_bb.jpg'
        cv2.imwrite(img_with_boxes_path, bbpic)
    
    url = upload_image_to_storage_base64(base64_image, file.filename)
    cursor2 = mydb.cursor()
    query2 = "INSERT INTO prediksi (nik, penyakit, nama, gambar) VALUES (%s,%s,%s,%s)"
    val = (nik, label, nama, url)
    cursor2.execute(query2, val)
    mydb.commit()
    
    try:
        cursor.execute(query)
        row = cursor.fetchone()
        return {
                "gambar":  base64_image,
                "conf": conf,
                "label": label,
                "id": row[0],
                "label_penyakit": row[1],
                "tentang_penyakit": row[2],
                "gejala": row[3],
                "penanganan": row[4]
            }
    
    except Exception as e:
        mydb.close()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return response
    
def proses_upload_opsi(file, model_baru, jenis, nik, nama, bb):
    # Read the image into memory
    image_data = BytesIO(file.file.read())
    image = Image.open(image_data)
    
    # Optional: Save the image if necessary
    saved_path = f'saved_pictures/{file.filename}'
    image.save(saved_path)

    # Preprocess the image
    img_array = preprocess_input(np.array(image.resize((456,456))))

    # Ensure img_array is 4D
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the model
    p = model_baru.predict(img_array)
    kelas = p.argmax(axis=1)[0]
    label = jenis[kelas]
    conf = float(p[0][kelas])
    
    mydb.connect()
    cursor = mydb.cursor()
    query = "SELECT * FROM penyakit WHERE label_penyakit = '" + label + "'"
    
    # # Remove last layer's softmax
    model_baru.layers[-1].activation = None
    
    # # Convert PIL image to numpy array (OpenCV format)
    test_img = np.array(image.convert('RGB'))  # Ensure the image is in RGB format
    
    # # Resize the image to the required input size of the model
    resized_test_img = cv2.resize(test_img, (456,456))
    
    daun_healthy = ['Jagung_Healthy','Mangga_Healthy','Padi_Healthy','Pisang_Healthy','Kentang__Healthy', ]
    
    if label in daun_healthy or bb == 'no':
        _, buffer = cv2.imencode('.jpg', resized_test_img)
        io_buf = BytesIO(buffer)
        # Encode the image buffer to Base64
        base64_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
    else:
        # # Run Grad-CAM and bounding box drawing
        bbpic = VizGradCAMBBfix(model_baru, resized_test_img, plot_results=True)
        # Save the image to a buffer rather than a file
        _, buffer = cv2.imencode('.jpg', bbpic)
        io_buf = BytesIO(buffer)
        # Encode the image buffer to Base64
        base64_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        # # Create and save the original image with bounding box
        if not os.path.exists('saved_pictures_bb'):
            os.makedirs('saved_pictures_bb')
        img_with_boxes_path = f'saved_pictures_bb/{file.filename}_bb.jpg'
        cv2.imwrite(img_with_boxes_path, bbpic)
    
    url = upload_image_to_storage_base64(base64_image, file.filename)
    cursor2 = mydb.cursor()
    query2 = "INSERT INTO prediksi (nik, penyakit, nama, gambar) VALUES (%s,%s,%s,%s)"
    val = (nik, label, nama, url)
    cursor2.execute(query2, val)
    mydb.commit()
    
    try:
        cursor.execute(query)
        row = cursor.fetchone()
        return {
                "gambar":  base64_image,
                "conf": conf,
                "label": label,
                "id": row[0],
                "label_penyakit": row[1],
                "tentang_penyakit": row[2],
                "gejala": row[3],
                "penanganan": row[4]
            }
    
    except Exception as e:
        mydb.close()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return response