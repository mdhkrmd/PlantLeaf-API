import csv
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import keras
from keras.models import load_model
import mysql.connector

def proses(file):
    model_baru=load_model('effNetV2.h5')
    jenis = ['Kawung', 'Megamendung', 'Parang', 'Sekarjagad', 'Truntum']

    image = Image.open(file.file)
    image.save(f'saved_pictures/{file.filename}')

    image = preprocess_input(image)
    image = image.convert('RGB')
    image = image.resize((224,224))
    image = np.asarray(image)
    image = np.expand_dims(image,0)

    p = model_baru.predict(image)
    kelas = p.argmax(axis = 1)[0]
    label = jenis[kelas]
    conf = p[0][kelas]

    # Write to CSV
    with open('predictions.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['Image', 'Class', 'Probability'])
        writer.writerow([file.filename, label, conf])
        
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="batikclf"
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO predictresult (gambar, hasil, prob) VALUES (%s, %s, %s)"
    val = (file.filename, label, float(conf))
    mycursor.execute(sql, val)

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")

    return conf, label