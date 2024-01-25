from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from predict import proses
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import  StreamingResponse
import uvicorn
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from typing import Generator
from PIL import Image
import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import mysql.connector
import time
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")

# Load your model
model_baru=load_model('effNetV2CAM.h5')
jenis = ['Kawung', 'Megamendung', 'Parang', 'Sekarjagad', 'Truntum']

@app.get("/")
def home(request: Request):
    return print("Hello")

@app.post("/prediksi")
async def predict_image(file: UploadFile = File(...)):
    conf, label = proses(file)
    if label == "Megamendung":
        info = "Megamendung batik comes from Cirebon, West Java. The pattern is cloudy clouds depicted with color gradations. The philosophy of Megamendung batik is a life entirely of change, while its meaning is the eternity of love and affection."
        image = "assets\img\Megamendung.jpg"
    elif label == "Kawung":
        info = "Kawung batik comes from Central Java. The pattern is a kawung fruit depicted in a geometric pattern. The philosophy of Kawung batik is power and prosperity, while its meaning is success and prosperity."
        image = "assets\img\Kawung.jfif"
    elif label == "Parang":
        info = "Parang Batik comes from Solo, Central Java. The pattern is sea waves depicted in geometric patterns. The philosophy of Parang batik is strength and eternity, while its meaning is the spirit of struggle and never giving up."
        image = "assets\img\Parang.jpg"
    elif label == "Sekarjagad":
        info = "Batik Sekar Jagad comes from Yogyakarta. The pattern is the universe depicted with various kinds of flora and fauna. The philosophy of Sekar Jagad batik is harmony and balance of nature, while its meaning is the beauty and wonder of the universe."
        image = "assets\img\Sekarjagad.jpg"
    elif label == "Truntum":
        info = "Truntum batik comes from Yogyakarta. The pattern is a truntum flower depicted in a geometric pattern. The philosophy of Truntum batik is fertility and prosperity, while its meaning is happiness and love."
        image = "assets\img\Truntum.jpg"
    
    hasil = label + " ("+str(f"{conf*100:.2f}") + "%)" + "\n\n" + info + "\n\n"

    return {"Text": hasil, "Image": image, "Label": label}

# python -m uvicorn predictCV:app --reload
if __name__ == '__main__':
    # nanti di cloud run samain juga CONTAINER PORT -> 3000
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3000))) 