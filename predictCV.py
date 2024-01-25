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

# Load your model
model_baru=load_model('effNetV2CAM.h5')
jenis = ['Jagung_Blight', 'Jagung_Common_Rust', 'Jagung_Gray_Leaf_Spot', 'Jagung_Healthy', 'Padi_Bacterialblight', 'Padi_Blast', 'Padi_Brownspot', 'Pisang_Cordana', 'Pisang_Healthy', 'Pisang_Pestalotiopsis', 'Pisang_Sigatoka', 'Singkong_Bacterial_Blight', 'Singkong_Brown_Streak_Disease', 'Singkong_Green_Mottle', 'Singkong_Healthy', 'Singkong_Mosaic_Disease', 'Tebu_Healthy', 'Tebu_Mosaic', 'Tebu_RedRot', 'Tebu_Rust', 'Tebu_Yellow']

@app.get("/")
def home(request: Request):
    return "Hello"

@app.post("/prediksi")
async def predict_image(file: UploadFile = File(...)):
    conf, label = proses(file)

    return {"Conf": str(f"{conf*100:.2f}"), "Label": label}

# python -m uvicorn predictCV:app --reload
if __name__ == '__main__':
    # nanti di cloud run samain juga CONTAINER PORT -> 3000
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3000))) 