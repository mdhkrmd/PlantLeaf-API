from fastapi import FastAPI, File, UploadFile, Request
from fastapi.param_functions import Form
import uvicorn
import os
from tensorflow.keras.models import load_model
from pydantic import BaseModel

from predict import proses, proses_upload, proses_upload_opsi
from artikel.artikel import get_artikel
from auth.register import register, showUsers
from auth.login import login
from auth.forgot import forgot
from auth.updateProfil import updateProf
from prediksi.result import showResult
from tanaman.show import showTanaman
from riwayat.riwayat import showRiwayat

app = FastAPI()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth/key.json"

# Load your model
model_baru=load_model('i-WO Singkong - Split-98.65.h5')
jenis = ['Jagung_Blight', 'Jagung_Common_Rust', 'Jagung_Gray_Leaf_Spot', 'Jagung_Healthy',
         'Kentang__Early_blight', 'Kentang__Healthy', 'Kentang__Late_blight', 
         'Mangga_Anthracnose', 'Mangga_Bacterial_Canker', 'Mangga_Gall_Midge', 'Mangga_Healthy', 'Mangga_Powdery_Mildew', 'Mangga_Sooty_Mould', 
         'Padi_Bacterialblight', 'Padi_Blast', 'Padi_Brownspot', 'Padi_Healthy', 
         'Pisang_Cordana', 'Pisang_Healthy', 'Pisang_Pestalotiopsis', 'Pisang_Sigatoka', 
         'Tebu_Healthy', 'Tebu_Mosaic', 'Tebu_RedRot', 'Tebu_Rust', 'Tebu_Yellow']

@app.get("/")
def home(request: Request):
    return "Hello"


#=============================================================================
# Auth

@app.get("/users")
def get_users_route(nik: str = None):
    return showUsers(nik)

@app.post("/register")
async def post_register_route(request: Request):
    return await register(request)

@app.post("/login")
async def post_login_route(request: Request):
    return await login(request)

@app.post("/forgot")
async def post_forgot_route(request: Request):
    return await forgot(request)

@app.post("/update")
async def post_update_route(request: Request):
    return await updateProf(request)

#=============================================================================
# Predict

class UserData(BaseModel):
    nik: str
    nama: str

@app.post("/prediksi")
async def predict_image(file: UploadFile = File(...)):
    return proses(file, model_baru, jenis)

@app.get("/result")
async def get_result_route(label: str = None):
    return await showResult(label)

@app.post("/prediksiupload")
async def predict_image_upload(file: UploadFile = File(...), nik: str = Form(...), nama: str = Form(...)):
    return proses_upload(file, model_baru, jenis, nik, nama)

@app.post("/prediksiuploadopsi")
async def predict_image_upload_opsi(file: UploadFile = File(...), nik: str = Form(...), nama: str = Form(...), bb: str = Form(...)):
    return proses_upload_opsi(file, model_baru, jenis, nik, nama, bb)


#=============================================================================
# Artikel
@app.get("/artikel")
def get_artikel_route():
    return get_artikel()

#=============================================================================
# Tanaman
@app.get("/tanaman")
def get_tanaman_route(nama: str = None):
    return showTanaman(nama)

#=============================================================================
# Riwayat
@app.get("/riwayat")
def get_riwayat_route(nik: str = None):
    return showRiwayat(nik)

# python -m uvicorn app:app --reload
if __name__ == '__main__':
    # nanti di cloud run samain juga CONTAINER PORT -> 3000
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)), reload=True) 