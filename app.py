from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import os
from tensorflow.keras.models import load_model
from pydantic import BaseModel

from predict import proses
from artikel.artikel import get_artikel
from auth.register import register, showUsers
from auth.login import login
from auth.forgot import forgot

app = FastAPI()

# Load your model
# model_baru=load_model('i-WO Singkong - Split-98.65.h5')
# jenis = ['Jagung_Blight', 'Jagung_Common_Rust', 'Jagung_Gray_Leaf_Spot', 'Jagung_Healthy', 
#          'Kentang__Early_blight', 'Kentang__Healthy', 'Kentang__Late_blight', 
#          'Mangga_Anthracnose', 'Mangga_Bacterial_Canker', 'Mangga_Gall_Midge', 'Mangga_Healthy', 'Mangga_Powdery_Mildew', 'Mangga_Sooty_Mould', 
#          'Padi_Bacterialblight', 'Padi_Blast', 'Padi_Brownspot', 'Padi_Healthy', 
#          'Pisang_Cordana', 'Pisang_Healthy', 'Pisang_Pestalotiopsis', 'Pisang_Sigatoka', 
#          'Tebu_Healthy', 'Tebu_Mosaic', 'Tebu_RedRot', 'Tebu_Rust', 'Tebu_Yellow']

@app.get("/")
def home(request: Request):
    return "Hello"


#=============================================================================
# Auth
class authreq(BaseModel):
    id: int
    username: str
    password: str
    nik: str
    nama: str

@app.get("/users")
async def get_users_route(nik: str = None):
    return await showUsers(nik)

@app.post("/register")
async def post_register_route(request: Request):
    return await register(request)

@app.post("/login")
async def post_login_route(request: Request):
    return await login(request)

@app.post("/forgot")
async def post_forgot_route(request: Request):
    return await forgot(request)


#=============================================================================
# Predict
@app.post("/prediksi")
async def predict_image(file: UploadFile = File(...)):
    conf, label = proses(file)

    return {"Conf": str(f"{conf*100:.2f}"), "Label": label}


#=============================================================================
# Artikel
@app.get("/artikel")
def get_artikel_route():
    return get_artikel()

# python -m uvicorn predictCV:app --reload
if __name__ == '__main__':
    # nanti di cloud run samain juga CONTAINER PORT -> 3000
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3000))) 