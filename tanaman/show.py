from fastapi import FastAPI, Request
import mysql.connector
from pydantic import BaseModel

app = FastAPI()

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="plantleaf"
)

def showTanaman(nama):
    cursor = mydb.cursor()
    
    if nama is None:
        query = "SELECT * FROM tanaman"
    else:
        query = "SELECT * FROM tanaman WHERE nama_tanaman = '" + nama + "'"
    
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return [
            {
                "id": row[0],
                "nama_tanaman": row[1],
                "tentang": row[2],
                "merawat": row[3],
                "gambar": row[4]
            } for row in result
        ]
    
    except Exception as e:
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return [response] if nama is None else response