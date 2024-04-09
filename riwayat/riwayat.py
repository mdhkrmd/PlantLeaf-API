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

def showRiwayat(nik):
    mydb.connect()
    cursor = mydb.cursor()
    
    if nik is None:
        query = "SELECT * FROM prediksi"
    else:
        query = "SELECT * FROM prediksi WHERE nik = '" + nik + "' ORDER BY id DESC"
    
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return [
            {
                "id": row[0],
                "nik": row[1],
                "tanggal": row[2],
                "penyakit": row[3],
                "nama": row[4],
                "gambar": row[5],
                "catatan": row[6]
            } for row in result
        ]
    
    except Exception as e:
        mydb.close()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return [response] if nik is None else response