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

async def showResult(label):
    cursor = mydb.cursor()
    
    if label is None:
        query = "SELECT * FROM penyakit"
    else:
        query = "SELECT * FROM penyakit WHERE label_penyakit = '" + label + "'"
    
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return [
            {
                "id": row[0],
                "label_penyakit": row[1],
                "tentang_penyakit": row[2],
                "gejala": row[3],
                "penanganan": row[4]
            } for row in result
        ]
    
    except Exception as e:
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return [response] if label is None else response