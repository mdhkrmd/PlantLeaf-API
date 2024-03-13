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

async def login(request: Request):
    data = await request.json()  # Use await here
    username = data['username']
    password = data['password']
    
    # Koneksi MySQL
    cursor = mydb.cursor()

    try:
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        result = cursor.fetchone()

        if result:
            nik = result[3]
            nama = result[4]

            response = {
                'status': 'success',
                'message': 'Login berhasil',
                'nik': nik,
                'nama': nama
            }

            return response
        else:
            mydb.rollback()
            response = {
                'status': 'error',
                'message': 'Username atau Password Salah'
            }
            return response
    
    except Exception as e:
        mydb.rollback()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat login',
            'error': str(e)
        }
        return response