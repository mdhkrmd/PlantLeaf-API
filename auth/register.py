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

async def showUsers(nik):
    cursor = mydb.cursor()
    
    if nik is None:
        query = "SELECT * FROM users"
    else:
        query = "SELECT * FROM users WHERE nik = '" + nik + "'"
    
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return [
            {
                "id": row[0],
                "username": row[1],
                "password": row[2],
                "nik": row[3],
                "nama": row[4]
            } for row in result
        ]
    
    except Exception as e:
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return [response] if nik is None else response

async def register(request: Request):
    data = await request.json()  # Use await here
    username = data['username']
    password = data['password']
    nik = data['nik']
    nama = data['nama']
    
    cursor = mydb.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result:
            response = {
                'status': 'error',
                'message': 'Username sudah terdaftar'
            }
            return response
    
        sql = "INSERT INTO users (username, password, nik, nama) VALUES (%s,%s,%s,%s)"
        val = (username, password, nik, nama)
        cursor.execute(sql, val)
        mydb.commit()
        return {"message": "Akun Berhasil ditambah"}
    
    except Exception as e:
        mydb.rollback()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat membuat akun',
            'error': str(e)
        }
        return response