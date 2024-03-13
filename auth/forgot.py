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

async def forgot(request: Request):
    data = await request.json()  # Use await here
    username = data['username']
    new_password = data['new_password']
    
    # Koneksi MySQL
    cursor = mydb.cursor()
    
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        if not result:
            mydb.rollback()
            response = {
                'status': 'error',
                'message': 'Username tidak terdaftar'
            }
            return response
        else:
            query = "UPDATE users SET `password` = '" + new_password + "' WHERE `username` = '" + username + "'"
            cursor.execute(query)
            mydb.commit()

            response = {
                'status': 'success',
                'message': 'Berhasil ganti password'
            }
            return response
    
    except Exception as e:
        mydb.rollback()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan',
            'error': str(e)
        }
        return response