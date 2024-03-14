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

async def updateProf(request: Request):
    data = await request.json()  # Use await here
    nik = data['nik']
    nama = data['nama']
    
    # Koneksi MySQL
    cursor = mydb.cursor()

    try:
        cursor.execute("SELECT * FROM users WHERE nik = %s", (nik,))
        result = cursor.fetchone()

        if not result:
            mydb.rollback()
            response = {
                'status': 'error',
                'message': 'NIK tidak terdaftar'
            }
            return response
        else:
            query = """
                        UPDATE users 
                        SET 
                            `nama` = %s
                        WHERE `nik` = %s
                    """
            cursor.execute(query, (nama, nik))
            mydb.commit()
            mydb.rollback()
            response = {
                'status': 'success',
                'message': 'Berhasil ganti data'
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