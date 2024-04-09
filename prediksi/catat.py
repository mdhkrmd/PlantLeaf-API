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

async def updateCatatan(request: Request):
    data = await request.json()  # Use await here
    id_catatan = data['id']
    catatan = data['catatan']
    
    # Koneksi MySQL
    mydb.connect()
    cursor = mydb.cursor()

    try:
        cursor.execute("SELECT * FROM prediksi WHERE id = %s", (int(id_catatan),))
        result = cursor.fetchone()

        if not result:
            mydb.close()
            response = {
                'status': 'error',
                'message': 'Prediksi tidak terdaftar'
            }
            return response
        else:
            query = """
                        UPDATE prediksi 
                        SET 
                            catatan = %s
                        WHERE id = %s
                    """
            cursor.execute(query, (catatan, int(id_catatan)))
            mydb.commit()
            mydb.close()
            response = {
                'status': 'success',
                'message': 'Berhasil menambahkan catatan'
            }

            return response
    except Exception as e:
        mydb.rollback()
        mydb.close()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan',
            'error': str(e)
        }
        return response