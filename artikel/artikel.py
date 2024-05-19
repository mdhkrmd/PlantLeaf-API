from fastapi import FastAPI
import mysql.connector
from pydantic import BaseModel

app = FastAPI()

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="plantleaf"
)

def get_artikel():
    mydb.connect()
    cursor = mydb.cursor()
    try:    
        cursor.execute("SELECT * FROM artikel ORDER BY id DESC")
        result = cursor.fetchall()
        return [
            {
                "id": row[0],
                "judul": row[1],
                "penulis": row[2],
                "foto": row[3],
                "link": row[4]
            } for row in result
        ]
    
    except Exception as e:
        mydb.close()
        response = {
            'status': 'error',
            'message': 'Terjadi kesalahan saat mengambil data',
            'error': str(e)
        }
        return [response]