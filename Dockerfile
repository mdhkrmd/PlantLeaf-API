# Menggunakan base image yang sudah terinstall Python
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependencies menggunakan pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh konten aplikasi ke dalam container
COPY . .

# Menjalankan command untuk menjalankan aplikasi saat container berjalan
# CMD ["python", "-m", "uvicorn", "predictCV:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["python", "predictCV.py"]

# DEPLOY TO CLOUD RUN:
# docker build -t batikclassification .
# docker tag batikclassification gcr.io/skinnie-project/batikclassification
# docker push gcr.io/skinnie-project/batikclassification