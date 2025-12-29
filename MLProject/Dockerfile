# Gunakan base image python yang ringan
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy file requirements (kita generate manual dari conda.yaml logic)
RUN pip install pandas scikit-learn mlflow==2.19.0 dagshub matplotlib seaborn

# Copy seluruh file di folder MLProject ke dalam image
COPY . /app

# Command default (opsional, bisa untuk serving nanti)
CMD ["python", "modelling.py"]