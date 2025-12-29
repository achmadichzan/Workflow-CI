FROM python:3.12.7-slim

WORKDIR /app

RUN pip install pandas scikit-learn mlflow==2.19.0 dagshub matplotlib seaborn

COPY . /app

CMD ["python", "modelling.py"]