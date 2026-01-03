FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY test_data/ ./test_data/

ENTRYPOINT ["python", "src/model_predict.py"]

CMD ["--help"]
