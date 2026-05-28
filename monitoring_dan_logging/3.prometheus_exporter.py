from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
import time
import pandas as pd
import numpy as np
import pickle
import mlflow.sklearn
import os

app = Flask(__name__)

# ── Metrics ────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    'model_request_count_total',
    'Total jumlah request ke model',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Latency request dalam detik',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)
PREDICTION_VALUE = Gauge(
    'model_prediction_value',
    'Nilai prediksi terakhir dari model'
)
PREDICTION_COUNT = Counter(
    'model_prediction_count_total',
    'Total jumlah prediksi',
    ['predicted_class']
)
INPUT_FEATURE = Gauge(
    'model_input_feature_value',
    'Nilai input fitur',
    ['feature_name']
)
MODEL_ERROR_COUNT = Counter(
    'model_error_count_total',
    'Total error saat inferensi'
)
REQUEST_IN_PROGRESS = Gauge(
    'model_requests_in_progress',
    'Jumlah request yang sedang diproses'
)
PREDICTION_CONFIDENCE = Gauge(
    'model_prediction_confidence',
    'Confidence score prediksi tertinggi'
)
RESPONSE_SIZE = Summary(
    'model_response_size_bytes',
    'Ukuran response dalam bytes'
)
UPTIME = Gauge(
    'model_server_uptime_seconds',
    'Uptime server dalam detik'
)

START_TIME = time.time()

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_IN_PROGRESS.inc()
    start = time.time()
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Log input features
        for col in df.columns:
            INPUT_FEATURE.labels(feature_name=col).set(float(df[col].values[0]))

        prediction = model.predict(df)
        proba = model.predict_proba(df)
        confidence = float(np.max(proba))

        PREDICTION_VALUE.set(int(prediction[0]))
        PREDICTION_COUNT.labels(predicted_class=str(prediction[0])).inc()
        PREDICTION_CONFIDENCE.set(confidence)

        response = jsonify({
            "prediction": int(prediction[0]),
            "confidence": confidence
        })
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        RESPONSE_SIZE.observe(len(response.get_data()))
        return response

    except Exception as e:
        MODEL_ERROR_COUNT.inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        return jsonify({"error": str(e)}), 500

    finally:
        REQUEST_IN_PROGRESS.dec()
        REQUEST_LATENCY.observe(time.time() - start)
        UPTIME.set(time.time() - START_TIME)

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)