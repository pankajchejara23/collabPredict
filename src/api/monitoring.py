from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# Initialize metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version', 'status', 'endpoint']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency distribution',
    ['model_version', 'endpoint'],
    buckets=[0.1, 0.5, 1, 2, 5]
)

FEATURE_DRIFT = Gauge(
    'feature_drift_score',
    'Data drift score per feature',
    ['feature_name']
)

def start_metrics_server(port=8001):
    """Start Prometheus metrics server"""
    start_http_server(port)

def track_latency(model_version: str, endpoint: str):
    """Decorator to track prediction latency"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                PREDICTION_LATENCY.labels(
                    model_version=model_version,
                    endpoint=endpoint
                ).observe(time.time() - start_time)
                return result
            except Exception as e:
                raise e
        return wrapper
    return decorator