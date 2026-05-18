"""
AetherCompute — ML Worker  (Gold Master)
=========================================
Architecture
  - Outer try/except  : payload parsing, model loading, setup errors
  - Inner try/except  : image download + YOLOv8 inference (isolated failures)
  - persist_job()     : unified Redis + PostgreSQL write for success & FAILED states
  - ensure_connection(): auto-reconnects a stale PostgreSQL connection
  - publish_heartbeat(): writes a TTL-60s liveness key every HEARTBEAT_INTERVAL seconds
                         so the API /health endpoint can report worker status without SSH
"""

import os
import io
import gc
import json
import time
import logging
from collections import OrderedDict

import redis
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import psycopg2
from psycopg2.extras import Json
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Worker')

# ── Environment Config ───────────────────────────────────────────────────────
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')

DB_HOST     = os.environ.get('DB_HOST',     'localhost')
DB_PORT     = os.environ.get('DB_PORT',     '5432')
DB_USER     = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_NAME     = os.environ.get('DB_NAME')

# ── Tuning Constants ─────────────────────────────────────────────────────────
MODEL_CACHE_MAX_SIZE = int(os.environ.get('MODEL_CACHE_MAX_SIZE', 5))
# Max YOLO models kept in RAM simultaneously. Each model ≈ 6 MB.
# LRU eviction kicks in beyond this limit.

HEARTBEAT_INTERVAL   = int(os.environ.get('HEARTBEAT_INTERVAL', 30))
# Seconds between worker liveness pulses written to Redis.
# blpop uses this as its timeout so heartbeats fire even during idle periods.


# ════════════════════════════════════════════════════════════════════════════
# Database helpers
# ════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Connect to PostgreSQL with exponential-back retry logic."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=DB_HOST, port=DB_PORT,
                user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
            )
            logger.info("Successfully connected to PostgreSQL.")
            return conn
        except psycopg2.OperationalError as e:
            wait = 2 ** attempt          # 1 s, 2 s, 4 s, 8 s, 16 s
            logger.warning(
                f"DB connection failed (attempt {attempt + 1}/{max_retries}): {e} "
                f"— retrying in {wait}s"
            )
            time.sleep(wait)
    logger.error("Could not connect to PostgreSQL after multiple attempts.")
    return None


def ensure_connection(db_conn):
    """
    Probe the existing connection with a no-op SELECT.
    If it's stale (e.g., server restart, idle timeout), transparently reconnect.
    Returns a guaranteed-live connection (or None if reconnect also fails).
    """
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_conn.rollback()   # reset any implicit transaction from the probe
        return db_conn
    except Exception:
        logger.warning("PostgreSQL connection lost — reconnecting...")
        try:
            db_conn.close()
        except Exception:
            pass
        return get_db_connection()


# ════════════════════════════════════════════════════════════════════════════
# Redis helpers
# ════════════════════════════════════════════════════════════════════════════

def publish_heartbeat(r, status: str, model_loaded: bool, last_inference_ms=None):
    """
    Write a TTL-60s liveness key to Redis so the API /health endpoint can
    surface real worker status without SSH or log-diving.
    If the worker process dies, the key expires automatically within 60 s.

    Keys written
      worker:heartbeat  →  JSON { status, model_loaded, last_inference_ms,
                                  pid, timestamp }
    """
    payload = {
        "status":            status,
        "model_loaded":      model_loaded,
        "last_inference_ms": last_inference_ms,
        "pid":               os.getpid(),
        "timestamp":         time.time(),
    }
    r.set("worker:heartbeat", json.dumps(payload), ex=60)
    logger.debug(f"Heartbeat published: status={status}")


# ════════════════════════════════════════════════════════════════════════════
# Persistence helper
# ════════════════════════════════════════════════════════════════════════════

def persist_job(db_conn, r, job_id, model_type, component_type, inspection_type,
                image_url, status, detected_objects, severity_score, error_message=None):
    """
    Atomically write the final job result to:
      1. Redis   — TTL 1 hour  (for fast /status polling)
      2. PostgreSQL             (permanent audit history)

    CRITICAL ARCHITECTURAL DESIGN:
    Captures failure reasons explicitly via the `error_message` parameter and stores them in a
    dedicated database column. This preserves strict JSON schema compliance for `detected_objects`
    (which defaults to `[]` on failure) and prevents schema corruption in downstream dashboards.

    Called for BOTH 'completed' and 'FAILED' outcomes.
    Returns the (possibly refreshed) db_conn so the caller keeps a live reference.
    """
    # ── Redis ──────────────────────────────────────────────────────────────
    redis_payload = {
        "jobId":           job_id,
        "status":          status,
        "model_type":      model_type,
        "component_type":  component_type,
        "inspection_type": inspection_type,
        "severity_score":  severity_score,
        "result":          detected_objects,
    }
    if error_message:
        redis_payload["error"] = error_message

    r.set(f"job:result:{job_id}", json.dumps(redis_payload), ex=3600)

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    db_conn = ensure_connection(db_conn)    # auto-heal stale connection
    with db_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO inspections
                (job_id, model_type, component_type, inspection_type,
                 status, severity_score, image_url, detection_results, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO UPDATE
                SET status            = EXCLUDED.status,
                    severity_score    = EXCLUDED.severity_score,
                    detection_results = EXCLUDED.detection_results,
                    error_message     = EXCLUDED.error_message
        """, (
            job_id, model_type, component_type, inspection_type,
            status, severity_score, image_url, Json(detected_objects), error_message
        ))
    db_conn.commit()
    logger.info(f"Persisted job {job_id} → status={status} | severity={severity_score} | error={error_message}")
    return db_conn


# ════════════════════════════════════════════════════════════════════════════
# LRU-bounded model cache
# ════════════════════════════════════════════════════════════════════════════

def load_model(models_cache: OrderedDict, weight_file: str) -> YOLO:
    """
    Return a cached YOLO model, loading it if necessary.
    Evicts the least-recently-used model when the cache exceeds MODEL_CACHE_MAX_SIZE.
    Prevents unbounded RAM growth when many distinct model_url values are submitted.
    """
    if weight_file in models_cache:
        models_cache.move_to_end(weight_file)   # mark as recently used
        return models_cache[weight_file]

    logger.info(f"Loading YOLO weights: {weight_file} (cache size: {len(models_cache)})")
    model = YOLO(weight_file)
    models_cache[weight_file] = model

    if len(models_cache) > MODEL_CACHE_MAX_SIZE:
        evicted, _ = models_cache.popitem(last=False)   # remove least-recently used
        logger.info(f"Model cache full — evicted: {evicted}")

    return model


def get_requests_session():
    """
    Create a robust requests Session configured with urllib3 Retry logic.
    Provides automatic failover and exponential backoff for transient network instability
    (e.g., 502 Bad Gateway, 504 Gateway Timeout) when retrieving factory floor imagery.
    """
    session = requests.Session()
    # Configure 3 retries with exponential backoff factor (1s, 2s, 4s) across common transient errors
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ════════════════════════════════════════════════════════════════════════════
# Main worker loop
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("AetherCompute ML Worker starting...")

    # ── Redis ────────────────────────────────────────────────────────────────
    r = redis.Redis(host=REDIS_HOST, port=int(REDIS_PORT), decode_responses=True)
    try:
        r.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}.")
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    db_conn = get_db_connection()
    if not db_conn:
        return

    # ── Model cache (LRU-bounded OrderedDict) ────────────────────────────────
    logger.info("Initialising model cache...")
    models_cache = OrderedDict()
    models_cache['yolov8n.pt'] = YOLO('yolov8n.pt')
    logger.info("Default model yolov8n.pt loaded successfully.")

    # ── HTTP Session with Retries ────────────────────────────────────────────
    http_session = get_requests_session()

    # ── Initial heartbeat ────────────────────────────────────────────────────
    publish_heartbeat(r, 'idle', model_loaded=True)
    last_heartbeat = time.time()

    logger.info("Worker is listening for tasks in queue 'inference_queue'...")

    # ── Event loop ───────────────────────────────────────────────────────────
    while True:

        # Heartbeat timer — fires when blpop returns None on timeout
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            publish_heartbeat(r, 'idle', model_loaded=True)
            last_heartbeat = now

        # BLPOP with timeout so idle periods still publish heartbeats
        raw = r.blpop('inference_queue', timeout=HEARTBEAT_INTERVAL)
        if not raw:
            continue    # timeout expired — loop back, heartbeat will fire above

        _, payload_str = raw
        job_id = 'unknown-id'

        # ── Outer try: payload parsing, model loading, setup errors ──────────
        try:
            payload         = json.loads(payload_str)
            job_id          = payload.get('jobId', 'unknown-id')
            task_data       = payload.get('taskData', {})
            image_url       = task_data.get('image_url')
            model_type      = task_data.get('model_type', 'YOLOv8')
            component_type  = task_data.get('component_type', 'Unknown Component')
            inspection_type = task_data.get('inspection_type', 'General Inspection')
            model_url       = task_data.get('model_url')

            logger.info(
                f"Processing job: {job_id} | "
                f"Component: {component_type} | Inspection: {inspection_type}"
            )

            if not image_url:
                raise ValueError("No image_url provided in taskData")

            if model_type != 'YOLOv8':
                raise ValueError(
                    f"Unsupported model_type: '{model_type}'. Only YOLOv8 is supported."
                )

            # Download custom model weights (once; cached afterward)
            weight_file = 'yolov8n.pt'
            if model_url:
                weight_file = model_url.split('/')[-1]
                if not os.path.exists(weight_file):
                    logger.info(f"Downloading custom model weights from {model_url}...")
                    m_resp = http_session.get(model_url, timeout=30)
                    m_resp.raise_for_status()
                    with open(weight_file, 'wb') as f:
                        f.write(m_resp.content)
                    logger.info(f"Saved custom model: {weight_file}")

            yolo_model = load_model(models_cache, weight_file)

            # Publish 'processing' heartbeat so the API can show work-in-progress
            publish_heartbeat(r, 'processing', model_loaded=True)
            last_heartbeat = time.time()

            # ── Inner try: image download + inference (isolated) ─────────────
            # Failures here (bad URL, corrupt image, YOLO runtime error) are
            # caught and persisted as FAILED without terminating the worker.
            inference_start = time.perf_counter()
            try:
                # 1. Download image
                logger.info(f"Downloading image from {image_url}...")
                img_response = http_session.get(image_url, timeout=15)
                img_response.raise_for_status()         # raises on 4xx / 5xx

                # 2. Decode & validate image bytes
                raw_bytes = img_response.content
                del img_response                        # ← free network buffer immediately

                img = Image.open(io.BytesIO(raw_bytes))
                img.verify()                            # raises on corrupt file
                img = Image.open(io.BytesIO(raw_bytes))  # reopen after verify()
                del raw_bytes                           # ← free raw bytes

                # 3. Run YOLOv8 inference
                #    save=False  → CRITICAL: prevents YOLO writing annotated images
                #                  to runs/detect/predict/ (disk bloat)
                #    verbose=False → suppress per-frame console spam
                logger.info("Running YOLOv8 inference...")
                yolo_results = yolo_model(img, save=False, verbose=False)
                del img        # ← free PIL image after inference
                gc.collect()   # ← explicitly trigger GC after releasing large objects

                # 4. Extract detections and compute severity score
                detected_objects = []
                severity_score   = 0.0
                for r_obj in yolo_results:
                    for box in r_obj.boxes:
                        label      = yolo_model.names[int(box.cls[0])]
                        confidence = float(round(float(box.conf[0]), 4))
                        detected_objects.append({"label": label, "confidence": confidence})
                        if confidence > severity_score:
                            severity_score = confidence

                inference_ms = round((time.perf_counter() - inference_start) * 1000)
                logger.info(
                    f"Inference complete — job: {job_id} | "
                    f"objects: {len(detected_objects)} | "
                    f"severity: {severity_score} | {inference_ms} ms"
                )

                # 5. Persist success
                db_conn = persist_job(
                    db_conn, r, job_id, model_type, component_type, inspection_type,
                    image_url, 'completed', detected_objects, severity_score
                )

                # 6. Update heartbeat with real inference timing
                publish_heartbeat(
                    r, 'idle', model_loaded=True, last_inference_ms=inference_ms
                )
                last_heartbeat = time.time()

            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as net_err:
                error_msg = f"Image download failed: {net_err}"
                logger.error(f"Job {job_id} — {error_msg}")
                db_conn = persist_job(
                    db_conn, r, job_id, model_type, component_type, inspection_type,
                    image_url, 'FAILED', [], 0.0, error_message=error_msg
                )

            except (UnidentifiedImageError, OSError) as img_err:
                error_msg = f"Invalid or corrupt image: {img_err}"
                logger.error(f"Job {job_id} — {error_msg}")
                db_conn = persist_job(
                    db_conn, r, job_id, model_type, component_type, inspection_type,
                    image_url, 'FAILED', [], 0.0, error_message=error_msg
                )

            except Exception as inference_err:
                error_msg = f"Inference error: {inference_err}"
                logger.error(f"Job {job_id} — {error_msg}", exc_info=True)
                db_conn = persist_job(
                    db_conn, r, job_id, model_type, component_type, inspection_type,
                    image_url, 'FAILED', [], 0.0, error_message=error_msg
                )

        except Exception as outer_err:
            # Setup/parse failure — worker MUST continue to next job.
            logger.error(f"Fatal setup error [job: {job_id}]: {outer_err}", exc_info=True)
            try:
                db_conn.rollback()
            except Exception:
                pass
            # Best-effort Redis record (job_id may still be 'unknown-id')
            if job_id != 'unknown-id':
                r.set(
                    f"job:result:{job_id}",
                    json.dumps({"jobId": job_id, "status": "FAILED", "error": str(outer_err)}),
                    ex=3600
                )


if __name__ == "__main__":
    main()
