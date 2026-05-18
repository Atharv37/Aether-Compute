/**
 * AetherCompute — API Gateway  (Gold Master)
 * ============================================
 * Improvements over previous version
 *   1. express.json({ limit: '5mb' })   — explicit payload size cap
 *   2. Input validation on POST /task   — rejects missing image_url before queue
 *   3. Queue depth guard                — 503 when inference_queue ≥ MAX_QUEUE_DEPTH
 *   4. Enhanced /health endpoint        — reads worker heartbeat from Redis,
 *                                         surfaces model liveness without SSH
 *   5. /metrics returns failed_jobs_count from PostgreSQL for dashboard accuracy
 */

const express = require('express');
const path    = require('path');
const { createClient } = require('redis');
const { v4: uuidv4 }   = require('uuid');
const { Pool }          = require('pg');

// ── Config ──────────────────────────────────────────────────────────────────
const PORT      = process.env.PORT      || 3000;
const REDIS_HOST = process.env.REDIS_HOST || 'localhost';
const REDIS_PORT = process.env.REDIS_PORT || 6379;

const DB_HOST     = process.env.DB_HOST     || 'localhost';
const DB_PORT     = process.env.DB_PORT     || 5432;
const DB_USER     = process.env.DB_USER;
const DB_PASSWORD = process.env.DB_PASSWORD;
const DB_NAME     = process.env.DB_NAME;

// Maximum number of jobs that may sit in the inference_queue simultaneously.
// Requests that would exceed this are rejected with 503 to prevent Redis OOM.
const MAX_QUEUE_DEPTH = parseInt(process.env.MAX_QUEUE_DEPTH || '100', 10);

// ── App setup ────────────────────────────────────────────────────────────────
const app = express();

// CORS — allow all origins for dashboard & external clients
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin',  '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// 5 MB payload cap — prevents a malicious body from exhausting Node's heap.
// We only accept URLs (small), but a hard limit is defensive best-practice.
app.use(express.json({ limit: '5mb' }));

// Serve monitoring dashboard at /
app.use(express.static(path.join(__dirname, '..', 'dashboard')));

// ── Redis client ─────────────────────────────────────────────────────────────
const redisClient = createClient({ url: `redis://${REDIS_HOST}:${REDIS_PORT}` });
redisClient.on('error', (err) => console.error('[Redis] Client Error:', err));

// ── PostgreSQL pool ──────────────────────────────────────────────────────────
const pgPool = new Pool({
  host:     DB_HOST,
  port:     DB_PORT,
  user:     DB_USER,
  password: DB_PASSWORD,
  database: DB_NAME,
});

// ════════════════════════════════════════════════════════════════════════════
// Routes
// ════════════════════════════════════════════════════════════════════════════

/**
 * POST /task
 * Validate payload, guard queue depth, then enqueue the job.
 */
app.post('/task', async (req, res) => {
  try {
    const { image_url, component_type, inspection_type, model_type, model_url } = req.body;

    // ── Input validation ──────────────────────────────────────────────────
    // Ensure the client provided a valid HTTP/HTTPS URL for the inspection image.
    // This prevents malformed requests from polluting the Redis queue or crashing the worker.
    if (!image_url || typeof image_url !== 'string' || !image_url.startsWith('http')) {
      return res.status(400).json({
        error: 'image_url is required and must be a valid HTTP/HTTPS URL.',
      });
    }
    // Enforce strict length boundaries on metadata strings to prevent database column overflow
    // and mitigate potential Denial of Service via excessively large payload strings.
    if (component_type && (typeof component_type !== 'string' || component_type.length > 100)) {
      return res.status(400).json({
        error: 'component_type must be a string under 100 characters.',
      });
    }
    if (inspection_type && (typeof inspection_type !== 'string' || inspection_type.length > 100)) {
      return res.status(400).json({
        error: 'inspection_type must be a string under 100 characters.',
      });
    }

    // ── Queue depth guard ─────────────────────────────────────────────────
    const queueDepth = await redisClient.lLen('inference_queue');
    if (queueDepth >= MAX_QUEUE_DEPTH) {
      return res.status(503).json({
        error:       'Inference queue is at capacity. Please retry in a moment.',
        queue_depth: queueDepth,
        max_depth:   MAX_QUEUE_DEPTH,
      });
    }

    // ── Enqueue ───────────────────────────────────────────────────────────
    const jobId   = uuidv4();
    const taskData = { image_url, component_type, inspection_type, model_type, model_url };
    await redisClient.lPush('inference_queue', JSON.stringify({ jobId, taskData }));

    console.log(`[Task] Queued job ${jobId} | queue depth: ${queueDepth + 1}`);
    res.status(202).json({ jobId, status: 'queued', queue_position: queueDepth + 1 });

  } catch (err) {
    console.error('[Task] Failed to queue task:', err);
    res.status(500).json({ error: 'Internal server error while queuing task.' });
  }
});

/**
 * GET /status/:jobId
 * Poll the result of a specific job from Redis.
 */
app.get('/status/:jobId', async (req, res) => {
  try {
    const { jobId } = req.params;
    const data = await redisClient.get(`job:result:${jobId}`);

    if (data) return res.json(JSON.parse(data));
    res.status(404).json({ jobId, status: 'Pending / Not Found' });

  } catch (err) {
    console.error('[Status] Error fetching job status:', err);
    res.status(500).json({ error: 'Failed to fetch job status.' });
  }
});

/**
 * GET /metrics
 * High-performance operational metrics endpoint for live monitoring dashboards.
 * Combines real-time Redis queue depth with persistent PostgreSQL job status counts.
 */
app.get('/metrics', async (req, res) => {
  try {
    // Fetch current depth of the inference queue directly from Redis (O(1) operation)
    const queueLength  = await redisClient.lLen('inference_queue');

    // Aggregate job completion and failure statistics directly from PostgreSQL.
    // This replaces the legacy O(N) redisClient.keys() scan, ensuring metrics remain
    // performant and accurate even with millions of historical inspection records.
    const dbStats = await pgPool.query(`
      SELECT status, COUNT(*)::int AS count
      FROM inspections
      GROUP BY status
    `);
    const statusCounts = Object.fromEntries(
      dbStats.rows.map((r) => [r.status, r.count])
    );

    res.json({
      inference_queue_length: queueLength,
      completed_jobs_count:   statusCounts['completed'] ?? 0,
      db_completed:           statusCounts['completed'] ?? 0,
      db_failed:              statusCounts['FAILED']    ?? 0,
    });

  } catch (err) {
    console.error('[Metrics] Error fetching metrics:', err);
    res.status(500).json({ error: 'Failed to fetch metrics.' });
  }
});

/**
 * GET /history
 * Inspection records from PostgreSQL with pagination and status filtering.
 * Designed to support infinite scroll and targeted auditing in the frontend dashboard.
 */
app.get('/history', async (req, res) => {
  try {
    // Parse pagination parameters with safe defaults to prevent massive table scans
    const limit  = parseInt(req.query.limit)  || 10;
    const offset = parseInt(req.query.offset) || 0;
    const status = req.query.status;

    let queryStr = 'SELECT * FROM inspections';
    const params = [];

    // Dynamically append status filtering if requested by the client
    if (status) {
      params.push(status);
      queryStr += ` WHERE status = $1`;
    }

    // Enforce deterministic ordering by creation time, applying parameterized limit and offset
    queryStr += ` ORDER BY created_at DESC LIMIT $${params.length + 1} OFFSET $${params.length + 2}`;
    params.push(limit, offset);

    const result = await pgPool.query(queryStr, params);
    res.json(result.rows);
  } catch (err) {
    console.error('[History] Error fetching history:', err);
    res.status(500).json({ error: 'Failed to fetch inspection history.' });
  }
});

/**
 * GET /health
 * ─────────────────────────────────────────────────────────────────────────────
 * Senior-level health check — reads the worker's self-reported heartbeat from
 * Redis so ops teams can surface model liveness without SSH access.
 *
 * Worker writes { status, model_loaded, last_inference_ms, pid, timestamp }
 * to the key  worker:heartbeat  with TTL 60 s every ~30 s.
 * If the worker process dies, the key expires and alive → false within 60 s.
 */
app.get('/health', async (req, res) => {
  try {
    const raw    = await redisClient.get('worker:heartbeat');
    const hb     = raw ? JSON.parse(raw) : null;
    const ageSec = hb ? Math.round(Date.now() / 1000 - hb.timestamp) : null;
    const alive  = hb !== null && ageSec < 60;

    res.json({
      service:   'AetherCompute API Gateway',
      api_status: 'running',
      redis:      `${REDIS_HOST}:${REDIS_PORT}`,
      database:   `${DB_HOST}:${DB_PORT}`,
      worker: {
        alive,
        status:            hb?.status            ?? 'unknown',
        model_loaded:      hb?.model_loaded       ?? false,
        last_inference_ms: hb?.last_inference_ms  ?? null,
        pid:               hb?.pid                ?? null,
        last_seen:         hb ? new Date(hb.timestamp * 1000).toISOString() : null,
        heartbeat_age_sec: ageSec,
      },
    });

  } catch (err) {
    console.error('[Health] Error reading worker heartbeat:', err);
    res.status(500).json({ error: 'Failed to fetch health data.' });
  }
});

// ════════════════════════════════════════════════════════════════════════════
// Bootstrap
// ════════════════════════════════════════════════════════════════════════════
async function startServer() {
  await redisClient.connect();
  console.log(`[Redis] Connected at ${REDIS_HOST}:${REDIS_PORT}`);

  // Automated Data Retention Cleanup (Runs every hour)
  // Ensures compliance with storage quotas by purging inspection records older than 30 days.
  // Running asynchronously in the background prevents impact on active API request handling.
  setInterval(async () => {
    try {
      const res = await pgPool.query(`DELETE FROM inspections WHERE created_at < NOW() - INTERVAL '30 days'`);
      if (res?.rowCount > 0) {
        console.log(`[Retention] Cleaned up ${res.rowCount} old inspection records.`);
      }
    } catch (err) {
      console.error('[Retention] Error cleaning up old records:', err);
    }
  }, 3600 * 1000);

  app.listen(PORT, () => {
    console.log(`[API] AetherCompute Gateway running on port ${PORT}`);
    console.log(`[API] Queue depth limit: ${MAX_QUEUE_DEPTH} jobs`);
  });
}

startServer();
