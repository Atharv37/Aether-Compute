CREATE TABLE IF NOT EXISTS inspections (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    component_type VARCHAR(100) NOT NULL DEFAULT 'Unknown Component',
    inspection_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'completed',
    severity_score FLOAT,
    image_url TEXT NOT NULL,
    detection_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast status-based filtering
CREATE INDEX IF NOT EXISTS idx_inspections_status ON inspections(status);

