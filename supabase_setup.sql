-- ============================================================================
-- SUPABASE ML STORAGE - DATABASE SETUP
-- Simple, working SQL for Supabase PostgreSQL
-- ============================================================================

-- ============================================================================
-- TABLE 1: ML Training Data
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_training_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    symbol TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence NUMERIC,
    features TEXT,
    actual_outcome TEXT,
    outcome_timestamp TIMESTAMPTZ,
    profit_loss NUMERIC,
    notes TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_training_symbol ON ml_training_data(symbol);
CREATE INDEX IF NOT EXISTS idx_training_timestamp ON ml_training_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_training_outcome ON ml_training_data(actual_outcome);


-- ============================================================================
-- TABLE 2: ML Models
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_models (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    accuracy NUMERIC,
    precision_score NUMERIC,
    recall_score NUMERIC,
    f1_score NUMERIC,
    training_samples INTEGER,
    feature_count INTEGER,
    hyperparameters TEXT,
    model_file_url TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_models_active ON ml_models(is_active);
CREATE INDEX IF NOT EXISTS idx_models_created ON ml_models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_name ON ml_models(model_name);


-- ============================================================================
-- TABLE 3: ML Statistics
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_statistics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    stats TEXT NOT NULL
);

-- Index
CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON ml_statistics(timestamp DESC);


-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name LIKE 'ml_%'
ORDER BY table_name;

-- Check table structure
SELECT
    'ml_training_data' as table_name,
    COUNT(*) as row_count
FROM ml_training_data
UNION ALL
SELECT
    'ml_models' as table_name,
    COUNT(*) as row_count
FROM ml_models
UNION ALL
SELECT
    'ml_statistics' as table_name,
    COUNT(*) as row_count
FROM ml_statistics;


-- ============================================================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================================================

-- Insert sample training record
INSERT INTO ml_training_data (symbol, prediction, confidence, features, notes)
VALUES (
    'NIFTY',
    'BUY',
    0.85,
    '{"rsi": 45, "macd": 1.2}',
    'Test record'
);

-- Insert sample model
INSERT INTO ml_models (
    model_name,
    version,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    training_samples,
    feature_count,
    hyperparameters,
    is_active
)
VALUES (
    'test_model',
    1,
    0.87,
    0.85,
    0.89,
    0.87,
    1000,
    150,
    '{"max_depth": 5, "learning_rate": 0.1}',
    FALSE
);


-- ============================================================================
-- DONE!
-- ============================================================================

-- Run this to verify everything worked:
SELECT 'Setup Complete!' as status,
       (SELECT COUNT(*) FROM ml_training_data) as training_records,
       (SELECT COUNT(*) FROM ml_models) as models,
       (SELECT COUNT(*) FROM ml_statistics) as statistics;
