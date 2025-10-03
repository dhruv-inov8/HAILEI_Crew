-- HAILEI Database Initialization Script
-- PostgreSQL schema for production data persistence

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS hailei;
CREATE SCHEMA IF NOT EXISTS sessions;
CREATE SCHEMA IF NOT EXISTS metrics;

-- Sessions table for persistent session storage
CREATE TABLE IF NOT EXISTS sessions.conversation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    course_request JSONB NOT NULL,
    user_preferences JSONB,
    current_phase VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Context storage for conversation context
CREATE TABLE IF NOT EXISTS sessions.conversation_context (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES sessions.conversation_sessions(session_id) ON DELETE CASCADE,
    context_type VARCHAR(50) NOT NULL, -- global, phase, agent
    content JSONB NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Agent execution history
CREATE TABLE IF NOT EXISTS sessions.agent_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES sessions.conversation_sessions(session_id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    phase_id VARCHAR(100),
    task_description TEXT,
    input_data JSONB,
    output_data JSONB,
    execution_time_ms INTEGER,
    status VARCHAR(50) DEFAULT 'completed',
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- User feedback and refinements
CREATE TABLE IF NOT EXISTS sessions.user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES sessions.conversation_sessions(session_id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    phase_id VARCHAR(100),
    feedback_text TEXT NOT NULL,
    feedback_type VARCHAR(50) DEFAULT 'refinement', -- refinement, approval, rejection
    refined_output JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Metrics and analytics
CREATE TABLE IF NOT EXISTS metrics.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    metric_type VARCHAR(50) DEFAULT 'counter', -- counter, gauge, histogram
    tags JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS metrics.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions.conversation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions.conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions.conversation_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions.conversation_sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_context_session_id ON sessions.conversation_context(session_id);
CREATE INDEX IF NOT EXISTS idx_context_type ON sessions.conversation_context(context_type);
CREATE INDEX IF NOT EXISTS idx_context_tags ON sessions.conversation_context USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_context_created_at ON sessions.conversation_context(created_at);

CREATE INDEX IF NOT EXISTS idx_executions_session_id ON sessions.agent_executions(session_id);
CREATE INDEX IF NOT EXISTS idx_executions_agent_id ON sessions.agent_executions(agent_id);
CREATE INDEX IF NOT EXISTS idx_executions_phase_id ON sessions.agent_executions(phase_id);
CREATE INDEX IF NOT EXISTS idx_executions_started_at ON sessions.agent_executions(started_at);

CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON sessions.user_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_agent_id ON sessions.user_feedback(agent_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON sessions.user_feedback(created_at);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics.system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_tags ON metrics.system_metrics USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON metrics.api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_requests_timestamp ON metrics.api_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_requests_status_code ON metrics.api_requests(status_code);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_sessions_updated_at 
    BEFORE UPDATE ON sessions.conversation_sessions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE OR REPLACE VIEW sessions.active_sessions AS
SELECT 
    session_id,
    user_id,
    current_phase,
    created_at,
    updated_at,
    extract(epoch from (NOW() - updated_at)) as inactive_seconds
FROM sessions.conversation_sessions
WHERE status = 'active';

CREATE OR REPLACE VIEW metrics.endpoint_performance AS
SELECT 
    endpoint,
    method,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END)::float / COUNT(*) * 100 as error_rate
FROM metrics.api_requests
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY endpoint, method
ORDER BY request_count DESC;

-- Cleanup function for old data
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean up old completed sessions
    DELETE FROM sessions.conversation_sessions 
    WHERE status = 'completed' 
    AND completed_at < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old metrics
    DELETE FROM metrics.api_requests 
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;
    
    DELETE FROM metrics.system_metrics 
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_to_keep;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA sessions TO hailei;
GRANT USAGE ON SCHEMA metrics TO hailei;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA sessions TO hailei;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO hailei;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA sessions TO hailei;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO hailei;

-- Insert initial data
INSERT INTO metrics.system_metrics (metric_name, metric_value, metric_type, tags) 
VALUES ('database_initialized', 1, 'counter', '{"version": "1.0.0", "timestamp": "' || NOW() || '"}')
ON CONFLICT DO NOTHING;