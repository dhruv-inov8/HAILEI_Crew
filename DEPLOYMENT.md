# HAILEI Production Deployment Guide

## Quick Start

### Prerequisites
- Docker 20.10+ 
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### 1. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (REQUIRED)
nano .env
```

**Critical Settings:**
- `JWT_SECRET_KEY` - Change from default
- `POSTGRES_PASSWORD` - Set secure password  
- `GRAFANA_PASSWORD` - Set admin password

### 2. Deploy
```bash
# Full deployment with backup
./deploy.sh deploy --backup

# Or simple start
./deploy.sh start
```

### 3. Verify
- API: http://localhost:8000/health
- Docs: http://localhost:8000/docs
- Frontend: http://localhost:8000/frontend/
- Monitoring: http://localhost:3000

## Services Overview

| Service | Port | Purpose |
|---------|------|---------|
| hailei-api | 8000 | Main FastAPI application |
| nginx | 80/443 | Reverse proxy & load balancer |
| redis | 6379 | Session storage & caching |
| postgres | 5432 | Persistent data storage |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboards |

## Architecture

```
[Client] → [Nginx:80] → [HAILEI API:8000] ↔ [Redis:6379]
                                         ↔ [PostgreSQL:5432]
                                         ↔ [Prometheus:9090] → [Grafana:3000]
```

## Configuration

### Core API Settings
```bash
# .env file
ENVIRONMENT=production
JWT_SECRET_KEY=your-secure-secret-key
API_PORT=8000
MAX_WORKERS=4
```

### Database Configuration
```bash
POSTGRES_DB=hailei
POSTGRES_USER=hailei
POSTGRES_PASSWORD=secure_password
```

### Security Settings
```bash
CORS_ORIGINS=https://yourdomain.com
JWT_EXPIRE_MINUTES=1440
```

## Commands

### Management
```bash
./deploy.sh deploy [--backup]  # Full deployment
./deploy.sh start              # Start services
./deploy.sh stop               # Stop services  
./deploy.sh restart            # Restart all
./deploy.sh status             # Show status
```

### Monitoring
```bash
./deploy.sh logs [service]     # View logs
./deploy.sh health             # Health checks
./deploy.sh backup             # Create backup
```

### Maintenance
```bash
./deploy.sh cleanup            # Clean unused resources
docker system prune -a         # Deep cleanup
```

## Production Checklist

### Security
- [ ] Change default passwords
- [ ] Set secure JWT secret
- [ ] Configure CORS origins
- [ ] Enable HTTPS (add certificates)
- [ ] Set up firewall rules
- [ ] Review nginx security headers

### Performance
- [ ] Tune worker processes (MAX_WORKERS)
- [ ] Configure Redis memory limits
- [ ] Set up database connection pooling
- [ ] Enable nginx compression
- [ ] Configure rate limiting

### Monitoring
- [ ] Set up Grafana dashboards
- [ ] Configure alerting rules
- [ ] Set up log aggregation
- [ ] Monitor disk usage
- [ ] Set up backup schedule

### High Availability
- [ ] Set up load balancing
- [ ] Configure database replication
- [ ] Set up container orchestration
- [ ] Implement health checks
- [ ] Configure auto-restart policies

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.override.yml
services:
  hailei-api:
    deploy:
      replicas: 3
    environment:
      - MAX_WORKERS=2
```

### Resource Limits
```yaml
services:
  hailei-api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## SSL/HTTPS Setup

### 1. Add certificates
```bash
mkdir -p certs
# Add your cert.pem and key.pem files
```

### 2. Enable HTTPS in nginx
```bash
# Uncomment HTTPS server block in config/nginx.conf
```

### 3. Update environment
```bash
# .env
CORS_ORIGINS=https://yourdomain.com
```

## Backup & Recovery

### Automated Backup
```bash
# Daily backup (add to crontab)
0 2 * * * /path/to/hailei/deploy.sh backup
```

### Manual Backup
```bash
./deploy.sh backup
# Creates timestamped backup in ./backups/
```

### Restore Process
```bash
# Stop services
./deploy.sh stop

# Restore volumes from backup
docker run --rm -v hailei_postgres-data:/data -v ./backups/backup_timestamp:/backup ubuntu tar xzf /backup/postgres_data.tar.gz -C /data

# Start services
./deploy.sh start
```

## Troubleshooting

### Common Issues

**503 Service Unavailable**
```bash
# Check API health
curl http://localhost:8000/health

# Check logs
./deploy.sh logs hailei-api

# Restart API
docker-compose restart hailei-api
```

**Database Connection Issues**
```bash
# Check PostgreSQL
./deploy.sh logs postgres
docker exec hailei-postgres pg_isready -U hailei

# Reset database
docker-compose down
docker volume rm hailei_postgres-data
./deploy.sh start
```

**Redis Issues**
```bash
# Check Redis
docker exec hailei-redis redis-cli ping

# Clear cache
docker exec hailei-redis redis-cli FLUSHALL
```

**WebSocket Connection Failures**
```bash
# Check nginx websocket config
./deploy.sh logs nginx

# Test websocket endpoint
wscat -c ws://localhost/ws/test-session
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=debug
./deploy.sh restart

# View detailed logs
./deploy.sh logs hailei-api
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check system metrics
curl http://localhost:8000/statistics

# View Grafana dashboards
open http://localhost:3000
```

## API Endpoints

### Core API
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - API documentation
- `GET /statistics` - System metrics

### Authentication
- `POST /auth/login` - User login
- `GET /auth/me` - User info
- `GET /auth/demo-credentials` - Demo credentials

### Frontend API
- `POST /frontend/quick-start` - Quick session creation
- `POST /frontend/chat/{session_id}` - Chat interface
- `GET /frontend/progress/{session_id}` - Progress tracking
- `POST /frontend/actions/{session_id}/approve` - Approve output
- `GET /frontend/templates/course-types` - Course templates

### WebSocket
- `WS /ws/{session_id}` - Real-time communication

## Monitoring

### Health Checks
All services include health checks:
- API: HTTP health endpoint
- PostgreSQL: pg_isready
- Redis: ping command
- Nginx: HTTP status check

### Metrics Collection
- Prometheus scrapes metrics every 30s
- Custom HAILEI metrics available at `/metrics`
- System metrics via node-exporter
- Application performance monitoring

### Dashboards
Grafana provides:
- System overview dashboard
- API performance metrics
- Database performance
- Error rate monitoring
- Resource utilization

## Support

### Logs Location
- Application: `./logs/hailei.log`
- Nginx: Container logs via `docker logs`
- Database: Container logs via `docker logs`

### Status Endpoints
- Overall: `GET /health`
- Detailed: `GET /statistics`
- Frontend-specific: `GET /frontend/health/frontend`

### Community
- Documentation: `/docs` endpoint
- Issues: Create GitHub issue
- Updates: Check for new releases

---

**Security Note**: This is a production deployment guide. Always review security settings for your specific environment and compliance requirements.