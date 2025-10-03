#!/bin/bash

# HAILEI Deployment Script
# Production deployment automation for Docker containers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deploy.log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        warning "Environment file $ENV_FILE not found. Copying from example..."
        if [ -f ".env.example" ]; then
            cp .env.example "$ENV_FILE"
            warning "Please edit $ENV_FILE with your configuration before continuing."
            exit 1
        else
            error "No environment template found. Please create $ENV_FILE"
        fi
    fi
    
    success "Prerequisites check completed"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p "$BACKUP_DIR"
    mkdir -p config/grafana/dashboards
    mkdir -p config/grafana/datasources
    mkdir -p certs
    
    success "Directories created"
}

# Backup existing data
backup_data() {
    if [ "$1" = "--backup" ]; then
        log "Creating backup..."
        
        BACKUP_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
        BACKUP_PATH="$BACKUP_DIR/hailei_backup_$BACKUP_TIMESTAMP"
        
        mkdir -p "$BACKUP_PATH"
        
        # Backup volumes if they exist
        if docker volume ls | grep -q hailei_postgres-data; then
            log "Backing up PostgreSQL data..."
            docker run --rm -v hailei_postgres-data:/data -v "$PWD/$BACKUP_PATH":/backup ubuntu tar czf /backup/postgres_data.tar.gz -C /data .
        fi
        
        if docker volume ls | grep -q hailei_redis-data; then
            log "Backing up Redis data..."
            docker run --rm -v hailei_redis-data:/data -v "$PWD/$BACKUP_PATH":/backup ubuntu tar czf /backup/redis_data.tar.gz -C /data .
        fi
        
        # Backup configuration
        cp -r config "$BACKUP_PATH/" 2>/dev/null || true
        cp "$ENV_FILE" "$BACKUP_PATH/" 2>/dev/null || true
        
        success "Backup created at $BACKUP_PATH"
    fi
}

# Build and deploy
deploy() {
    log "Starting deployment..."
    
    # Pull latest images
    log "Pulling latest images..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" pull
    fi
    
    # Build HAILEI image
    log "Building HAILEI image..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" build hailei-api
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" build hailei-api
    fi
    
    # Start services
    log "Starting services..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" up -d
    fi
    
    success "Deployment completed"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check HAILEI API
    if curl -f http://localhost:8000/health &> /dev/null; then
        success "HAILEI API is healthy"
    else
        error "HAILEI API health check failed"
    fi
    
    # Check Nginx
    if curl -f http://localhost/health &> /dev/null; then
        success "Nginx is healthy"
    else
        warning "Nginx health check failed"
    fi
    
    # Check Redis
    if docker exec hailei-redis redis-cli ping | grep -q PONG; then
        success "Redis is healthy"
    else
        warning "Redis health check failed"
    fi
    
    # Check PostgreSQL
    if docker exec hailei-postgres pg_isready -U hailei | grep -q "accepting connections"; then
        success "PostgreSQL is healthy"
    else
        warning "PostgreSQL health check failed"
    fi
}

# Show status
show_status() {
    log "Service status:"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" ps
    fi
    
    echo
    log "Available endpoints:"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • API Health Check: http://localhost:8000/health"
    echo "  • Frontend API: http://localhost:8000/frontend/"
    echo "  • WebSocket: ws://localhost:8000/ws/"
    echo "  • Nginx Proxy: http://localhost/"
    echo "  • Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
}

# Stop services
stop() {
    log "Stopping services..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" down
    fi
    
    success "Services stopped"
}

# Clean up
cleanup() {
    log "Cleaning up..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans
    fi
    
    # Remove unused Docker resources
    docker system prune -f
    
    success "Cleanup completed"
}

# Main script
main() {
    case "$1" in
        "deploy")
            check_prerequisites
            setup_directories
            backup_data "$2"
            deploy
            health_check
            show_status
            ;;
        "start")
            if command -v docker-compose &> /dev/null; then
                docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
            else
                docker compose -f "$DOCKER_COMPOSE_FILE" up -d
            fi
            health_check
            show_status
            ;;
        "stop")
            stop
            ;;
        "restart")
            stop
            sleep 5
            if command -v docker-compose &> /dev/null; then
                docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
            else
                docker compose -f "$DOCKER_COMPOSE_FILE" up -d
            fi
            health_check
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            if command -v docker-compose &> /dev/null; then
                docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f "${2:-hailei-api}"
            else
                docker compose -f "$DOCKER_COMPOSE_FILE" logs -f "${2:-hailei-api}"
            fi
            ;;
        "backup")
            backup_data "--backup"
            ;;
        "cleanup")
            cleanup
            ;;
        "health")
            health_check
            ;;
        *)
            echo "HAILEI Deployment Script"
            echo
            echo "Usage: $0 {deploy|start|stop|restart|status|logs|backup|cleanup|health}"
            echo
            echo "Commands:"
            echo "  deploy [--backup]  - Full deployment with optional backup"
            echo "  start             - Start all services"
            echo "  stop              - Stop all services"
            echo "  restart           - Restart all services"
            echo "  status            - Show service status"
            echo "  logs [service]    - Show logs (default: hailei-api)"
            echo "  backup            - Create data backup"
            echo "  cleanup           - Clean up containers and volumes"
            echo "  health            - Run health checks"
            echo
            echo "Examples:"
            echo "  $0 deploy --backup    # Deploy with backup"
            echo "  $0 logs nginx         # Show nginx logs"
            echo "  $0 restart            # Restart all services"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"