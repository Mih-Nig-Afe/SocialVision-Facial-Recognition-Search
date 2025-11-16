#!/bin/bash

# SocialVision Docker Demo Script
# Quick script to build, start, and test the application

set -e

echo "🔍 SocialVision Docker Demo Script"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "Docker and Docker Compose are installed"

# Step 1: Build the image
echo ""
echo "Step 1: Building Docker image..."
echo "This may take 10-15 minutes on first run..."
docker-compose build

if [ $? -eq 0 ]; then
    print_status "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Step 2: Start the container
echo ""
echo "Step 2: Starting container..."
docker-compose up -d

if [ $? -eq 0 ]; then
    print_status "Container started successfully"
else
    print_error "Failed to start container"
    exit 1
fi

# Step 3: Wait for application to be ready
echo ""
echo "Step 3: Waiting for application to be ready..."
echo "This may take 30-60 seconds..."

MAX_WAIT=120
WAIT_TIME=0
INTERVAL=5

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        print_status "Application is ready!"
        break
    fi
    
    echo -n "."
    sleep $INTERVAL
    WAIT_TIME=$((WAIT_TIME + INTERVAL))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    print_warning "Application may not be ready yet. Check logs with: docker-compose logs"
else
    echo ""
fi

# Step 4: Show status
echo ""
echo "Step 4: Container Status"
echo "========================"
docker-compose ps

# Step 5: Show logs
echo ""
echo "Step 5: Recent Logs"
echo "=================="
docker-compose logs --tail=20

# Step 6: Open browser (macOS)
echo ""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening browser..."
    open http://localhost:8501
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Open your browser and navigate to: http://localhost:8501"
    xdg-open http://localhost:8501 2>/dev/null || true
else
    echo "Open your browser and navigate to: http://localhost:8501"
fi

echo ""
echo "===================================="
print_status "Demo setup complete!"
echo ""
echo "Application URL: http://localhost:8501"
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop container:   docker-compose down"
echo "  Restart:          docker-compose restart"
echo "  Check status:    docker-compose ps"
echo ""
echo "For testing guide, see: docs/DOCKER_TESTING_GUIDE.md"
echo "For demonstration guide, see: docs/DEMONSTRATION_GUIDE.md"
echo ""

