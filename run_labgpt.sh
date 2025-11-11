#!/bin/bash

# LabGPT Unified App Launcher
# Starts both Celery worker and Flask application

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}       LabGPT Unified Application${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if Redis is running
echo -e "${YELLOW}Checking Redis server...${NC}"
if redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${RED}✗ Redis is not running${NC}"
    echo -e "${YELLOW}Starting Redis server...${NC}"

    # Try to start Redis
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes
        sleep 2
        if redis-cli ping &> /dev/null; then
            echo -e "${GREEN}✓ Redis started successfully${NC}"
        else
            echo -e "${RED}✗ Failed to start Redis${NC}"
            echo "Please start Redis manually: redis-server"
            exit 1
        fi
    else
        echo -e "${RED}✗ Redis not installed${NC}"
        echo "Install Redis:"
        echo "  Ubuntu/Debian: sudo apt-get install redis-server"
        echo "  macOS: brew install redis"
        exit 1
    fi
fi

echo ""

# Check if Python packages are installed
echo -e "${YELLOW}Checking Python dependencies...${NC}"
if python -c "import flask" &> /dev/null; then
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}✗ Python dependencies missing${NC}"
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to install dependencies${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

echo ""

# Set PYTHONPATH to ensure unified_app module can be imported
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down LabGPT...${NC}"

    # Kill background processes
    if [ ! -z "$CELERY_PID" ]; then
        kill $CELERY_PID 2>/dev/null
        echo -e "${GREEN}✓ Celery worker stopped${NC}"
    fi

    if [ ! -z "$FLASK_PID" ]; then
        kill $FLASK_PID 2>/dev/null
        echo -e "${GREEN}✓ Flask application stopped${NC}"
    fi

    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}       LabGPT stopped successfully${NC}"
    echo -e "${BLUE}================================================${NC}"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start Celery worker in background
echo -e "${YELLOW}Starting Celery worker...${NC}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
celery -A unified_app.celery_app worker \
    -P solo \
    -c 1 \
    -O fair \
    --max-tasks-per-child=1 \
    --loglevel=info \
    > logs/celery.log 2>&1 &

CELERY_PID=$!

# Wait a moment for Celery to start
sleep 2

# Check if Celery is running
if ps -p $CELERY_PID > /dev/null; then
    echo -e "${GREEN}✓ Celery worker started (PID: $CELERY_PID)${NC}"
else
    echo -e "${RED}✗ Failed to start Celery worker${NC}"
    echo "Check logs/celery.log for details"
    exit 1
fi

echo ""

# Start Flask application in background
echo -e "${YELLOW}Starting Flask application...${NC}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
python -m unified_app.app > logs/flask.log 2>&1 &

FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

# Check if Flask is running
if ps -p $FLASK_PID > /dev/null; then
    echo -e "${GREEN}✓ Flask application started (PID: $FLASK_PID)${NC}"
else
    echo -e "${RED}✗ Failed to start Flask application${NC}"
    echo "Check logs/flask.log for details"
    kill $CELERY_PID 2>/dev/null
    exit 1
fi

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}       LabGPT is now running!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Web Interface:${NC} http://localhost:5003"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo -e "  Celery: logs/celery.log"
echo -e "  Flask:  logs/flask.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop LabGPT${NC}"
echo ""

# Wait for both processes
wait $CELERY_PID $FLASK_PID
