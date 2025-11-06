#!/bin/bash
# LabGPT Docker Setup Script
# Quick setup for VM deployment

set -e

echo "🚀 LabGPT Docker Setup"
echo "======================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker found"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and tokens!"
    echo ""
else
    echo "✓ .env file exists"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p indices models data output logs
echo "✓ Directories created"

# Build Docker images
echo ""
echo "🔨 Building Docker images..."
echo "This may take a while on first run..."
docker-compose build

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your credentials"
echo "  2. Add documents to ./data/documents/"
echo "  3. Run: make index DOCS=./data/documents"
echo "  4. Run: make inference QUERY='Your question'"
echo ""
echo "Or see DOCKER.md for full documentation."

