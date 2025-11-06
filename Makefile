# LabGPT Docker Makefile
# Convenience commands for Docker operations

.PHONY: help build up down logs clean inference index train data-gen web-apps

# Default target
help:
	@echo "LabGPT Docker Commands:"
	@echo ""
	@echo "  make build          - Build all Docker images"
	@echo "  make up             - Start inference service"
	@echo "  make down           - Stop all services"
	@echo "  make logs           - View inference logs"
	@echo "  make inference      - Run inference interactively"
	@echo "  make index          - Create RAG index from documents"
	@echo "  make train          - Run training pipeline"
	@echo "  make data-gen       - Generate training data"
	@echo "  make web-apps       - Start all web applications"
	@echo "  make clean          - Remove containers and volumes"
	@echo "  make cpu            - Use CPU-only version"
	@echo ""

# Build images
build:
	docker-compose build

# Start inference service
up:
	docker-compose up -d inference

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f inference

# Run inference interactively
inference:
	docker-compose run --rm inference python inference.py \
		"$${QUERY:-What is CRISPR?}" \
		--index /app/indices/rag_demo_storage

# Create RAG index
index:
	@if [ -z "$${DOCS}" ]; then \
		echo "Usage: make index DOCS=/path/to/documents"; \
		echo "Example: make index DOCS=./data/documents"; \
		exit 1; \
	fi
	docker-compose --profile indexing run --rm rag-indexer \
		python -m RAG.cli ingest \
		--docs $${DOCS} \
		--index /app/indices/rag_demo_storage \
		--preset research

# Run training
train:
	@if [ -z "$${TRAIN_FILE}" ] || [ -z "$${OUTPUT}" ]; then \
		echo "Usage: make train TRAIN_FILE=/path/to/train.jsonl OUTPUT=/path/to/output"; \
		exit 1; \
	fi
	docker-compose run --rm inference python labgpt_cli.py train \
		--train-file $${TRAIN_FILE} \
		--output $${OUTPUT}

# Generate training data
data-gen:
	@if [ -z "$${CODE_REPOS}" ]; then \
		echo "Usage: make data-gen CODE_REPOS=/path/to/repos OUTPUT=/path/to/output"; \
		exit 1; \
	fi
	docker-compose run --rm inference python labgpt_cli.py data-gen \
		--code-repos $${CODE_REPOS} \
		--output $${OUTPUT:-/app/output/data_generation}

# Start web applications
web-apps:
	docker-compose --profile web-apps up -d

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# CPU-only version
cpu:
	docker-compose -f docker-compose.cpu.yml up inference

# Shell into container
shell:
	docker-compose exec inference /bin/bash

# Check status
status:
	docker-compose ps
	@echo ""
	@echo "Volume mounts:"
	@ls -lh indices/ models/ data/ output/ logs/ 2>/dev/null || echo "Some directories don't exist yet"

