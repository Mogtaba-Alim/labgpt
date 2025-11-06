# LabGPT Docker Deployment Guide

Complete guide for running LabGPT on a VM using Docker containers.

## 🚀 Quick Start

### Prerequisites

1. **Docker & Docker Compose** installed on your VM
2. **NVIDIA Docker** (nvidia-docker2) for GPU support (optional but recommended)
3. **Environment variables** set up (API keys, tokens)

### Setup

1. **Clone and navigate to the repository:**
```bash
cd /path/to/labgpt
```

2. **Create `.env` file** with your credentials:
```bash
cat > .env << EOF
HF_TOKEN=your_huggingface_token
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
CUDA_VISIBLE_DEVICES=0
EOF
```

3. **Create necessary directories:**
```bash
mkdir -p indices models data output logs
```

## 📦 Container Services

### 1. **Inference Service** (Main Service)

Run inference with RAG-augmented responses:

```bash
# GPU version (recommended)
docker-compose up inference

# CPU version
docker-compose -f docker-compose.cpu.yml up inference
```

**Interactive usage:**
```bash
# Run inference with custom query
docker-compose run --rm inference python inference.py \
  "What is CRISPR gene editing?" \
  --index /app/indices/rag_demo_storage

# With custom parameters
docker-compose run --rm inference python inference.py \
  "Your question" \
  --index /app/indices/rag_demo_storage \
  --top-k 5 \
  --max-new-tokens 800 \
  --preset research
```

### 2. **RAG Indexing Service**

Create vector database from your documents:

```bash
# First, place your documents in ./data/documents/
# Then run indexing
docker-compose --profile indexing run --rm rag-indexer

# Or with custom paths
docker-compose run --rm rag-indexer \
  python -m RAG.cli ingest \
  --docs /app/data/your_documents \
  --index /app/indices/your_index_name
```

### 3. **Web Applications** (Optional)

#### Training Web App (Port 5002)
```bash
docker-compose --profile web-apps up training-app
# Access at http://your-vm-ip:5002
```

#### Data Generation Web App (Port 5001)
```bash
docker-compose --profile web-apps up data-gen-app
# Access at http://your-vm-ip:5001
```

#### Grant Generation Web App (Port 5000)
```bash
docker-compose --profile web-apps up grant-app
# Access at http://your-vm-ip:5000
```

## 🔧 Common Operations

### Building Images

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build inference

# Rebuild without cache
docker-compose build --no-cache inference
```

### Running CLI Commands

```bash
# Run unified CLI
docker-compose run --rm inference python labgpt_cli.py run-all \
  --code-repos /app/data/repos \
  --papers /app/data/papers \
  --output /app/output

# Run RAG CLI
docker-compose run --rm inference python -m RAG.cli ask \
  --index /app/indices/rag_demo_storage \
  --query "Your question"

# Run training
docker-compose run --rm inference python labgpt_cli.py train \
  --train-file /app/data/train.jsonl \
  --output /app/models/my_model
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f inference

# Last 100 lines
docker-compose logs --tail=100 inference
```

### Stopping Services

```bash
# Stop all running containers
docker-compose down

# Stop and remove volumes (careful!)
docker-compose down -v

# Stop specific service
docker-compose stop inference
```

## 📁 Volume Mounts

The following directories are mounted as volumes:

- `./indices` → `/app/indices` - RAG vector databases
- `./models` → `/app/models` - Model weights and cache
- `./data` → `/app/data` - Input documents and data
- `./output` → `/app/output` - Pipeline outputs
- `./logs` → `/app/logs` - Application logs

**Important:** These directories persist data between container restarts.

## 🎯 Complete Workflow Example

### Step 1: Index Your Documents

```bash
# Place documents in ./data/documents/
cp -r /path/to/your/papers ./data/documents/

# Create vector database
docker-compose --profile indexing run --rm rag-indexer \
  python -m RAG.cli ingest \
  --docs /app/data/documents \
  --index /app/indices/rag_demo_storage \
  --preset research
```

### Step 2: Run Inference

```bash
# Start inference service
docker-compose up -d inference

# Query the system
docker-compose exec inference python inference.py \
  "What is gene editing?" \
  --index /app/indices/rag_demo_storage
```

### Step 3: Generate Training Data (Optional)

```bash
docker-compose run --rm inference python labgpt_cli.py data-gen \
  --code-repos /app/data/repos \
  --papers /app/data/papers \
  --output /app/output/data_generation
```

### Step 4: Train Model (Optional)

```bash
docker-compose run --rm inference python labgpt_cli.py train \
  --train-file /app/output/data_generation/combined_instruct_train.jsonl \
  --val-file /app/output/data_generation/combined_instruct_val.jsonl \
  --output /app/models/my_labgpt_model
```

## 🖥️ GPU Support

### For GPU-enabled VMs:

1. **Install NVIDIA Docker:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

3. **Use GPU-enabled compose file:**
```bash
docker-compose up inference  # Uses GPU by default
```

### For CPU-only VMs:

```bash
# Use CPU-only compose file
docker-compose -f docker-compose.cpu.yml up inference
```

## 🔒 Security Considerations

1. **Environment Variables:** Never commit `.env` file. Use Docker secrets in production.
2. **Port Exposure:** Only expose necessary ports to public networks.
3. **Volume Permissions:** Ensure proper file permissions on mounted volumes.
4. **API Keys:** Rotate keys regularly and use least-privilege access.

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use CPU version:
```bash
docker-compose -f docker-compose.cpu.yml up inference
```

### Issue: "Permission denied" on volumes
**Solution:** Fix permissions:
```bash
sudo chown -R $USER:$USER ./indices ./models ./data ./output ./logs
```

### Issue: "Model not found" or "Index not found"
**Solution:** Check volume mounts and ensure data exists:
```bash
docker-compose exec inference ls -la /app/indices
docker-compose exec inference ls -la /app/models
```

### Issue: Slow inference
**Solution:** 
- Use GPU version if available
- Reduce `--max-new-tokens`
- Use smaller models (4-bit quantization is enabled by default)

### Issue: Container won't start
**Solution:** Check logs:
```bash
docker-compose logs inference
docker-compose logs training-app
```

## 📊 Resource Requirements

### Minimum (CPU-only):
- **CPU:** 4 cores
- **RAM:** 16GB
- **Disk:** 50GB free space
- **Speed:** Very slow inference (~30-60s per query)

### Recommended (GPU-enabled):
- **CPU:** 8+ cores
- **RAM:** 32GB+
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3060, A100, etc.)
- **Disk:** 100GB+ free space (models are large)
- **Speed:** Fast inference (~2-5s per query)

## 🚀 Production Deployment

For production, consider:

1. **Reverse Proxy:** Use nginx/traefik for SSL and routing
2. **Monitoring:** Add Prometheus/Grafana for metrics
3. **Logging:** Centralized logging with ELK stack
4. **Scaling:** Use Kubernetes for multi-instance deployment
5. **Backup:** Regular backups of `./indices` and `./models`

## 📝 Example Production Setup

```bash
# docker-compose.prod.yml
version: '3.8'
services:
  inference:
    # ... same config ...
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 🎓 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Guide](https://github.com/NVIDIA/nvidia-docker)
- [LabGPT README](README.md) for application-specific documentation

