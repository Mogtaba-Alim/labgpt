# LabGPT - AI-Powered Laboratory Research Assistant

A comprehensive web application for fine-tuning and deploying custom AI models for laboratory research. LabGPT provides an integrated platform for data generation, model training, RAG-powered chat, and grant writing assistance.

## Quick Start

### Prerequisites

- Python 3.10+
- Redis server
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 50GB+ disk space

### Installation

**1. Install Python dependencies:**

```bash
# From the FINAL_LABGPT directory
pip install -r requirements.txt
```

**2. Install and start Redis:**

```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Or run manually
redis-server
```

**3. Set up environment variables:**

```bash
# For data generation pipeline
export CLAUDE_API_KEY=your_claude_api_key
export OPENAI_KEY=your_openai_api_key

# For HuggingFace models
export HF_TOKEN=your_huggingface_token
```

### Running LabGPT

**Option 1: Using the convenience script (Recommended)**

```bash
# From the FINAL_LABGPT directory
./run_labgpt.sh
```

This starts both the Celery worker and Flask web application.

**Option 2: Manual start**

```bash
# Terminal 1: Start Celery worker
celery -A unified_app.celery_app worker \
    -P solo \
    -c 1 \
    -O fair \
    --max-tasks-per-child=1 \
    --loglevel=info

# Terminal 2: Start Flask application
python unified_app/app.py
```

**Option 3: Using bash alias**

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias labgpt-run='cd /path/to/FINAL_LABGPT && ./run_labgpt.sh'
```

Then simply run:

```bash
labgpt-run
```

### Access the Application

Open your browser to: **http://localhost:5003**

## Features

### Chat-First Interface
- **Default Mode**: Immediate access with pre-trained LabGPT model from HuggingFace
- **Project Mode**: Custom fine-tuned models with project-specific data
- **RAG Integration**: Optional document citations from indexed papers and lab docs
- **Project Switching**: Seamlessly switch between default and custom models

### Flexible Pipeline Configuration
- **Optional Pipelines**: All data sources are optional - configure only what you need
- **External Artifacts**: Use existing RAG indexes or training data
- **Skip and Resume**: Configure pipelines individually as needed
- **File Upload**: Upload training data directly or provide external paths

### Complete Workflow
1. **Data Configuration**: Code repositories, research papers, lab documents
2. **RAG Indexing**: Create searchable knowledge base (optional)
3. **Data Generation**: Generate training data from code and papers (optional)
4. **Model Training**: Fine-tune Llama 3.1 8B with LoRA
5. **Chat Interface**: Interact with trained model
6. **Grant Generation**: AI-assisted grant writing with RAG

## Project Structure

```
FINAL_LABGPT/
├── unified_app/              # Web application (main interface)
│   ├── app.py               # Flask application
│   ├── routes/              # API endpoints and pages
│   ├── models/              # Database models
│   ├── services/            # Business logic
│   ├── tasks/               # Celery background tasks
│   └── templates/           # HTML templates
├── unified_app_projects/    # Project workspaces (created at runtime)
├── labgpt-final-index/     # Default RAG index (optional)
├── data_generation/         # Data generation pipeline
├── RAG/                     # RAG indexing pipeline
├── training/                # Model training pipeline
└── grant_generation/        # Grant writing pipeline
```

## Usage

### Creating Your First Project

1. Navigate to **http://localhost:5003**
2. Click **"Projects"** in the header
3. Click **"Create New Project"**
4. Enter project name and description
5. Configure data sources (all optional):
   - Code repositories (local paths or GitHub URLs)
   - Research papers (PDF, TXT, MD)
   - Lab documents
   - Or provide existing RAG index / training data
6. Click **"Start Configured Pipelines"** or **"Skip for Now"**
7. Monitor pipeline progress on the status page

### Using Default Chat (No Project Required)

1. Navigate to **http://localhost:5003** (landing page is chat)
2. Start chatting immediately with the default LabGPT model
3. Toggle RAG if default index is available
4. Use project selector dropdown to switch to project models

## Documentation

- **Comprehensive Guide**: `unified_app/unified_app_comprehensive_guide.md` - Complete technical documentation
- **CLI Tools**: `LABGPT_CLI.md` - Command-line interface documentation
- **Data Generation**: `data_generation/data_generation_comprehensive_guide.md` - Pipeline details

## Configuration

### Default Settings

- **Port**: 5003
- **Base Model**: meta-llama/Llama-3.1-8B
- **Default LabGPT Model**: MogtabaAlim/llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba
- **LoRA Rank**: 16
- **Batch Size**: 2
- **Training Epochs**: 3

### Environment Variables

Create `unified_app/.env` for custom configuration:

```bash
# Flask
SECRET_KEY=your-secure-secret-key
FLASK_ENV=development

# Database
DATABASE_URL=sqlite:///unified_app_projects/labgpt_unified.db

# Celery & Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# GPU
CUDA_VISIBLE_DEVICES=0

# API Keys
CLAUDE_API_KEY=your_claude_api_key
OPENAI_KEY=your_openai_api_key

# HuggingFace
HUGGINGFACE_TOKEN=your_huggingface_token
```

## Troubleshooting

### Redis Connection Error

```bash
# Check if Redis is running
redis-cli ping

# Should return: PONG
```

### Port Already in Use

```bash
# Find process using port 5003
lsof -i :5003

# Kill process
kill -9 <PID>
```

### GPU Out of Memory

Reduce batch size in training configuration or use CPU mode.

### Celery Tasks Not Starting

```bash
# Check Celery worker status
celery -A unified_app.celery_app inspect active

# Restart worker
pkill -f "celery worker"
./run_labgpt.sh
```

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB
- GPU: Not required (CPU inference supported)

### Recommended
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB+ SSD
- GPU: NVIDIA RTX 3090 or better (10GB+ VRAM)

## Advanced Usage

### Production Deployment

See `unified_app/unified_app_comprehensive_guide.md` for:
- Docker deployment with docker-compose
- Systemd service configuration
- PostgreSQL setup
- Nginx reverse proxy
- SSL/TLS configuration

### Custom Model Training

1. Provide your own training data (JSONL format)
2. Configure hyperparameters in training page
3. Monitor progress with TensorBoard
4. Deploy trained model for chat

### RAG System Configuration

1. Index your lab documents and papers
2. Configure chunk size and overlap
3. Adjust retrieval parameters (top-k, similarity threshold)
4. Enable in chat for document-grounded responses

## Support

- **Issues**: Open an issue on GitHub
- **Documentation**: See `unified_app/unified_app_comprehensive_guide.md`
- **Email**: support@bhklab.ca

## License

MIT License - See LICENSE file for details

## Citation

If you use LabGPT in your research, please cite:

```bibtex
@software{labgpt2025,
  title = {LabGPT: A Unified Framework for Instant AI-Powered Lab Research Assistants},
  author = {Mogtaba Alim},
  year = {2025}
  url = {https://github.com/Mogtaba-Alim/labgpt}
}
```

---

**Version**: 2.0.0
**Last Updated**: 2025-01-10
**Status**: Production Ready
