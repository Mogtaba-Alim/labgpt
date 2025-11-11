"""
Configuration for the unified LabGPT web application.
"""

import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).parent.parent
UNIFIED_APP_DIR = Path(__file__).parent
PROJECTS_DIR = BASE_DIR / "unified_app_projects"
TEMP_DIR = BASE_DIR / "unified_app_temp"

# Ensure directories exist
PROJECTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Flask configuration
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{PROJECTS_DIR / "labgpt_unified.db"}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload configuration
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload
    UPLOAD_FOLDER = TEMP_DIR / "uploads"
    UPLOAD_FOLDER.mkdir(exist_ok=True)

    # Celery configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'
    CELERY_TASK_TRACK_STARTED = True
    CELERY_TASK_TIME_LIMIT = 86400  # 24 hours max per task

    # GPU Management (Redis-based locking)
    REDIS_GPU_LOCK_KEY = "gpu:mutex"
    REDIS_GPU_LOCK_TIMEOUT = 86400  # 24 hours (max task duration)
    REDIS_GPU_LOCK_BLOCKING_TIMEOUT = 5  # 5 seconds to wait for GPU lock in auto mode

    # Project structure
    PROJECTS_BASE_DIR = PROJECTS_DIR

    # Pipeline defaults
    DEFAULT_RAG_PRESET = "research"
    DEFAULT_TRAIN_RATIO = 0.8
    DEFAULT_MAX_SYMBOLS = 30
    DEFAULT_LANGUAGES = ["python", "r", "c", "cpp"]

    # Model defaults
    DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B"
    DEFAULT_LORA_RANK = 16
    DEFAULT_LORA_ALPHA = 32
    DEFAULT_MAX_SEQ_LENGTH = 8192
    DEFAULT_BATCH_SIZE = 2
    DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
    DEFAULT_NUM_TRAIN_EPOCHS = 3
    DEFAULT_LEARNING_RATE = 2e-4

    # Inference defaults
    DEFAULT_TOP_K = 3
    DEFAULT_MAX_NEW_TOKENS = 600
    DEFAULT_TEMPERATURE = 0.4
    DEFAULT_TOP_P = 0.9

    # Default chat settings (for chat without project)
    DEFAULT_RAG_INDEX_DIR = BASE_DIR / "labgpt-final-index"
    DEFAULT_MODEL_ADAPTER = "MogtabaAlim/llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba"


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

"""
DEPLOYMENT NOTES:

Solo Worker Command (Optimized for Memory & Performance):
    celery -A unified_app.celery_app worker \\
        -P solo \\
        -c 1 \\
        -O fair \\
        --max-tasks-per-child=1 \\
        --loglevel=info

What this does:
    -P solo: Solo pool (no prefork, eliminates memory duplication)
    -c 1: Concurrency of 1 (serialize all tasks, prevent concurrent GPU usage)
    -O fair: Fair scheduling (combined with prefetch_multiplier=1, prevents prefetching)
    --max-tasks-per-child=1: Restart worker after each task (prevents memory leaks)

Expected Results:
    - Memory usage: ~1.5 GB steady state (down from 3.2+ GB)
    - No concurrent GPU usage (Redis lock coordination)
    - Clean process termination (no orphans)
    - Tasks run sequentially: RAG → DataGen → Training
"""
