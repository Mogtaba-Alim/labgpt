"""
Celery tasks for running pipelines in the background.
"""

import json
import redis
from pathlib import Path
from datetime import datetime
from unified_app.celery_app import celery_app
from unified_app.services.orchestration_service import OrchestrationService
from unified_app.services.state_manager import StateManager
from unified_app.models import Job, Project, Artifact
from unified_app.extensions import db
from unified_app.config import Config


def get_redis_client():
    """
    Get Redis client for GPU lock coordination.

    Returns:
        redis.Redis: Redis client instance
    """
    return redis.Redis.from_url(Config.CELERY_BROKER_URL, decode_responses=True)


def get_flask_app():
    """
    Lazy load Flask app to avoid circular imports.
    Creates app only when first task runs, not at module import time.
    """
    if not hasattr(get_flask_app, '_app'):
        from unified_app.app import create_app
        get_flask_app._app = create_app('development')
    return get_flask_app._app


@celery_app.task(bind=True)
def run_rag_pipeline_task(
    self,
    project_id: int,
    papers_paths: list,
    lab_docs_paths: list,
    rag_preset: str = "research"
):
    """
    Background task for running RAG indexing pipeline.

    Args:
        self: Celery task instance
        project_id: Project ID
        papers_paths: List of paths to research papers
        lab_docs_paths: List of paths to lab documents
        rag_preset: RAG preset ('default' or 'research')

    Returns:
        Dict with result information
    """
    with get_flask_app().app_context():
        # Get job from database
        job = StateManager.get_job_by_celery_id(self.request.id)
        if not job:
            return {'success': False, 'error': 'Job not found in database'}

        try:
            # Mark job as running
            job.mark_running()

            # Get project and create orchestration service
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            project_dir = Config.PROJECTS_BASE_DIR / project.project_dir
            orchestrator = OrchestrationService(project_dir)

            # Predict log file path so UI can start reading it immediately
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = project_dir / 'logs' / f'rag_ingestion_{timestamp}.log'
            job.log_file = str(log_file_path)
            db.session.commit()

            # Update progress
            job.update_progress(10, "Starting RAG pipeline...")

            # Run RAG pipeline (this is where most time is spent)
            # Update progress to show it's running
            job.update_progress(20, "Processing documents (this may take several minutes)...")

            result = orchestrator.run_rag_pipeline(
                project_id=project_id,
                papers_paths=papers_paths,
                lab_docs_paths=lab_docs_paths,
                rag_preset=rag_preset
            )

            if not result['success']:
                raise Exception(result.get('error', 'RAG pipeline failed'))

            # Update log file path if different from prediction (unlikely but possible)
            if result.get('log_file') and result['log_file'] != job.log_file:
                job.log_file = result['log_file']
                db.session.commit()

            # Update progress
            job.update_progress(90, "Saving RAG artifacts...")

            # Create artifact entry
            artifact = Artifact(
                project_id=project_id,
                artifact_type='rag_index',
                artifact_name='RAG Index',
                artifact_path=result['index_dir'],
                metadata=json.dumps({
                    'num_documents': result.get('num_documents', 0),
                    'num_chunks': result.get('num_chunks', 0),
                    'preset': rag_preset
                })
            )
            db.session.add(artifact)

            # Mark pipeline as completed in project
            StateManager.mark_pipeline_complete(project_id, 'rag')

            # Mark job as completed
            job.mark_completed(json.dumps(result))

            return {'success': True, 'result': result}

        except Exception as e:
            # Mark job as failed
            job.mark_failed(str(e))
            return {'success': False, 'error': str(e)}


@celery_app.task(bind=True)
def run_data_generation_task(
    self,
    project_id: int,
    code_repo_paths: list,
    papers_dir: str = None,
    max_symbols: int = 30,
    languages: list = None
):
    """
    Background task for running data generation pipeline.

    Args:
        self: Celery task instance
        project_id: Project ID
        code_repo_paths: List of paths to code repositories
        papers_dir: Optional path to papers directory
        max_symbols: Maximum symbols per file
        languages: List of languages to process

    Returns:
        Dict with result information
    """
    with get_flask_app().app_context():
        # Get job from database
        job = StateManager.get_job_by_celery_id(self.request.id)
        if not job:
            return {'success': False, 'error': 'Job not found in database'}

        try:
            # Mark job as running
            job.mark_running()

            # Get project and create orchestration service
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            project_dir = Config.PROJECTS_BASE_DIR / project.project_dir
            orchestrator = OrchestrationService(project_dir)

            # Predict log file path so UI can start reading it immediately
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = project_dir / 'logs' / f'data_generation_{timestamp}.log'
            job.log_file = str(log_file_path)
            db.session.commit()

            # Update progress
            job.update_progress(10, "Starting data generation pipeline...")

            # Run data generation pipeline (this is where most time is spent)
            # Update progress to show it's running
            job.update_progress(20, "Generating synthetic training data (this may take 10-30 minutes)...")

            result = orchestrator.run_data_generation(
                code_repo_paths=code_repo_paths,
                papers_dir=papers_dir,
                max_symbols=max_symbols,
                languages=languages or Config.DEFAULT_LANGUAGES
            )

            if not result['success']:
                raise Exception(result.get('error', 'Data generation pipeline failed'))

            # Update log file path if different from prediction (unlikely but possible)
            if result.get('log_file') and result['log_file'] != job.log_file:
                job.log_file = result['log_file']
                db.session.commit()

            # Update progress
            job.update_progress(90, "Saving generated data artifacts...")

            # Create artifact entries for train and val files
            if result.get('train_file'):
                train_artifact = Artifact(
                    project_id=project_id,
                    artifact_type='training_data',
                    artifact_name='Training Data',
                    artifact_path=result['train_file'],
                    metadata=json.dumps({
                        'type': 'train',
                        'num_examples': result.get('num_examples', 0) // 2
                    })
                )
                db.session.add(train_artifact)

            if result.get('val_file'):
                val_artifact = Artifact(
                    project_id=project_id,
                    artifact_type='training_data',
                    artifact_name='Validation Data',
                    artifact_path=result['val_file'],
                    metadata=json.dumps({
                        'type': 'validation',
                        'num_examples': result.get('num_examples', 0) // 2
                    })
                )
                db.session.add(val_artifact)

            # Mark pipeline as completed in project
            StateManager.mark_pipeline_complete(project_id, 'data_generation')

            # Mark job as completed
            job.mark_completed(json.dumps(result))

            return {'success': True, 'result': result}

        except Exception as e:
            # Mark job as failed
            job.mark_failed(str(e))
            return {'success': False, 'error': str(e)}


@celery_app.task(bind=True)
def run_training_task(
    self,
    project_id: int,
    training_config: dict
):
    """
    Background task for running model training.

    GPU-bound task: Acquires Redis GPU lock to prevent concurrent GPU usage.

    Args:
        self: Celery task instance
        project_id: Project ID
        training_config: Dict with training configuration

    Returns:
        Dict with result information
    """
    with get_flask_app().app_context():
        # Get job from database
        job = StateManager.get_job_by_celery_id(self.request.id)
        if not job:
            return {'success': False, 'error': 'Job not found in database'}

        # Acquire GPU lock
        redis_client = get_redis_client()
        gpu_lock = redis_client.lock(
            Config.REDIS_GPU_LOCK_KEY,
            timeout=Config.REDIS_GPU_LOCK_TIMEOUT,
            blocking_timeout=None  # Wait indefinitely for GPU
        )

        try:
            # Acquire GPU lock (blocks if GPU is busy)
            if not gpu_lock.acquire(blocking=True):
                raise Exception("Failed to acquire GPU lock")

            # Mark job as running
            job.mark_running()

            # Get project and create orchestration service
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            project_dir = Config.PROJECTS_BASE_DIR / project.project_dir
            orchestrator = OrchestrationService(project_dir)

            # Predict log file path so UI can start reading it immediately
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = project_dir / 'logs' / f'training_{timestamp}.log'
            job.log_file = str(log_file_path)
            db.session.commit()

            # Update progress
            job.update_progress(10, "Starting training pipeline (GPU acquired)...")

            # Determine data files
            if training_config.get('use_generated_data', True):
                data_files = orchestrator.get_generated_data_files()
                train_file = data_files['train_file']
                val_file = data_files['val_file']
            else:
                train_file = training_config.get('custom_train_file')
                val_file = training_config.get('custom_val_file')

            if not train_file or not val_file:
                raise ValueError("Training data files not found")

            # Prepare training arguments
            training_kwargs = {
                'use_lora': training_config.get('use_lora', True),
                'lora_rank': training_config.get('lora_rank', 16),
                'lora_alpha': training_config.get('lora_alpha', 32),
                'lora_dropout': training_config.get('lora_dropout', 0.05),
                'use_4bit': training_config.get('use_4bit', True),
                'max_seq_length': training_config.get('max_seq_length', 8192),
                'batch_size': training_config.get('batch_size', 2),
                'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 4),
                'num_train_epochs': training_config.get('num_train_epochs', 3),
                'learning_rate': training_config.get('learning_rate', 2e-4),
                'warmup_ratio': training_config.get('warmup_ratio', 0.1),
                'save_steps': training_config.get('save_steps', 500),
                'logging_steps': training_config.get('logging_steps', 10)
            }

            # Update progress before long-running training
            job.update_progress(20, "Training model (this may take several hours)...")

            # Run training pipeline
            result = orchestrator.run_training(
                train_file=train_file,
                val_file=val_file,
                base_model=training_config.get('base_model', Config.DEFAULT_BASE_MODEL),
                output_model_name=training_config.get('output_model_name', 'trained_model'),
                **training_kwargs
            )

            if not result['success']:
                raise Exception(result.get('error', 'Training pipeline failed'))

            # Update log file path if different from prediction (unlikely but possible)
            if result.get('log_file') and result['log_file'] != job.log_file:
                job.log_file = result['log_file']
                db.session.commit()

            # Update progress
            job.update_progress(95, "Saving trained model artifacts...")

            # Create artifact entry for trained model
            model_artifact = Artifact(
                project_id=project_id,
                artifact_type='trained_model',
                artifact_name=training_config.get('output_model_name', 'Trained Model'),
                artifact_path=result['model_path'],
                metadata=json.dumps({
                    'base_model': training_config.get('base_model'),
                    'final_loss': result.get('final_loss'),
                    'total_steps': result.get('total_steps'),
                    'num_epochs': training_config.get('num_train_epochs')
                })
            )
            db.session.add(model_artifact)

            # Mark pipeline as completed in project
            StateManager.mark_pipeline_complete(project_id, 'training')

            # Mark job as completed
            job.mark_completed(json.dumps(result))

            return {'success': True, 'result': result}

        except Exception as e:
            # Mark job as failed
            job.mark_failed(str(e))
            return {'success': False, 'error': str(e)}
        finally:
            # Always release GPU lock
            try:
                gpu_lock.release()
            except:
                pass  # Lock may have already expired or been released
