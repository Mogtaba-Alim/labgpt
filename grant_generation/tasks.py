"""
tasks.py

Celery async tasks for grant generation pipeline.
Handles background document indexing so users don't wait for RAG indexing to complete.
"""

import os
import logging
from typing import List
from celery import Celery

logger = logging.getLogger(__name__)

# Configure Celery
# Use Redis as broker and backend (install: pip install redis celery)
# Start Redis: redis-server
# Start worker: celery -A grant_generation.tasks worker --loglevel=info
celery_app = Celery(
    'grant_generation',
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
)


@celery_app.task(bind=True, name='grant_generation.index_documents')
def index_documents_async(self, project_id: str, file_paths: List[str], base_rag_index: str = None):
    """
    Background task for indexing uploaded documents into project RAG.

    Updates task state for progress tracking in UI.

    Args:
        self: Celery task instance (injected when bind=True)
        project_id: Project identifier
        file_paths: List of file paths to index
        base_rag_index: Base RAG index directory (for corpus manager)

    Returns:
        Dict with:
        - status: 'complete'
        - files_processed: Number of files indexed
        - total_chunks: Total chunks created
        - cache_hits: Embedding cache hits
        - cache_misses: New embeddings generated
        - time_seconds: Indexing time

    Task States:
    - PENDING: Task waiting to be executed
    - PROCESSING: Task is running (custom state)
    - SUCCESS: Task completed successfully
    - FAILURE: Task failed with error
    """
    from .corpus_manager import GrantCorpusManager
    from .database import GrantDatabase

    logger.info(f"Starting indexing task for project {project_id} with {len(file_paths)} files")

    # Update state to PROCESSING
    self.update_state(
        state='PROCESSING',
        meta={
            'current': 0,
            'total': len(file_paths),
            'status': 'Initializing...'
        }
    )

    try:
        # Initialize managers
        corpus_manager = GrantCorpusManager(
            base_index_dir=base_rag_index or os.environ.get('BASE_RAG_INDEX'),
            projects_dir='grant_generation/projects'
        )
        db = GrantDatabase('grant_generation/grants.db')

        # Update state: starting indexing
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 0,
                'total': len(file_paths),
                'status': 'Indexing documents into RAG...'
            }
        )

        # Add documents to RAG (benefits from embedding cache)
        stats = corpus_manager.add_documents(project_id, file_paths)

        # Calculate per-file chunk count (approximate)
        chunks_per_file = stats['total_chunks'] // len(file_paths) if len(file_paths) > 0 else 0

        # Update database: mark documents as indexed
        for i, file_path in enumerate(file_paths):
            filename = os.path.basename(file_path)

            # Determine document type from filename heuristics
            filename_lower = filename.lower()
            if 'biosketch' in filename_lower or 'cv' in filename_lower:
                doc_type = 'biosketch'
            elif 'facilities' in filename_lower or 'resources' in filename_lower:
                doc_type = 'facilities'
            elif 'grant' in filename_lower or 'proposal' in filename_lower:
                doc_type = 'grant'
            else:
                doc_type = 'document'

            # Add to database
            db.add_document(
                project_id=project_id,
                filename=filename,
                file_path=file_path,
                doc_type=doc_type,
                indexed=True,
                chunk_count=chunks_per_file
            )

            # Update progress
            self.update_state(
                state='PROCESSING',
                meta={
                    'current': i + 1,
                    'total': len(file_paths),
                    'status': f'Indexed {i+1}/{len(file_paths)} files'
                }
            )

            logger.info(f"Indexed {filename} ({i+1}/{len(file_paths)})")

        # Log cache performance
        cache_hits = stats.get('cache_hits', 0)
        cache_misses = stats.get('cache_misses', 0)
        total_embeddings = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_embeddings * 100) if total_embeddings > 0 else 0

        logger.info(
            f"Indexing complete for project {project_id}: "
            f"{stats['files_processed']} files, {stats['total_chunks']} chunks, "
            f"cache hit rate: {cache_hit_rate:.1f}%"
        )

        # Return success result
        return {
            'status': 'complete',
            'files_processed': stats['files_processed'],
            'total_chunks': stats['total_chunks'],
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'time_seconds': stats.get('time_seconds', 0),
            'chunks_per_second': stats.get('chunks_per_second', 0)
        }

    except Exception as e:
        # Log error
        logger.error(f"Error during indexing task for project {project_id}: {e}", exc_info=True)

        # Update state to FAILURE
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'error_type': type(e).__name__
            }
        )

        # Re-raise exception so Celery marks task as failed
        raise


@celery_app.task(name='grant_generation.cleanup_old_projects')
def cleanup_old_projects(days: int = 30):
    """
    Periodic task to clean up old inactive projects.

    Args:
        days: Delete projects inactive for this many days

    Returns:
        Dict with cleanup statistics
    """
    from .database import GrantDatabase
    from .corpus_manager import GrantCorpusManager
    import shutil
    from datetime import datetime, timedelta

    logger.info(f"Starting cleanup of projects inactive for {days}+ days")

    try:
        db = GrantDatabase('grant_generation/grants.db')
        corpus_manager = GrantCorpusManager(
            base_index_dir=os.environ.get('BASE_RAG_INDEX'),
            projects_dir='grant_generation/projects'
        )

        # Query for old projects
        cutoff_date = datetime.now() - timedelta(days=days)

        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT project_id, title, updated_at
            FROM projects
            WHERE updated_at < ? AND status = 'active'
        """, (cutoff_date.isoformat(),))

        old_projects = cursor.fetchall()

        deleted_count = 0
        for project in old_projects:
            project_id = project['project_id']
            title = project['title']

            try:
                # Delete project files
                corpus_manager.delete_project(project_id)

                # Mark as archived in database (don't delete records)
                db.update_project(project_id, status='archived')

                deleted_count += 1
                logger.info(f"Archived project {project_id}: {title}")

            except Exception as e:
                logger.error(f"Error archiving project {project_id}: {e}")

        logger.info(f"Cleanup complete: {deleted_count} projects archived")

        return {
            'status': 'complete',
            'projects_checked': len(old_projects),
            'projects_archived': deleted_count,
            'cutoff_days': days
        }

    except Exception as e:
        logger.error(f"Error during cleanup task: {e}", exc_info=True)
        raise


# Configure periodic tasks (requires celery beat)
# Start beat: celery -A grant_generation.tasks beat --loglevel=info
celery_app.conf.beat_schedule = {
    'cleanup-old-projects-monthly': {
        'task': 'grant_generation.cleanup_old_projects',
        'schedule': 86400 * 30,  # Every 30 days
        'args': (30,)  # Delete projects inactive for 30 days
    },
}
