"""
Celery application configuration for background task execution.
"""

from celery import Celery
from unified_app.config import Config


def make_celery(app_name=__name__):
    """
    Create and configure Celery application.

    Args:
        app_name: Name for the Celery application

    Returns:
        Configured Celery instance
    """
    celery = Celery(
        app_name,
        broker=Config.CELERY_BROKER_URL,
        backend=Config.CELERY_RESULT_BACKEND,
        include=['unified_app.tasks.pipeline_tasks']
    )

    celery.conf.update(
        task_track_started=Config.CELERY_TASK_TRACK_STARTED,
        task_time_limit=Config.CELERY_TASK_TIME_LIMIT,
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True,

        # Performance & Memory Optimization
        worker_prefetch_multiplier=1,  # No prefetching - reduces memory duplication
        task_acks_late=True,  # Acknowledge after completion - prevents task loss
        worker_max_tasks_per_child=1,  # Restart worker after each task - prevents memory leaks

        # Task Routing - all tasks use default 'celery' queue
        # GPU serialization is handled by Redis locks, not separate queues
        task_routes={
            'unified_app.tasks.pipeline_tasks.run_training_task': {'queue': 'celery'},
            'unified_app.tasks.pipeline_tasks.run_data_generation_task': {'queue': 'celery'},
            'unified_app.tasks.pipeline_tasks.run_rag_pipeline_task': {'queue': 'celery'},
        },
    )

    return celery


# Create celery instance
celery_app = make_celery()
