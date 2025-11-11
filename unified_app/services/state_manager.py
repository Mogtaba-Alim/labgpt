"""
State management service for tracking project progress and dependencies.
"""

from typing import Dict, List, Optional
from unified_app.extensions import db
from unified_app.models import Project, Job


class StateManager:
    """
    Manages project state, pipeline dependencies, and navigation guards.
    """

    # Define pipeline dependencies
    DEPENDENCIES = {
        'training': ['data_generation'],
        'chat': ['training'],
        'grant': ['rag', 'training']
    }

    @staticmethod
    def check_dependencies(project_id: int, pipeline: str) -> Dict[str, any]:
        """
        Check if all dependencies are met for a given pipeline.

        Args:
            project_id: Project ID
            pipeline: Pipeline name ('training', 'chat', or 'grant')

        Returns:
            Dict with 'allowed' (bool) and 'missing' (list of missing dependencies)
        """
        project = Project.query.get(project_id)
        if not project:
            return {'allowed': False, 'missing': ['project_not_found']}

        required = StateManager.DEPENDENCIES.get(pipeline, [])
        missing = []

        for dep in required:
            if dep == 'data_generation' and not project.data_generation_completed:
                missing.append('data_generation')
            elif dep == 'rag' and not project.rag_completed:
                missing.append('rag')
            elif dep == 'training' and not project.training_completed:
                missing.append('training')

        return {
            'allowed': len(missing) == 0,
            'missing': missing
        }

    @staticmethod
    def get_project_status(project_id: int) -> Dict[str, any]:
        """
        Get comprehensive status of a project.

        Args:
            project_id: Project ID

        Returns:
            Dict with project info, completion status, and active jobs
        """
        project = Project.query.get(project_id)
        if not project:
            return None

        # Get active jobs
        active_jobs = Job.query.filter_by(
            project_id=project_id
        ).filter(
            Job.status.in_(['pending', 'running'])
        ).all()

        # Get latest completed jobs
        latest_jobs = {}
        for job_type in ['rag', 'data_generation', 'training']:
            latest = Job.query.filter_by(
                project_id=project_id,
                job_type=job_type
            ).order_by(Job.created_at.desc()).first()
            if latest:
                latest_jobs[job_type] = latest.to_dict()

        return {
            'project': project.to_dict(),
            'completion_status': {
                'rag': project.rag_completed,
                'data_generation': project.data_generation_completed,
                'training': project.training_completed
            },
            'active_jobs': [job.to_dict() for job in active_jobs],
            'latest_jobs': latest_jobs,
            'can_access': {
                'training': StateManager.check_dependencies(project_id, 'training')['allowed'],
                'chat': StateManager.check_dependencies(project_id, 'chat')['allowed'],
                'grant': StateManager.check_dependencies(project_id, 'grant')['allowed']
            }
        }

    @staticmethod
    def mark_pipeline_complete(project_id: int, pipeline: str):
        """
        Mark a pipeline as completed for a project.

        Args:
            project_id: Project ID
            pipeline: Pipeline name ('rag', 'data_generation', or 'training')
        """
        project = Project.query.get(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        if pipeline == 'rag':
            project.rag_completed = True
        elif pipeline == 'data_generation':
            project.data_generation_completed = True
        elif pipeline == 'training':
            project.training_completed = True
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        db.session.commit()

    @staticmethod
    def get_job_by_celery_id(celery_task_id: str) -> Optional[Job]:
        """
        Retrieve a job by its Celery task ID.

        Args:
            celery_task_id: Celery task ID

        Returns:
            Job object or None
        """
        return Job.query.filter_by(celery_task_id=celery_task_id).first()

    @staticmethod
    def get_active_jobs(project_id: int, job_type: Optional[str] = None) -> List[Job]:
        """
        Get all active jobs for a project.

        Args:
            project_id: Project ID
            job_type: Optional filter by job type

        Returns:
            List of active Job objects
        """
        query = Job.query.filter_by(project_id=project_id).filter(
            Job.status.in_(['pending', 'running'])
        )

        if job_type:
            query = query.filter_by(job_type=job_type)

        return query.all()

    @staticmethod
    def create_job(project_id: int, job_type: str, celery_task_id: str, log_file: str = None) -> Job:
        """
        Create a new job entry.

        Args:
            project_id: Project ID
            job_type: Type of job ('rag', 'data_generation', 'training')
            celery_task_id: Celery task ID
            log_file: Optional path to log file

        Returns:
            Created Job object
        """
        job = Job(
            project_id=project_id,
            job_type=job_type,
            celery_task_id=celery_task_id,
            status='pending',
            log_file=log_file
        )
        db.session.add(job)
        db.session.commit()
        return job

    @staticmethod
    def update_job_progress(job_id: int, progress: int, step: str = None):
        """
        Update job progress.

        Args:
            job_id: Job ID
            progress: Progress percentage (0-100)
            step: Optional current step description
        """
        job = Job.query.get(job_id)
        if job:
            job.update_progress(progress, step)
