"""
Job model for tracking background pipeline execution.
"""

from datetime import datetime
from unified_app.extensions import db


class Job(db.Model):
    """
    Represents a background job (RAG, data generation, or training pipeline).
    """
    __tablename__ = 'jobs'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)

    # Job identification
    celery_task_id = db.Column(db.String(200), unique=True, nullable=False)
    job_type = db.Column(db.String(50), nullable=False)  # 'rag', 'data_generation', 'training'

    # Job status
    status = db.Column(db.String(20), default='pending', nullable=False)
    # Status values: 'pending', 'running', 'completed', 'failed'

    # Progress tracking
    progress_percentage = db.Column(db.Integer, default=0)
    current_step = db.Column(db.String(500))
    log_file = db.Column(db.String(1000))  # Path to log file

    # Result information
    result_data = db.Column(db.Text)  # JSON string with results
    error_message = db.Column(db.Text)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)

    # Relationships
    project = db.relationship('Project', back_populates='jobs')

    def __repr__(self):
        return f'<Job {self.id}: {self.job_type} - {self.status}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'celery_task_id': self.celery_task_id,
            'job_type': self.job_type,
            'status': self.status,
            'progress_percentage': self.progress_percentage,
            'current_step': self.current_step,
            'log_file': self.log_file,
            'result_data': self.result_data,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

    def update_progress(self, percentage, step=None):
        """Update job progress"""
        self.progress_percentage = percentage
        if step:
            self.current_step = step
        db.session.commit()

    def mark_running(self):
        """Mark job as running"""
        self.status = 'running'
        self.started_at = datetime.utcnow()
        db.session.commit()

    def mark_completed(self, result_data=None):
        """Mark job as completed"""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100
        if result_data:
            self.result_data = result_data
        db.session.commit()

    def mark_failed(self, error_message):
        """Mark job as failed"""
        self.status = 'failed'
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        db.session.commit()
