"""
Project model representing a complete LabGPT workflow instance.
"""

from datetime import datetime
from unified_app.extensions import db


class Project(db.Model):
    """
    A project represents one complete workflow through the LabGPT pipelines.
    Each project has its own directory with all generated artifacts.
    """
    __tablename__ = 'projects'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Project directory path (relative to PROJECTS_BASE_DIR)
    project_dir = db.Column(db.String(500), nullable=False, unique=True)

    # Pipeline completion status
    rag_completed = db.Column(db.Boolean, default=False, nullable=False)
    data_generation_completed = db.Column(db.Boolean, default=False, nullable=False)
    training_completed = db.Column(db.Boolean, default=False, nullable=False)

    # Relationships
    code_repos = db.relationship('CodeRepo', back_populates='project', cascade='all, delete-orphan')
    research_papers = db.relationship('ResearchPaper', back_populates='project', cascade='all, delete-orphan')
    lab_documents = db.relationship('LabDocument', back_populates='project', cascade='all, delete-orphan')
    jobs = db.relationship('Job', back_populates='project', cascade='all, delete-orphan')
    training_config = db.relationship('TrainingConfig', back_populates='project', uselist=False, cascade='all, delete-orphan')
    artifacts = db.relationship('Artifact', back_populates='project', cascade='all, delete-orphan')
    chat_messages = db.relationship('ChatMessage', back_populates='project', cascade='all, delete-orphan')
    section_drafts = db.relationship('SectionDraft', back_populates='project', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Project {self.id}: {self.name}>'

    def to_dict(self):
        """Convert project to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'project_dir': self.project_dir,
            'rag_completed': self.rag_completed,
            'data_generation_completed': self.data_generation_completed,
            'training_completed': self.training_completed,
            'num_code_repos': len(self.code_repos),
            'num_research_papers': len(self.research_papers),
            'num_lab_documents': len(self.lab_documents)
        }
