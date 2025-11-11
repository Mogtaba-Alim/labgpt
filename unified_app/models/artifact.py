"""
Artifact model for tracking generated outputs.
"""

from datetime import datetime
from unified_app.extensions import db


class Artifact(db.Model):
    """
    Represents a generated artifact (dataset, model, RAG index, etc.).
    """
    __tablename__ = 'artifacts'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)

    # Artifact identification
    artifact_type = db.Column(db.String(50), nullable=False)
    # Types: 'rag_index', 'training_data', 'trained_model', 'tensorboard_logs'

    artifact_name = db.Column(db.String(200), nullable=False)
    artifact_path = db.Column(db.String(1000), nullable=False)

    # Metadata
    file_size = db.Column(db.Integer)  # Size in bytes
    metadata_json = db.Column(db.Text)  # JSON string with additional info

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = db.relationship('Project', back_populates='artifacts')

    def __repr__(self):
        return f'<Artifact {self.id}: {self.artifact_type} - {self.artifact_name}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'artifact_type': self.artifact_type,
            'artifact_name': self.artifact_name,
            'artifact_path': self.artifact_path,
            'file_size': self.file_size,
            'metadata': self.metadata_json,
            'created_at': self.created_at.isoformat()
        }
