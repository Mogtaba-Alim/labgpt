"""
Code repository model for tracking source code inputs.
"""

from datetime import datetime
from unified_app.extensions import db


class CodeRepo(db.Model):
    """
    Represents a code repository (local path or GitHub URL).
    """
    __tablename__ = 'code_repos'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)

    # Source information
    source_type = db.Column(db.String(20), nullable=False)  # 'local' or 'github'
    source_path = db.Column(db.String(1000), nullable=False)  # Original path/URL
    local_path = db.Column(db.String(1000), nullable=False)  # Path where code is stored

    # Metadata
    languages = db.Column(db.String(200))  # Comma-separated: "python,r,c"
    num_files = db.Column(db.Integer)
    num_symbols = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = db.relationship('Project', back_populates='code_repos')

    def __repr__(self):
        return f'<CodeRepo {self.id}: {self.source_type} - {self.source_path}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'source_type': self.source_type,
            'source_path': self.source_path,
            'local_path': self.local_path,
            'languages': self.languages.split(',') if self.languages else [],
            'num_files': self.num_files,
            'num_symbols': self.num_symbols,
            'created_at': self.created_at.isoformat()
        }
