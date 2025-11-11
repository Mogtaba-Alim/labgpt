"""
Document models for research papers and lab documents.
"""

from datetime import datetime
from unified_app.extensions import db


class ResearchPaper(db.Model):
    """
    Represents a research paper used for RAG and data generation.
    """
    __tablename__ = 'research_papers'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)

    # File information
    filename = db.Column(db.String(500), nullable=False)
    file_path = db.Column(db.String(1000), nullable=False)  # Path to stored file
    file_size = db.Column(db.Integer)  # Size in bytes
    file_type = db.Column(db.String(10))  # 'pdf', 'txt', 'md'

    # Metadata extracted during processing
    title = db.Column(db.String(1000))
    authors = db.Column(db.Text)
    num_pages = db.Column(db.Integer)
    num_chunks = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = db.relationship('Project', back_populates='research_papers')

    def __repr__(self):
        return f'<ResearchPaper {self.id}: {self.filename}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'title': self.title,
            'authors': self.authors,
            'num_pages': self.num_pages,
            'num_chunks': self.num_chunks,
            'created_at': self.created_at.isoformat()
        }


class LabDocument(db.Model):
    """
    Represents a lab document used for RAG indexing.
    """
    __tablename__ = 'lab_documents'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)

    # File information
    filename = db.Column(db.String(500), nullable=False)
    file_path = db.Column(db.String(1000), nullable=False)  # Path to stored file
    file_size = db.Column(db.Integer)  # Size in bytes
    file_type = db.Column(db.String(10))  # 'pdf', 'txt', 'md', 'py', etc.

    # Metadata
    num_chunks = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = db.relationship('Project', back_populates='lab_documents')

    def __repr__(self):
        return f'<LabDocument {self.id}: {self.filename}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'num_chunks': self.num_chunks,
            'created_at': self.created_at.isoformat()
        }
