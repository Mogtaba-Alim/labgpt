"""
Grant section draft model for versioned grant writing.
"""

from datetime import datetime
from unified_app.extensions import db


class SectionDraft(db.Model):
    """
    Represents a draft version of a grant section.
    Supports versioning with parent-child relationships.
    """
    __tablename__ = 'section_drafts'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)

    # Section identification
    section_name = db.Column(db.String(200), nullable=False)
    # Sections: Background, Objectives, Specific Aims, Methods, Preliminary Work,
    # Impact/Relevance, Feasibility/Risks, Outcomes/Future, Data Management,
    # Expertise/Resources, Progress Summary, Lay Abstract

    # Versioning
    version = db.Column(db.Integer, default=1, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('section_drafts.id'))
    is_current = db.Column(db.Boolean, default=True, nullable=False)

    # Content
    content = db.Column(db.Text, nullable=False)
    citations = db.Column(db.Text)  # JSON string with citation sources

    # Generation metadata
    prompt_used = db.Column(db.Text)
    model_used = db.Column(db.String(200))
    temperature = db.Column(db.Float)

    # Feedback and improvements
    feedback = db.Column(db.Text)
    improvement_notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = db.relationship('Project', back_populates='section_drafts')
    parent = db.relationship('SectionDraft', remote_side=[id], backref='children')

    def __repr__(self):
        return f'<SectionDraft {self.id}: {self.section_name} v{self.version}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'section_name': self.section_name,
            'version': self.version,
            'parent_id': self.parent_id,
            'is_current': self.is_current,
            'content': self.content,
            'citations': self.citations,
            'prompt_used': self.prompt_used,
            'model_used': self.model_used,
            'temperature': self.temperature,
            'feedback': self.feedback,
            'improvement_notes': self.improvement_notes,
            'created_at': self.created_at.isoformat()
        }

    @staticmethod
    def create_new_version(project_id, section_name, content, parent_draft=None, **kwargs):
        """
        Create a new version of a section draft.

        Args:
            project_id: Project ID
            section_name: Name of the section
            content: Draft content
            parent_draft: Parent SectionDraft object (for versioning)
            **kwargs: Additional metadata (citations, prompt_used, etc.)

        Returns:
            New SectionDraft instance
        """
        # Mark all previous versions as not current
        db.session.query(SectionDraft).filter_by(
            project_id=project_id,
            section_name=section_name,
            is_current=True
        ).update({'is_current': False})

        # Determine version number
        if parent_draft:
            version = parent_draft.version + 1
            parent_id = parent_draft.id
        else:
            # Find max version for this section
            max_version = db.session.query(db.func.max(SectionDraft.version)).filter_by(
                project_id=project_id,
                section_name=section_name
            ).scalar() or 0
            version = max_version + 1
            parent_id = None

        # Create new draft
        new_draft = SectionDraft(
            project_id=project_id,
            section_name=section_name,
            version=version,
            parent_id=parent_id,
            content=content,
            is_current=True,
            **kwargs
        )

        db.session.add(new_draft)
        db.session.commit()

        return new_draft
