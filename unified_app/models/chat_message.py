"""
Chat message model for storing conversation history.
"""

from datetime import datetime
from unified_app.extensions import db


class ChatMessage(db.Model):
    """
    Represents a message in the chat interface.
    """
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)

    # Message content
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)

    # RAG information (if RAG was used)
    used_rag = db.Column(db.Boolean, default=False, nullable=False)
    citations = db.Column(db.Text)  # JSON string with citation info

    # Token usage
    prompt_tokens = db.Column(db.Integer)
    completion_tokens = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = db.relationship('Project', back_populates='chat_messages')

    def __repr__(self):
        return f'<ChatMessage {self.id}: {self.role}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'used_rag': self.used_rag,
            'citations': self.citations,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'created_at': self.created_at.isoformat()
        }
