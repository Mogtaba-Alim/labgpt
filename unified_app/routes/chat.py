"""
Chat route - interface for chatting with fine-tuned model.
"""

from flask import Blueprint, render_template, redirect, url_for, flash
from unified_app.models import Project
from unified_app.services.state_manager import StateManager

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/')
def default_interface():
    """
    Default chat interface (no project).

    Uses:
    - Default LabGPT model from HuggingFace
    - Default RAG index (if available at labgpt-final-index/)
    - No project-specific data

    Shows:
    - Chat messages
    - Input area
    - Project selector dropdown
    - RAG configuration modal (if no RAG detected)
    """
    return render_template('chat/interface.html', project=None)


@chat_bp.route('/<int:project_id>')
def interface(project_id):
    """
    Project-specific chat interface.

    Shows:
    - Chat messages
    - Input area
    - RAG toggle
    - Citations (if RAG enabled)
    """
    project = Project.query.get_or_404(project_id)

    # Check if training is complete
    deps = StateManager.check_dependencies(project_id, 'chat')
    if not deps['allowed']:
        flash(f'Cannot access chat. Missing: {", ".join(deps["missing"])}', 'error')
        return redirect(url_for('pipelines.status', project_id=project_id))

    return render_template('chat/interface.html', project=project)
