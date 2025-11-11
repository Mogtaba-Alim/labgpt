"""
Grant generation route - write grant proposals with RAG citations.
"""

from flask import Blueprint, render_template, redirect, url_for, flash
from unified_app.models import Project
from unified_app.services.state_manager import StateManager

grant_bp = Blueprint('grant', __name__)


@grant_bp.route('/<int:project_id>')
def interface(project_id):
    """
    Grant generation interface.

    Shows:
    - Grant sections sidebar
    - Section editor and generator
    - RAG citations
    - Export functionality
    """
    project = Project.query.get_or_404(project_id)

    # Check if both RAG and training are complete
    deps = StateManager.check_dependencies(project_id, 'grant')
    if not deps['allowed']:
        flash(f'Cannot access grant generation. Missing: {", ".join(deps["missing"])}', 'error')
        return redirect(url_for('pipelines.status', project_id=project_id))

    return render_template('grant/interface.html', project=project)
