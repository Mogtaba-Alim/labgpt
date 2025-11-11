"""
Pipeline status route - monitor RAG and data generation progress.
"""

from flask import Blueprint, render_template
from unified_app.models import Project

pipelines_bp = Blueprint('pipelines', __name__)


@pipelines_bp.route('/<int:project_id>')
def status(project_id):
    """
    Pipeline status page with tabs for RAG and data generation.
    Shows real-time progress, current step, and logs.

    The page polls /api/job/<id>/status every 2 seconds to update progress.
    """
    project = Project.query.get_or_404(project_id)
    return render_template('pipelines/status.html', project=project)
