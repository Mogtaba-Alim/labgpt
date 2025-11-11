"""
Welcome page route - landing page for the unified LabGPT application.
"""

from flask import Blueprint, render_template, redirect, url_for, request, flash
from unified_app.extensions import db
from unified_app.models import Project
from unified_app.services.file_manager import FileManager
from datetime import datetime

welcome_bp = Blueprint('welcome', __name__)


@welcome_bp.route('/')
def index():
    """
    Landing page showing existing projects and option to create new project.
    """
    # Get all projects, ordered by most recent
    projects = Project.query.order_by(Project.updated_at.desc()).all()

    return render_template('welcome/index.html', projects=projects)


@welcome_bp.route('/create_project', methods=['POST'])
def create_project():
    """
    Create a new project and redirect to data configuration page.
    """
    project_name = request.form.get('project_name', '').strip()
    description = request.form.get('description', '').strip()

    if not project_name:
        flash('Project name is required', 'error')
        return redirect(url_for('welcome.index'))

    try:
        # Create project directory structure
        project_dir = FileManager.create_project_structure(project_name)

        # Create project in database
        project = Project(
            name=project_name,
            description=description,
            project_dir=project_dir.name
        )
        db.session.add(project)
        db.session.commit()

        flash(f'Project "{project_name}" created successfully!', 'success')
        return redirect(url_for('data_config.configure', project_id=project.id))

    except Exception as e:
        db.session.rollback()
        flash(f'Error creating project: {str(e)}', 'error')
        return redirect(url_for('welcome.index'))


@welcome_bp.route('/project/<int:project_id>')
def view_project(project_id):
    """
    View project details and status.
    """
    project = Project.query.get_or_404(project_id)

    return render_template('welcome/project_details.html', project=project)


@welcome_bp.route('/delete_project/<int:project_id>', methods=['POST'])
def delete_project(project_id):
    """
    Delete a project and all associated data.
    """
    project = Project.query.get_or_404(project_id)

    try:
        # Delete project from database (cascades to all related records)
        db.session.delete(project)
        db.session.commit()

        # TODO: Optionally delete project directory from filesystem

        flash(f'Project "{project.name}" deleted successfully', 'success')

    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting project: {str(e)}', 'error')

    return redirect(url_for('welcome.index'))
