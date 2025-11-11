"""
Data configuration route - collect code repos, papers, and lab documents paths.
Launches RAG and data generation pipelines using labgpt_cli.py.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from pathlib import Path
from unified_app.extensions import db
from unified_app.models import Project, CodeRepo
from unified_app.services.state_manager import StateManager
from unified_app.tasks.pipeline_tasks import run_rag_pipeline_task, run_data_generation_task
from unified_app.config import Config

data_config_bp = Blueprint('data_config', __name__)


@data_config_bp.route('/<int:project_id>')
def configure(project_id):
    """
    Data configuration page with three sections:
    1. Code repositories (local paths or GitHub URLs)
    2. Research papers (directory path)
    3. Lab documents (directory path)
    """
    project = Project.query.get_or_404(project_id)
    return render_template('data_config/configure.html', project=project)


@data_config_bp.route('/<int:project_id>/submit', methods=['POST'])
def submit_configuration(project_id):
    """
    Process submitted data configuration and start RAG + data generation pipelines.

    Workflow:
    1. Extract form data (code repos, papers dir, lab docs dir)
    2. Validate paths
    3. Register code repos in database
    4. Start RAG pipeline (Celery task) - uses labgpt_cli.py
    5. Start data generation pipeline (Celery task) - uses labgpt_cli.py
    6. Redirect to pipeline status page
    """
    project = Project.query.get_or_404(project_id)
    project_dir = Config.PROJECTS_BASE_DIR / project.project_dir

    try:
        # Extract code repository paths from form
        code_repos = []
        for key in request.form:
            if key.startswith('repo_'):
                repo_path = request.form[key].strip()
                if repo_path:
                    code_repos.append(repo_path)

        # Code repos are now optional - check if we have existing training data instead
        existing_data_mode = request.form.get('existing_data', 'no')
        existing_data_path = None
        existing_data_file = None

        if existing_data_mode == 'path':
            existing_data_path = request.form.get('existing_data_path', '').strip()
        elif existing_data_mode == 'upload':
            existing_data_file = request.files.get('existing_data_file')

        # Validate: need either code repos OR existing training data
        if not code_repos and not existing_data_path and not existing_data_file:
            flash('Please provide either code repositories or existing training data', 'error')
            return redirect(url_for('data_config.configure', project_id=project_id))

        # Extract papers directory path
        papers_type = request.form.get('papers_type', 'none')
        papers_dir = None
        if papers_type == 'directory':
            papers_dir = request.form.get('papers_dir', '').strip()
            if papers_dir and not Path(papers_dir).exists():
                flash(f'Papers directory does not exist: {papers_dir}', 'error')
                return redirect(url_for('data_config.configure', project_id=project_id))

        # Extract lab documents directory path
        labdocs_type = request.form.get('labdocs_type', 'none')
        labdocs_dir = None
        if labdocs_type == 'directory':
            labdocs_dir = request.form.get('labdocs_dir', '').strip()
            if labdocs_dir and not Path(labdocs_dir).exists():
                flash(f'Lab documents directory does not exist: {labdocs_dir}', 'error')
                return redirect(url_for('data_config.configure', project_id=project_id))

        # Check for existing RAG index
        existing_rag_mode = request.form.get('existing_rag', 'no')
        existing_rag_path = None
        if existing_rag_mode == 'yes':
            existing_rag_path = request.form.get('existing_rag_path', '').strip()
            if existing_rag_path and not Path(existing_rag_path).exists():
                flash(f'Existing RAG index path does not exist: {existing_rag_path}', 'error')
                return redirect(url_for('data_config.configure', project_id=project_id))

            # Validate RAG index files
            if existing_rag_path:
                rag_path = Path(existing_rag_path)
                required_files = ['faiss.index', 'documents_metadata.json', 'chunks.json']
                missing_files = [f for f in required_files if not (rag_path / f).exists()]
                if missing_files:
                    flash(f'RAG index is missing required files: {", ".join(missing_files)}', 'error')
                    return redirect(url_for('data_config.configure', project_id=project_id))

        # Validate that we have at least papers/lab docs for RAG OR existing RAG index
        if not papers_dir and not labdocs_dir and not existing_rag_path:
            flash('Warning: No RAG data provided. RAG features will be unavailable.', 'warning')

        # Handle existing training data
        if existing_data_path:
            # Validate path exists
            if not Path(existing_data_path).exists():
                flash(f'Training data path does not exist: {existing_data_path}', 'error')
                return redirect(url_for('data_config.configure', project_id=project_id))

            # Copy file to project directory
            import shutil
            data_gen_output_dir = project_dir / 'data-generation'
            data_gen_output_dir.mkdir(parents=True, exist_ok=True)
            dest_file = data_gen_output_dir / 'combined_instruct_train.jsonl'
            shutil.copy2(existing_data_path, dest_file)
            flash(f'Copied existing training data from {existing_data_path}', 'success')

        elif existing_data_file and existing_data_file.filename:
            # Save uploaded file
            data_gen_output_dir = project_dir / 'data-generation'
            data_gen_output_dir.mkdir(parents=True, exist_ok=True)
            dest_file = data_gen_output_dir / 'combined_instruct_train.jsonl'
            existing_data_file.save(str(dest_file))
            flash(f'Uploaded training data file: {existing_data_file.filename}', 'success')

        # Register code repositories in database
        for repo_path in code_repos:
            # Determine source type
            if repo_path.startswith('http://') or repo_path.startswith('https://'):
                source_type = 'github'
            else:
                source_type = 'local'
                # Validate local path exists
                if not Path(repo_path).exists():
                    flash(f'Local repository path does not exist: {repo_path}', 'error')
                    return redirect(url_for('data_config.configure', project_id=project_id))

            # Register in database
            code_repo = CodeRepo(
                project_id=project_id,
                source_type=source_type,
                source_path=repo_path,
                local_path=repo_path  # For local repos; GitHub repos will be cloned by labgpt_cli.py
            )
            db.session.add(code_repo)

        db.session.commit()

        # Track what was configured
        configured_items = []

        # Handle RAG: either use existing or start pipeline
        rag_job = None
        if existing_rag_path:
            # Use existing RAG index - create symlink
            import os
            rag_index_dir = project_dir / 'rag_index'
            if rag_index_dir.exists():
                if rag_index_dir.is_symlink():
                    rag_index_dir.unlink()
                else:
                    import shutil
                    shutil.rmtree(rag_index_dir)

            os.symlink(existing_rag_path, rag_index_dir)
            project.rag_completed = True
            db.session.commit()
            configured_items.append('Existing RAG index linked')

        elif papers_dir or labdocs_dir:
            # Start RAG pipeline
            rag_index_dir = project_dir / 'rag_index'
            rag_index_dir.mkdir(parents=True, exist_ok=True)

            rag_task = run_rag_pipeline_task.delay(
                project_id=project_id,
                papers_paths=[papers_dir] if papers_dir else [],
                lab_docs_paths=[labdocs_dir] if labdocs_dir else [],
                rag_preset='research'
            )

            rag_job = StateManager.create_job(
                project_id=project_id,
                job_type='rag',
                celery_task_id=rag_task.id,
                log_file=str(project_dir / 'logs' / f'rag_{rag_task.id}.log')
            )
            configured_items.append(f'RAG pipeline started (Job {rag_job.id})')

        # Handle data generation: only start if we have code repos AND no existing data
        data_gen_job = None
        if code_repos and not (existing_data_path or existing_data_file):
            data_gen_output_dir = project_dir / 'data-generation'
            data_gen_output_dir.mkdir(parents=True, exist_ok=True)

            data_gen_task = run_data_generation_task.delay(
                project_id=project_id,
                code_repo_paths=code_repos,
                papers_dir=papers_dir,
                max_symbols=Config.DEFAULT_MAX_SYMBOLS,
                languages=Config.DEFAULT_LANGUAGES
            )

            data_gen_job = StateManager.create_job(
                project_id=project_id,
                job_type='data_generation',
                celery_task_id=data_gen_task.id,
                log_file=str(project_dir / 'logs' / f'data_gen_{data_gen_task.id}.log')
            )
            configured_items.append(f'Data generation pipeline started (Job {data_gen_job.id})')

        # Show summary of what was configured
        if configured_items:
            flash(f'Configuration complete: {", ".join(configured_items)}', 'success')
        else:
            flash('No pipelines started. You can configure them later from the pipeline status page.', 'info')

        return redirect(url_for('pipelines.status', project_id=project_id))

    except Exception as e:
        db.session.rollback()
        flash(f'Error starting pipelines: {str(e)}', 'error')
        return redirect(url_for('data_config.configure', project_id=project_id))
