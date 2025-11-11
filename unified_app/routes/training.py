"""
Training route - configure and run model training using labgpt_cli.py.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from pathlib import Path
from unified_app.extensions import db
from unified_app.models import Project, TrainingConfig
from unified_app.services.state_manager import StateManager
from unified_app.services.orchestration_service import OrchestrationService
from unified_app.tasks.pipeline_tasks import run_training_task
from unified_app.config import Config

training_bp = Blueprint('training', __name__)


@training_bp.route('/<int:project_id>/config')
def configure(project_id):
    """
    Training configuration page.

    Shows form with:
    - Data source selection (generated vs custom)
    - Model configuration
    - LoRA parameters
    - Training hyperparameters
    - Advanced options
    """
    project = Project.query.get_or_404(project_id)

    # Check if data generation is complete
    deps = StateManager.check_dependencies(project_id, 'training')
    if not deps['allowed']:
        flash(f'Cannot access training. Missing: {", ".join(deps["missing"])}', 'error')
        return redirect(url_for('pipelines.status', project_id=project_id))

    return render_template('training/config.html', project=project)


@training_bp.route('/<int:project_id>/start', methods=['POST'])
def start_training(project_id):
    """
    Start training with submitted configuration.

    Workflow:
    1. Extract form data
    2. Save TrainingConfig to database
    3. Start training Celery task (calls labgpt_cli.py)
    4. Redirect to training status page
    """
    project = Project.query.get_or_404(project_id)
    project_dir = Config.PROJECTS_BASE_DIR / project.project_dir

    try:
        # Extract form data
        data_source = request.form.get('data_source', 'generated')
        use_generated = (data_source == 'generated')

        # Get training data files
        if use_generated:
            orchestrator = OrchestrationService(project_dir)
            data_files = orchestrator.get_generated_data_files()
            train_file = data_files['train_file']
            val_file = data_files['val_file']

            if not train_file:
                flash('Generated training data not found. Please run data generation pipeline first.', 'error')
                return redirect(url_for('training.configure', project_id=project_id))
        else:
            train_file = request.form.get('train_file', '').strip()
            val_file = request.form.get('val_file', '').strip()

            if not train_file or not Path(train_file).exists():
                flash('Training file does not exist', 'error')
                return redirect(url_for('training.configure', project_id=project_id))

        # Extract configuration parameters
        training_config = TrainingConfig(
            project_id=project_id,
            use_generated_data=use_generated,
            custom_train_file=train_file if not use_generated else None,
            custom_val_file=val_file if not use_generated else None,
            base_model=request.form.get('base_model', Config.DEFAULT_BASE_MODEL),
            output_model_name=request.form.get('output_model_name', 'labgpt-finetuned'),
            use_lora=True,
            lora_rank=int(request.form.get('lora_rank', Config.DEFAULT_LORA_RANK)),
            lora_alpha=int(request.form.get('lora_alpha', Config.DEFAULT_LORA_ALPHA)),
            lora_dropout=float(request.form.get('lora_dropout', 0.05)),
            use_4bit='use_4bit' in request.form,
            max_seq_length=int(request.form.get('max_seq_length', Config.DEFAULT_MAX_SEQ_LENGTH)),
            batch_size=int(request.form.get('batch_size', Config.DEFAULT_BATCH_SIZE)),
            gradient_accumulation_steps=int(request.form.get('gradient_accumulation', Config.DEFAULT_GRADIENT_ACCUMULATION_STEPS)),
            num_train_epochs=int(request.form.get('num_epochs', Config.DEFAULT_NUM_TRAIN_EPOCHS)),
            learning_rate=float(request.form.get('learning_rate', Config.DEFAULT_LEARNING_RATE)),
            warmup_ratio=0.1,
            save_steps=500,
            logging_steps=10
        )

        # Save to database
        db.session.add(training_config)
        db.session.commit()

        # Prepare training configuration dict for Celery task
        config_dict = training_config.to_dict()

        # Start training task
        training_task = run_training_task.delay(
            project_id=project_id,
            training_config=config_dict
        )

        # Create job entry
        training_job = StateManager.create_job(
            project_id=project_id,
            job_type='training',
            celery_task_id=training_task.id,
            log_file=str(project_dir / 'logs' / f'training_{training_task.id}.log')
        )

        flash(f'Training started! Job ID: {training_job.id}', 'success')
        return redirect(url_for('training.status', project_id=project_id))

    except Exception as e:
        db.session.rollback()
        flash(f'Error starting training: {str(e)}', 'error')
        return redirect(url_for('training.configure', project_id=project_id))


@training_bp.route('/<int:project_id>/status')
def status(project_id):
    """
    Training status page with real-time metrics.

    Shows:
    - Training progress
    - Current loss, step, epoch
    - Loss history chart (future enhancement)
    - Streaming logs
    """
    project = Project.query.get_or_404(project_id)
    return render_template('training/status.html', project=project)
