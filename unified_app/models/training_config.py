"""
Training configuration model for storing training parameters.
"""

from datetime import datetime
from unified_app.extensions import db


class TrainingConfig(db.Model):
    """
    Stores training configuration for a project.
    """
    __tablename__ = 'training_configs'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False, unique=True)

    # Data configuration
    use_generated_data = db.Column(db.Boolean, default=True, nullable=False)
    custom_train_file = db.Column(db.String(1000))
    custom_val_file = db.Column(db.String(1000))

    # Model configuration
    base_model = db.Column(db.String(200), nullable=False)
    output_model_name = db.Column(db.String(200), nullable=False)

    # LoRA configuration
    use_lora = db.Column(db.Boolean, default=True, nullable=False)
    lora_rank = db.Column(db.Integer, default=16)
    lora_alpha = db.Column(db.Integer, default=32)
    lora_dropout = db.Column(db.Float, default=0.05)

    # Quantization
    use_4bit = db.Column(db.Boolean, default=True, nullable=False)

    # Training hyperparameters
    max_seq_length = db.Column(db.Integer, default=8192)
    batch_size = db.Column(db.Integer, default=2)
    gradient_accumulation_steps = db.Column(db.Integer, default=4)
    num_train_epochs = db.Column(db.Integer, default=3)
    learning_rate = db.Column(db.Float, default=2e-4)
    warmup_ratio = db.Column(db.Float, default=0.1)

    # Other settings
    save_steps = db.Column(db.Integer, default=500)
    logging_steps = db.Column(db.Integer, default=10)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    project = db.relationship('Project', back_populates='training_config')

    def __repr__(self):
        return f'<TrainingConfig {self.id} for Project {self.project_id}>'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'use_generated_data': self.use_generated_data,
            'custom_train_file': self.custom_train_file,
            'custom_val_file': self.custom_val_file,
            'base_model': self.base_model,
            'output_model_name': self.output_model_name,
            'use_lora': self.use_lora,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'use_4bit': self.use_4bit,
            'max_seq_length': self.max_seq_length,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'num_train_epochs': self.num_train_epochs,
            'learning_rate': self.learning_rate,
            'warmup_ratio': self.warmup_ratio,
            'save_steps': self.save_steps,
            'logging_steps': self.logging_steps
        }
