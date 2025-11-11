"""
Flask extensions initialization.
Separating extensions from app.py avoids circular imports.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Initialize extensions (but don't bind to app yet)
db = SQLAlchemy()
migrate = Migrate()
