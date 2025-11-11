"""
Main Flask application factory for the unified LabGPT web interface.
"""

from flask import Flask, render_template, redirect, url_for
from unified_app.extensions import db, migrate


def create_app(config_name='default'):
    """
    Application factory pattern for creating Flask app instances.

    Args:
        config_name: Configuration to use ('development', 'production', or 'default')

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    from unified_app.config import config
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    from unified_app.routes.welcome import welcome_bp
    from unified_app.routes.data_config import data_config_bp
    from unified_app.routes.pipelines import pipelines_bp
    from unified_app.routes.training import training_bp
    from unified_app.routes.chat import chat_bp
    from unified_app.routes.grant import grant_bp
    from unified_app.routes.api import api_bp

    # Chat is now the main landing page
    app.register_blueprint(chat_bp, url_prefix='/chat')

    # Projects (previously welcome) moved to /projects
    app.register_blueprint(welcome_bp, url_prefix='/projects')

    # Other routes remain unchanged
    app.register_blueprint(data_config_bp, url_prefix='/data')
    app.register_blueprint(pipelines_bp, url_prefix='/pipelines')
    app.register_blueprint(training_bp, url_prefix='/training')
    app.register_blueprint(grant_bp, url_prefix='/grant')
    app.register_blueprint(api_bp, url_prefix='/api')

    # Root route redirects to chat
    @app.route('/')
    def index():
        """Redirect root to chat interface."""
        return redirect(url_for('chat.default_interface'))

    # Register error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500

    # Create database tables
    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    app = create_app('development')
    # use_reloader=False prevents double model loading during development
    # If you need auto-reload during active development, set to True
    app.run(host='0.0.0.0', port=5003, debug=True, use_reloader=False)
