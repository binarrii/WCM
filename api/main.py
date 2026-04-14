"""Flask application for face recognition service."""

from flask import Flask
from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.database import init_db

from .routes import api_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Enable CORS
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # Initialize database
    try:
        init_db()
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")

    return app


app = create_app()


def main():
    """Run the application."""
    app.run(
        host=settings.api_host,
        port=settings.api_port,
        debug=False,
    )


if __name__ == "__main__":
    main()
