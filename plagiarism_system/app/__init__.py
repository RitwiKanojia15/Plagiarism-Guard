"""Flask app factory for the plagiarism and AI detection system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from flask import Flask

from .database import init_database
from .routes import web_blueprint


def create_app(config: Optional[Dict[str, object]] = None) -> Flask:
    """Create and configure Flask application instance."""
    base_dir = Path(__file__).resolve().parents[1]
    app = Flask(
        __name__,
        static_folder=str(base_dir / "static"),
        template_folder=str(base_dir / "templates"),
    )
    app.config.update(
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        SECRET_KEY=os.getenv("PLAGIARISM_SYSTEM_SECRET_KEY", "plagiarism-system-dev-secret"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if config:
        app.config.update(config)

    init_database()
    app.register_blueprint(web_blueprint)
    return app
