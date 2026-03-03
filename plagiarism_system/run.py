"""Application entry point for local execution."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from plagiarism_system.app import create_app
else:
    from .app import create_app

flask_app = create_app()

# Expose an ASGI-compatible app so `uvicorn plagiarism_system.run:app` works.
try:  # pragma: no cover - optional dependency path
    from starlette.middleware.wsgi import WSGIMiddleware

    app = WSGIMiddleware(flask_app)
except Exception:  # pragma: no cover - fallback for environments without starlette
    try:
        from asgiref.wsgi import WsgiToAsgi

        app = WsgiToAsgi(flask_app)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "No ASGI wrapper available. Install `starlette` or `asgiref` to run with uvicorn."
        ) from exc


if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=9000, debug=False)
