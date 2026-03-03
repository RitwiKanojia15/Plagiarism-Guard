"""Authentication helpers and decorators for session-based access control."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Dict, Optional

from flask import jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .database import session_scope
from .models import User


SESSION_KEY = "ps_user_id"


def normalize_email(email: str) -> str:
    """Normalize email into canonical lowercase format."""
    return (email or "").strip().lower()


def hash_password(password: str) -> str:
    """Create password hash."""
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against stored hash."""
    return check_password_hash(password_hash, password)


def get_session_user_id() -> Optional[int]:
    """Get authenticated user id from session."""
    raw = session.get(SESSION_KEY)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def current_user_payload() -> Optional[Dict[str, object]]:
    """Return current user payload if authenticated."""
    user_id = get_session_user_id()
    if not user_id:
        return None
    with session_scope() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        return {
            "id": int(user.id),
            "full_name": str(user.full_name),
            "email": str(user.email),
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }


def login_user(user: User) -> None:
    """Write authenticated user id to session."""
    session[SESSION_KEY] = int(user.id)
    session.permanent = True


def logout_user() -> None:
    """Clear authentication session."""
    session.pop(SESSION_KEY, None)


def login_required(route_func: Callable):
    """Protect route and redirect/login-fail depending on route type."""

    @wraps(route_func)
    def wrapper(*args, **kwargs):
        user = current_user_payload()
        if user:
            return route_func(*args, **kwargs)

        is_api = request.path.startswith("/api/")
        if is_api:
            return jsonify({"error": "Authentication required"}), 401
        return redirect(url_for("web.login_page"))

    return wrapper

