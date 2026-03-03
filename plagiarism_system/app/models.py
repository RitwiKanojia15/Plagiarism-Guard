"""Database models for users and analysis history."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    """Application user model."""

    __tablename__ = "ps_users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(200), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    analyses = relationship("AnalysisRecord", back_populates="user")


class AnalysisRecord(Base):
    """Persisted analysis report metadata and payload."""

    __tablename__ = "ps_analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("ps_users.id"), nullable=False, index=True)
    file_name = Column(String(500), nullable=True)
    analysis_mode = Column(String(32), nullable=False, default="full")
    total_similarity = Column(Float, nullable=False, default=0.0)
    ai_likelihood = Column(Float, nullable=False, default=0.0)
    report_json = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")

