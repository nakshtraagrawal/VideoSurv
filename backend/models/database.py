"""SQLite database with SQLAlchemy"""
import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


DATABASE_URL = "sqlite:///./autosurveil.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    scene = Column(String)
    video_name = Column(String)
    frame_idx = Column(Integer)
    anomaly_score = Column(Float)
    model_used = Column(String)
    clip_path = Column(String)
    heatmap_path = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    reviewed = Column(Boolean, default=False)
    confirmed_anomaly = Column(Boolean, nullable=True)
    feedback_note = Column(String, nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)
    scene = Column(String)
    status = Column(String)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    final_loss = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)
    checkpoint_path = Column(String, nullable=True)


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)
    scene = Column(String)
    auc = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    fpr = Column(Float)
    threshold = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
