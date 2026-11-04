"""Bounded, persistent single-node job queue. Run one API process per queue directory."""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from uuid import UUID, uuid4

from fastapi import HTTPException

from config import Settings
from pipeline import PipelineBusyError

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    pass


class JobManager:
    def __init__(self, pipeline, settings: Settings):
        self.pipeline, self.settings = pipeline, settings
        self.root = Path(settings.job_directory).resolve()
        self.stop_event = Event()
        self.threads = []

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, options TEXT NOT NULL,
                result TEXT, error TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at)")
            # Inference is idempotent for a stored input. Re-run interrupted work after restart.
            db.execute("UPDATE jobs SET status='queued' WHERE status='running'")
            abandoned = db.execute("SELECT id FROM jobs WHERE status='uploading'").fetchall()
            for row in abandoned:
                self.audio_path(row["id"]).unlink(missing_ok=True)
            db.execute("DELETE FROM jobs WHERE status='uploading'")
        self.stop_event.clear()
        for number in range(self.settings.job_workers):
            worker = Thread(target=self.work, name=f"meeting-job-{number}", daemon=True)
            worker.start()
            self.threads.append(worker)

    def stop(self):
        self.stop_event.set()
        for worker in self.threads:
            worker.join(timeout=5)
        return not any(worker.is_alive() for worker in self.threads)

    def audio_path(self, job_id):
        try:
            if UUID(job_id).hex != job_id:
                raise ValueError
        except (ValueError, AttributeError) as exc:
            raise KeyError(job_id) from exc
        return self.root / f"{job_id}.audio"

    def submit(self, source, options):
        job_id, now = uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('uploading','queued','running')"
            ).fetchone()[0]
            if count >= self.settings.max_pending_jobs:
                raise QueueFullError("任务队列已满，请稍后重试")
            db.execute(
                "INSERT INTO jobs VALUES (?, 'uploading', ?, ?, ?, NULL, NULL)",
                (job_id, now, now, json.dumps(options)),
            )
        path = self.audio_path(job_id)
        try:
            total = 0
            with path.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "上传文件超过大小限制")
                    target.write(chunk)
            if not total:
                raise HTTPException(400, "上传文件为空")
            with self.connect() as db:
                db.execute(
                    "UPDATE jobs SET status='queued', updated_at=? WHERE id=?",
                    (time.time(), job_id),
                )
        except BaseException:
            path.unlink(missing_ok=True)
            with self.connect() as db:
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            raise
        return job_id

    def get(self, job_id):
        self.audio_path(job_id)  # Validate before constructing any filesystem path.
        with self.connect() as db:
            row = db.execute(
                "SELECT id,status,created_at,updated_at,error FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def result(self, job_id):
        self.audio_path(job_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        status = {key: row[key] for key in ("id", "status", "created_at", "updated_at", "error")}
        output = json.loads(row["result"]) if row["status"] == "completed" else None
        return status, output

    def delete(self, job_id):
        self.audio_path(job_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row[0] not in ("completed", "failed"):
                raise ValueError("只能删除已完成或失败的任务")
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        self.audio_path(job_id).unlink(missing_ok=True)

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id,options FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is not None:
                db.execute(
                    "UPDATE jobs SET status='running',updated_at=? WHERE id=?",
                    (time.time(), row["id"]),
                )
        return row

    def work(self):
        while not self.stop_event.is_set():
            try:
                row = self.claim()
                if row is None:
                    self.stop_event.wait(0.1)
                    continue
                job_id = row["id"]
                try:
                    result = self.pipeline.transcribe(
                        self.audio_path(job_id), **json.loads(row["options"])
                    )
                except PipelineBusyError:
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET status='queued' WHERE id=?", (job_id,))
                    self.stop_event.wait(0.2)
                    continue
                except Exception as exc:
                    logger.exception("Job %s failed", job_id)
                    with self.connect() as db:
                        db.execute(
                            "UPDATE jobs SET status='failed',error=?,updated_at=? WHERE id=?",
                            (str(exc)[:1000], time.time(), job_id),
                        )
                else:
                    with self.connect() as db:
                        db.execute(
                            """UPDATE jobs SET status='completed',result=?,updated_at=?
                                      WHERE id=?""",
                            (json.dumps(result, ensure_ascii=False), time.time(), job_id),
                        )
                self.audio_path(job_id).unlink(missing_ok=True)
            except Exception:
                logger.exception("Job worker failure")
                self.stop_event.wait(1)
