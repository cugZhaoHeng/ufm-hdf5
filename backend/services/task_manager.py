"""In-memory background task manager for animation jobs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable
from uuid import uuid4


class TaskManager:
    """Track animation jobs and expose progress to the API layer."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, dict] = {}
        self._lock = Lock()

    def create_task(self, total_groups: int) -> str:
        """Create a pending task and return its id."""
        task_id = uuid4().hex
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "percent": 0,
                "current_group": None,
                "current_group_index": 0,
                "total_groups": total_groups,
                "message": "Waiting to start.",
                "result_url": None,
                "result_path": None,
                "error": None,
            }
        return task_id

    def submit(self, task_id: str, job: Callable) -> None:
        """Submit a job to the background executor."""
        self._executor.submit(self._run, task_id, job)

    def get_task(self, task_id: str) -> dict | None:
        """Return a copy of task state, or None when the task does not exist."""
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None

    def update_progress(
        self,
        task_id: str,
        *,
        percent: int,
        current_group: str | None = None,
        current_group_index: int | None = None,
        total_groups: int | None = None,
        message: str = "",
    ) -> None:
        """Update task progress fields."""
        with self._lock:
            task = self._tasks[task_id]
            task["status"] = "running"
            task["percent"] = max(0, min(99, int(percent)))
            if current_group is not None:
                task["current_group"] = current_group
            if current_group_index is not None:
                task["current_group_index"] = current_group_index
            if total_groups is not None:
                task["total_groups"] = total_groups
            if message:
                task["message"] = message

    def _run(self, task_id: str, job: Callable) -> None:
        with self._lock:
            self._tasks[task_id]["status"] = "running"
            self._tasks[task_id]["message"] = "Task started."

        def progress_callback(**kwargs):
            self.update_progress(task_id, **kwargs)

        try:
            result_path = job(progress_callback)
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = "completed"
                task["percent"] = 100
                task["message"] = "Animation completed."
                task["result_path"] = str(result_path)
                task["result_url"] = f"/api/tasks/{task_id}/result"
        except Exception as exc:
            with self._lock:
                task = self._tasks[task_id]
                task["status"] = "failed"
                task["percent"] = 100
                task["message"] = "Animation failed."
                task["error"] = str(exc)


TASK_MANAGER = TaskManager()
