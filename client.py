# client.py
"""
Email Triage Environment — Python Client
=========================================
Async HTTP client wrapping the REST API.
All three spec methods (reset, step, state) use POST.
"""

import asyncio
import subprocess
import time
from typing import Any, Dict, Optional

import httpx

from server.models import (
    EmailTriageAction,
    EmailTriageState,
    StepResult,
)


class EmailTriageClient:
    """
    Async client for the Email Triage OpenEnv environment.

    Usage:
        client = EmailTriageClient("http://localhost:7860")
        result = await client.reset("label_only")
        while not result.done:
            action = EmailTriageAction(label="urgent", route="engineering")
            result = await client.step(action)
        state = await client.state()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── OpenEnv interface methods ──────────────────────────────────────────────

    async def reset(self, task_id: str = "label_only") -> StepResult:
        """POST /reset — start a new episode."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id},
            )
            resp.raise_for_status()
            return StepResult(**resp.json())

    async def step(self, action: EmailTriageAction) -> StepResult:
        """POST /step — submit an action for the current email."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/step",
                json=action.model_dump(exclude_none=True),
            )
            resp.raise_for_status()
            return StepResult(**resp.json())

    async def state(self) -> EmailTriageState:
        """POST /state — query current environment state."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/state",
                json={},
            )
            resp.raise_for_status()
            return EmailTriageState(**resp.json())

    # ── Convenience methods ────────────────────────────────────────────────────

    async def health(self) -> Dict[str, Any]:
        """GET /health — check server liveness."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def tasks(self) -> Dict[str, Any]:
        """GET /tasks — list available tasks."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/tasks")
            resp.raise_for_status()
            return resp.json()

    async def score(self) -> Dict[str, Any]:
        """GET /score — current episode score."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/score")
            resp.raise_for_status()
            return resp.json()

    @classmethod
    async def from_docker_image(
        cls,
        image_name: Optional[str] = None,
        port: int = 7860,
        wait_seconds: int = 8,
    ) -> "EmailTriageClient":
        """
        Start the environment container and return a connected client.

        Args:
            image_name:    Docker image name (e.g. "email-triage-env").
            port:          Host port to bind (default 7860).
            wait_seconds:  Seconds to wait for container startup.
        """
        if image_name:
            subprocess.Popen(
                [
                    "docker", "run", "-d",
                    "-p", f"{port}:7860",
                    "--name", "email-triage-env-container",
                    "--rm",
                    image_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Wait for container to be ready
            await asyncio.sleep(wait_seconds)

        instance = cls(base_url=f"http://localhost:{port}")

        # Verify connectivity
        for attempt in range(5):
            try:
                await instance.health()
                return instance
            except Exception:
                if attempt < 4:
                    await asyncio.sleep(2)
                else:
                    raise RuntimeError(
                        f"Could not connect to environment at port {port} "
                        "after 5 attempts."
                    )

        return instance  # unreachable but satisfies type checker