"""
Pydantic v2 models for the Email Triage OpenEnv environment.
All typed models used by the environment, server, and client.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Email(BaseModel):
    """A single workplace email to be triaged."""

    id: str
    subject: str
    sender: str
    body: str
    timestamp: str


class EmailTriageAction(BaseModel):
    """
    Action the agent submits for each email.

    For label_only:   set label only.
    For label_route:  set label + route.
    For full_triage:  set label + route + summary + reply (if needed).
    Setting skip=True defers the email with a -0.05 reward penalty.
    """

    label: Optional[str] = None
    # Allowed: urgent | normal | low | spam | needs_followup

    route: Optional[str] = None
    # Allowed: engineering | support | legal | finance | hr | it | security | management

    summary: Optional[str] = None
    # Concise summary, must be ≤ 280 characters

    reply: Optional[str] = None
    # Professional reply snippet — only when a response is actually warranted

    skip: bool = False
    # Defer this email; incurs a -0.05 reward penalty


class EmailTriageObservation(BaseModel):
    """Observation returned to the agent after each step."""

    email: Optional[Email] = None
    step: int = 0
    total_emails: int = 0
    task_id: str = ""
    episode_done: bool = False
    last_action_feedback: Optional[str] = None
    last_reward: float = 0.0


class StepResult(BaseModel):
    """Full result returned by /reset and /step endpoints."""

    observation: EmailTriageObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """
    Body for POST /reset.

    task_id is fully optional — defaults to label_only when body is empty,
    missing, or null (the OpenEnv validator sends POST /reset with no body
    at all during Phase 1 automated checks).
    """

    task_id: Optional[str] = "label_only"


class StateRequest(BaseModel):
    """Body for POST /state (empty by design)."""

    pass


class EmailTriageState(BaseModel):
    """Full internal state of the environment (returned by POST /state)."""

    task_id: str = ""
    current_email_index: int = 0
    total_emails: int = 0
    cumulative_reward: float = 0.0
    actions_log: List[Dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    step_count: int = 0
    task_score: float = 0.0
