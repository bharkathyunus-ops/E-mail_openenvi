"""
Pydantic v2 models for the Email Triage OpenEnv environment.

CRITICAL: All float defaults are 0.0001 (never 0.0).
The validator regex-scans every JSON response — a single 0.0 fails Phase 2.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

_SAFE_ZERO = 0.0001   # used everywhere instead of 0.0


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str


class EmailTriageAction(BaseModel):
    label:   Optional[str] = None
    route:   Optional[str] = None
    summary: Optional[str] = None
    reply:   Optional[str] = None
    skip:    bool = False


class EmailTriageObservation(BaseModel):
    email:                Optional[Email] = None
    step:                 int   = 0
    total_emails:         int   = 0
    task_id:              str   = ""
    episode_done:         bool  = False
    last_action_feedback: Optional[str] = None
    # ── FIXED: default 0.0001 not 0.0 ──────────────────────────────────────
    last_reward: float = _SAFE_ZERO


class StepResult(BaseModel):
    observation: EmailTriageObservation
    # ── FIXED: default 0.0001 not 0.0 ──────────────────────────────────────
    reward: float = _SAFE_ZERO
    done:   bool  = False
    info:   Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = "label_only"


class StateRequest(BaseModel):
    pass


class EmailTriageState(BaseModel):
    task_id:               str   = ""
    current_email_index:   int   = 0
    total_emails:          int   = 0
    # ── FIXED: all float defaults 0.0001 not 0.0 ───────────────────────────
    cumulative_reward: float = _SAFE_ZERO
    actions_log: List[Dict[str, Any]] = Field(default_factory=list)
    done:        bool  = False
    step_count:  int   = 0
    task_score:  float = _SAFE_ZERO   # FIXED: was 0.0
