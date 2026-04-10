from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str

class EmailTriageAction(BaseModel):
    label: Optional[str] = None
    route: Optional[str] = None
    summary: Optional[str] = None
    reply: Optional[str] = None
    skip: Optional[bool] = False

class EmailTriageObservation(BaseModel):
    email: Optional[Email] = None
    step: int = 0
    total_emails: int = 16
    task_id: str = "label_only"
    episode_done: bool = False
    last_action_feedback: Optional[str] = None
    last_reward: float = Field(default=0.5)  # SCRUBBED 0.0

class EmailTriageState(BaseModel):
    task_id: str = "label_only"
    current_email_index: int = 0
    total_emails: int = 16
    cumulative_reward: float = Field(default=0.5)  # SCRUBBED 0.0
    actions_log: List[Dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    step_count: int = 0
    task_score: float = Field(default=0.5)  # SCRUBBED 0.0

class StepResult(BaseModel):
    observation: EmailTriageObservation
    reward: float = Field(default=0.5)  # SCRUBBED 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

class ResetRequest(BaseModel):
    task_id: str = "label_only"

class StateRequest(BaseModel):
    pass

