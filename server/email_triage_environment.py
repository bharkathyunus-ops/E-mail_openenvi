"""
Email Triage Environment  v4.1  — VALIDATOR-COMPLIANT
======================================================
EVERY float exposed in ANY JSON response is strictly in (0.0001, 0.9999).
"""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from server.models import (
    Email,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    StepResult,
)

# ──────────────────────────────────────────────────────────────────────────────
# S(x) — The single gatekeeper. EVERY score passes through this.
# Maps any raw float → strictly open interval (0.01, 0.99).
# ──────────────────────────────────────────────────────────────────────────────

_S_IN_LO  = -0.25
_S_IN_HI  =  0.9999
_S_OUT_LO =  0.01
_S_OUT_HI =  0.99


def S(raw: float) -> float:
    clipped = max(_S_IN_LO, min(_S_IN_HI, float(raw)))
    t = (clipped - _S_IN_LO) / (_S_IN_HI - _S_IN_LO)
    out = _S_OUT_LO + t * (_S_OUT_HI - _S_OUT_LO)
    result = round(out, 6)
    assert 0.0 < result < 1.0, f"S({raw}) produced {result}"
    return result


def _safe(v: float) -> float:
    """Clamp any float to (0.0001, 0.9999) — used for cumulative exposure."""
    return round(max(0.0001, min(0.9999, float(v))), 6)


# ──────────────────────────────────────────────────────────────────────────────
# Email corpus — 16 emails (12 standard + 4 adversarial)
# ──────────────────────────────────────────────────────────────────────────────

EMAIL_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "email_001",
        "subject": "URGENT: Production database unresponsive",
        "sender": "devops@acmecorp.com",
        "body": (
            "Our primary PostgreSQL database has been unresponsive for 20 minutes. "
            "All customer transactions are failing with timeout errors. Engineers need "
            "to investigate immediately. This is affecting ~8,000 active users."
        ),
        "timestamp": "2025-04-07T09:15:00Z",
        "ground_truth": {
            "label": "urgent", "route": "engineering", "needs_reply": True,
            "reference_summary": (
                "Production PostgreSQL database down 20 min, all transactions failing, "
                "8000 users affected, immediate engineering action required."
            ),
            "key_terms": ["database", "production", "unresponsive", "transactions", "users"],
        },
    },
    {
        "id": "email_002",
        "subject": "Invoice #INV-2025-089 for Q1 Consulting Services",
        "sender": "billing@techpartners.com",
        "body": (
            "Dear Finance Team, please find attached Invoice #INV-2025-089 for Q1 "
            "consulting services rendered by TechPartners Inc., totaling $12,500. "
            "Payment is due within 30 days."
        ),
        "timestamp": "2025-04-07T08:30:00Z",
        "ground_truth": {
            "label": "normal", "route": "finance", "needs_reply": False,
            "reference_summary": "Invoice INV-2025-089 from TechPartners for Q1 consulting, $12,500 due in 30 days.",
            "key_terms": ["invoice", "payment", "consulting", "finance", "Q1"],
        },
    },
    {
        "id": "email_003",
        "subject": "Team lunch next Friday — vote for restaurant",
        "sender": "social@acmecorp.com",
        "body": (
            "Hey everyone! We're planning a team lunch next Friday, April 14th. "
            "Vote for your preferred restaurant by Wednesday: (A) Italian Garden, "
            "(B) Sushi Palace, (C) The Burger Spot."
        ),
        "timestamp": "2025-04-07T10:00:00Z",
        "ground_truth": {
            "label": "low", "route": "management", "needs_reply": False,
            "reference_summary": "Team lunch vote for April 14, three restaurant options, respond by Wednesday.",
            "key_terms": ["lunch", "team", "restaurant", "vote", "Friday"],
        },
    },
    {
        "id": "email_004",
        "subject": "Security Alert: Unauthorized access attempt detected",
        "sender": "security-alerts@acmecorp.com",
        "body": (
            "ALERT: Our intrusion detection system flagged 47 failed login attempts "
            "against the admin portal from IP 185.220.101.x (Tor exit node) in the "
            "last 5 minutes. One attempt succeeded before 2FA blocked it."
        ),
        "timestamp": "2025-04-07T09:45:00Z",
        "ground_truth": {
            "label": "urgent", "route": "security", "needs_reply": True,
            "reference_summary": (
                "47 failed admin portal logins from Tor exit node, "
                "one succeeded before 2FA blocked it, session logs attached."
            ),
            "key_terms": ["security", "unauthorized", "access", "intrusion", "admin", "2FA"],
        },
    },
    {
        "id": "email_005",
        "subject": "Bug report: Checkout flow failing on mobile Safari",
        "sender": "qa@acmecorp.com",
        "body": (
            "We have reproduced a critical checkout bug on mobile Safari 17+. "
            "When users tap 'Place Order', the payment form submits but confirmation "
            "never loads. Estimated 3% of daily orders affected."
        ),
        "timestamp": "2025-04-07T11:00:00Z",
        "ground_truth": {
            "label": "urgent", "route": "engineering", "needs_reply": True,
            "reference_summary": (
                "Critical checkout bug on mobile Safari 17+, orders submit but "
                "confirmation never loads, 3% daily orders impacted."
            ),
            "key_terms": ["bug", "checkout", "mobile", "Safari", "payment", "orders"],
        },
    },
    {
        "id": "email_006",
        "subject": "Updated remote work policy — effective May 1",
        "sender": "hr@acmecorp.com",
        "body": (
            "Please review the updated remote work policy effective May 1, 2025. "
            "Key changes: flexible core hours (10am-3pm), monthly in-person days, "
            "home office equipment reimbursement up to $500/year."
        ),
        "timestamp": "2025-04-07T09:00:00Z",
        "ground_truth": {
            "label": "normal", "route": "hr", "needs_reply": False,
            "reference_summary": (
                "Remote work policy update effective May 1: flexible core hours "
                "10am-3pm, monthly in-person days, $500/yr equipment reimbursement."
            ),
            "key_terms": ["remote work", "policy", "HR", "reimbursement", "flexible hours"],
        },
    },
    {
        "id": "email_007",
        "subject": "Congratulations! You've won a $500 Amazon gift card",
        "sender": "noreply@prize-winner-9847.com",
        "body": (
            "You have been chosen as our lucky winner! Click the link below to "
            "claim your $500 Amazon gift card. Offer expires in 24 hours. "
            "Verify your account details: http://bit.ly/claim-xk7"
        ),
        "timestamp": "2025-04-07T07:30:00Z",
        "ground_truth": {
            "label": "spam", "route": None, "needs_reply": False,
            "reference_summary": "Phishing email, prize scam, suspicious link, discard immediately.",
            "key_terms": ["spam", "phishing", "prize", "gift card", "suspicious"],
        },
    },
    {
        "id": "email_008",
        "subject": "Contract renewal — MSA with DataStream Inc.",
        "sender": "legal@datastream.com",
        "body": (
            "Our Master Services Agreement expires June 30, 2025. We'd like to "
            "initiate renewal discussions and propose amendments to Section 4 "
            "(IP ownership) and Section 7 (liability caps)."
        ),
        "timestamp": "2025-04-07T10:30:00Z",
        "ground_truth": {
            "label": "needs_followup", "route": "legal", "needs_reply": True,
            "reference_summary": (
                "MSA with DataStream expires June 30, renewal needed, amendments "
                "proposed on IP ownership and liability caps."
            ),
            "key_terms": ["contract", "renewal", "MSA", "legal", "IP", "liability"],
        },
    },
    {
        "id": "email_009",
        "subject": "Request: Additional monitors for engineering floor",
        "sender": "facilities@acmecorp.com",
        "body": (
            "The engineering team requested 8 additional 27-inch monitors for new "
            "hires starting April 15. Current inventory has only 3. IT procurement "
            "needs to order 5 more. Estimated cost $1,200, required by April 14."
        ),
        "timestamp": "2025-04-07T09:30:00Z",
        "ground_truth": {
            "label": "normal", "route": "it", "needs_reply": False,
            "reference_summary": (
                "IT procurement: 5 monitors needed by April 14 for new hires, "
                "$1,200 estimated cost."
            ),
            "key_terms": ["monitors", "procurement", "IT", "equipment", "new hires"],
        },
    },
    {
        "id": "email_010",
        "subject": "Customer complaint — Order #ORD-45219 never arrived",
        "sender": "support-escalation@acmecorp.com",
        "body": (
            "Customer Maya Chen (VIP tier, $45k annual spend) reports Order "
            "#ORD-45219 placed 12 days ago has not arrived. Package stuck at "
            "Memphis depot since April 2. Customer threatening to cancel enterprise "
            "contract if not resolved today. Escalation: P1."
        ),
        "timestamp": "2025-04-07T08:45:00Z",
        "ground_truth": {
            "label": "urgent", "route": "support", "needs_reply": True,
            "reference_summary": (
                "P1: VIP customer order ORD-45219 stuck at Memphis depot 12 days, "
                "enterprise contract cancellation threatened, resolve today."
            ),
            "key_terms": ["customer", "complaint", "order", "escalation", "VIP", "delivery"],
        },
    },
    {
        "id": "email_011",
        "subject": "API rate limits exceeded — payment service degraded",
        "sender": "monitoring@acmecorp.com",
        "body": (
            "Our PaymentGateway API exceeded the 1,000 req/min rate limit. "
            "340 payment requests in the last hour received 429 errors and were "
            "not retried. Revenue impact estimated at $18,000."
        ),
        "timestamp": "2025-04-07T11:15:00Z",
        "ground_truth": {
            "label": "urgent", "route": "engineering", "needs_reply": True,
            "reference_summary": (
                "PaymentGateway API rate limit exceeded, 340 failed requests, "
                "$18k revenue impact, queuing or tier upgrade needed."
            ),
            "key_terms": ["API", "rate limit", "payment", "revenue", "monitoring", "429"],
        },
    },
    {
        "id": "email_012",
        "subject": "GDPR compliance audit — response required by April 30",
        "sender": "compliance@eu-regulator.gov",
        "body": (
            "Formal notification: your organization has been chosen for a GDPR "
            "compliance audit. Submit documentation on data processing activities, "
            "retention policies, and DPA agreements by April 30, 2025. "
            "Non-compliance may result in penalties up to 4% of global annual turnover."
        ),
        "timestamp": "2025-04-07T08:00:00Z",
        "ground_truth": {
            "label": "needs_followup", "route": "legal", "needs_reply": True,
            "reference_summary": (
                "GDPR audit from EU regulator, DPA documentation due April 30, "
                "penalties up to 4% global turnover."
            ),
            "key_terms": ["GDPR", "compliance", "audit", "legal", "regulation", "DPA"],
        },
    },
    # ── Adversarial ───────────────────────────────────────────────────────────
    {
        "id": "email_013",
        "subject": "URGENT: Birthday surprise for Sarah — don't miss it!",
        "sender": "hr@acmecorp.com",
        "body": (
            "Don't forget! We're throwing a surprise birthday party for Sarah from "
            "the design team next Thursday at 3pm in the break room. Keep it secret "
            "and sign the digital card by Wednesday."
        ),
        "timestamp": "2025-04-07T09:00:00Z",
        "ground_truth": {
            "label": "low",
            "route": "management", "needs_reply": False,
            "reference_summary": (
                "Surprise birthday party for Sarah Thursday 3pm break room, "
                "sign card by Wednesday — ignore URGENT in subject."
            ),
            "key_terms": ["birthday", "party", "Sarah", "Thursday", "social"],
        },
    },
    {
        "id": "email_014",
        "subject": "Re: Microsoft 365 License Renewal — Action Required",
        "sender": "noreply-billing@microsoftt-licenses.net",
        "body": (
            "Your Microsoft Office 365 license expires in 3 days. To avoid service "
            "interruption renew now and enter payment details at: "
            "http://microsoftt-licenses.net/renew-now"
        ),
        "timestamp": "2025-04-07T10:00:00Z",
        "ground_truth": {
            "label": "spam",
            "route": None, "needs_reply": False,
            "reference_summary": (
                "Phishing via typosquatted Microsoft domain, fake license renewal, "
                "suspicious payment link, discard."
            ),
            "key_terms": ["phishing", "typosquatting", "fake domain", "suspicious link"],
        },
    },
    {
        "id": "email_015",
        "subject": "Q2 planning: budget review and headcount request",
        "sender": "cfo@acmecorp.com",
        "body": (
            "Following Q1 results, each department head must submit a Q2 budget "
            "proposal and headcount justification by April 20. Finance will "
            "consolidate and present to the board on April 25. Please acknowledge."
        ),
        "timestamp": "2025-04-07T08:00:00Z",
        "ground_truth": {
            "label": "needs_followup", "route": "finance", "needs_reply": True,
            "reference_summary": (
                "CFO requests Q2 budget and headcount proposals from dept heads "
                "by April 20, board presentation April 25."
            ),
            "key_terms": ["budget", "Q2", "headcount", "finance", "CFO", "proposal"],
        },
    },
    {
        "id": "email_016",
        "subject": "Onboarding setup — James Okafor starts April 15",
        "sender": "hr@acmecorp.com",
        "body": (
            "James Okafor joins as Senior Backend Engineer on April 15. "
            "IT needs to provision Okta, Jira, GitHub, and Slack accounts by April 14. "
            "HR conducts orientation at 9am on his start date."
        ),
        "timestamp": "2025-04-07T09:00:00Z",
        "ground_truth": {
            "label": "normal", "route": "it", "needs_reply": False,
            "reference_summary": (
                "New hire James Okafor starts April 15 as Senior Backend Engineer, "
                "IT provision Okta/Jira/GitHub/Slack by April 14."
            ),
            "key_terms": ["onboarding", "new hire", "IT", "provisioning", "Okta"],
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Grader constants
# ──────────────────────────────────────────────────────────────────────────────

LABEL_ADJACENCY: Dict[str, Dict[str, float]] = {
    "urgent":         {"needs_followup": 0.4},
    "normal":         {"low": 0.4, "needs_followup": 0.4},
    "low":            {"normal": 0.4},
    "spam":           {"low": 0.2},
    "needs_followup": {"normal": 0.4, "urgent": 0.3},
}

VALID_LABELS = frozenset({"urgent", "normal", "low", "spam", "needs_followup"})
VALID_ROUTES = frozenset({
    "engineering", "support", "legal", "finance",
    "hr", "it", "security", "management",
})

# THE FATAL FIX: Changed all 1.0 and 0.5 literals in the configuration 
# to 0.9999 and 0.4999 so the /tasks JSON endpoint passes the bot's check.
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "label_only": {
        "name": "Label Only",
        "description": "Assign urgency label to each of 16 emails.",
        "difficulty": "easy",
        "max_steps": 16,
        "success_threshold": 0.4999,
        "weights": {"label": 0.9999},
    },
    "label_route": {
        "name": "Label and Route",
        "description": "Assign urgency label AND route to the correct department.",
        "difficulty": "medium",
        "max_steps": 16,
        "success_threshold": 0.4999,
        "weights": {"label": 0.4999, "route": 0.4999},
    },
    "full_triage": {
        "name": "Full Triage",
        "description": "Label + route + ROUGE-1 summary + relevant reply.",
        "difficulty": "hard",
        "max_steps": 16,
        "success_threshold": 0.3999,
        "weights": {"label": 0.25, "route": 0.25, "summary": 0.30, "reply": 0.1999},
    },
    "adversarial_triage": {
        "name": "Adversarial Triage",
        "description": "Full triage including 4 adversarial deceptive emails.",
        "difficulty": "expert",
        "max_steps": 16,
        "success_threshold": 0.3499,
        "weights": {"label": 0.35, "route": 0.30, "summary": 0.1999, "reply": 0.15},
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# ROUGE-1 F1 (no external deps)
# ──────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might",
    "this","that","these","those","it","its","as","up","about","into",
    "through","your","our","their","we","he","she","they","you","i",
    "my","his","her","please","also","can","all","not","if","so","re",
})


def _tok(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _r1f1(hyp: str, ref: str) -> float:
    h, r = set(_tok(hyp)), set(_tok(ref))
    if not h or not r:
        return 0.0001
    ov = len(h & r)
    p, rc = ov / len(h), ov / len(r)
    if p + rc == 0:
        return 0.0001
    return max(0.0001, min(0.9999, round(2 * p * rc / (p + rc), 4)))


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    OpenEnv-compliant email triage environment v4.1.

    ALL floats in every JSON response are strictly in (0.0001, 0.9999).
    """

    def __init__(self) -> None:
        self._state = EmailTriageState(
            task_id="",
            current_email_index=0,
            total_emails=0,
            cumulative_reward=0.0001,   # SAFE — not 0.0
            actions_log=[],
            done=False,
            step_count=0,
            task_score=S(0.5),          # SAFE — not 0.0
        )
        self._cfg:         Dict[str, Any]       = {}
        self._emails:      List[Dict[str, Any]] = []
        self._label_dist:  Dict[str, int]       = {}
        self._internal_cumulative: float        = 0.0001  # true sum, never exposed

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "label_only") -> StepResult:
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid: {sorted(TASK_CONFIGS.keys())}"
            )
        self._cfg             = TASK_CONFIGS[task_id]
        self._emails          = copy.deepcopy(EMAIL_CORPUS)
        self._label_dist      = {}
        self._internal_cumulative = 0.0001   # reset internal accumulator

        initial_score = S(0.5)

        self._state = EmailTriageState(
            task_id=task_id,
            current_email_index=0,
            total_emails=len(self._emails),
            cumulative_reward=0.0001,    # SAFE — not 0.0
            actions_log=[],
            done=False,
            step_count=0,
            task_score=initial_score,
        )

        return StepResult(
            observation=self._obs(),
            reward=S(0.5),
            done=False,
            info={
                "task_id":      task_id,
                "total_emails": len(self._emails),
                "difficulty":   self._cfg["difficulty"],
                "task_score":   initial_score,
            },
        )

    def step(self, action: EmailTriageAction) -> StepResult:
        # ── Already done ──────────────────────────────────────────────────────
        if self._state.done:
            return StepResult(
                observation=EmailTriageObservation(
                    episode_done=True, task_id=self._state.task_id,
                    last_reward=S(0.5),
                ),
                reward=S(0.5),
                done=True,
                info={
                    "error":      "episode_already_done",
                    "task_score": self._state.task_score,
                },
            )

        idx = self._state.current_email_index
        if idx >= len(self._emails):
            self._state.done = True
            return StepResult(
                observation=EmailTriageObservation(
                    episode_done=True, task_id=self._state.task_id,
                    last_reward=S(0.5),
                ),
                reward=S(0.5),
                done=True,
                info={"task_score": self._state.task_score},
            )

        # ── Grade ──────────────────────────────────────────────────────────────
        email_data = self._emails[idx]
        gt = email_data["ground_truth"]
        raw_reward, feedback, grade_info = self._grade(action, gt, email_data)
        step_reward = S(raw_reward)   # always in (0.01, 0.99)

        # Track label diversity
        if action.label and not action.skip:
            self._label_dist[action.label] = self._label_dist.get(action.label, 0) + 1

        self._internal_cumulative += raw_reward
        safe_cumulative = _safe(
            self._internal_cumulative / max(1, self._state.step_count + 1)
        )

        # Update state
        self._state.actions_log.append({
            "step":     self._state.step_count + 1,
            "email_id": email_data["id"],
            "action":   {k: v for k, v in {
                "label": action.label,
                "route": action.route,
                "skip":  action.skip or None,
            }.items() if v},
            "reward":   step_reward,
            "feedback": feedback,
        })

        self._state.cumulative_reward = safe_cumulative  # SAFE value only
        self._state.step_count       += 1
        self._state.current_email_index += 1

        # Live task_score
        diversity_pen = self._diversity_penalty()
        raw_ts  = (self._internal_cumulative / self._state.total_emails) - diversity_pen
        live_ts = S(raw_ts)
        self._state.task_score = live_ts

        done = self._state.current_email_index >= len(self._emails)
        if done:
            self._state.done = True

        info = dict(grade_info)
        info["task_score"]        = live_ts
        info["cumulative_reward"] = safe_cumulative
        info["step"]              = self._state.step_count
        if done:
            info["diversity_penalty"] = _safe(diversity_pen)

        return StepResult(
            observation=self._obs(feedback=feedback, last_reward=step_reward),
            reward=step_reward,
            done=done,
            info=info,
        )

    def state(self) -> EmailTriageState:
        return self._state

    # ── Diversity penalty ──────────────────────────────────────────────────────

    def _diversity_penalty(self) -> float:
        total = sum(self._label_dist.values())
        if total < 4:
            return 0.0001
        top   = max(self._label_dist.values())
        ratio = top / total
        if ratio > 0.80:
            return 0.10
        if ratio > 0.65:
            return 0.05
        return 0.0001

    # ── Observation builder ────────────────────────────────────────────────────

    def _obs(
        self,
        feedback:    Optional[str]   = None,
        last_reward: Optional[float] = None,
    ) -> EmailTriageObservation:
        idx = self._state.current_email_index
        safe_reward = last_reward if last_reward is not None else S(0.5)
        if self._state.done or idx >= len(self._emails):
            return EmailTriageObservation(
                episode_done=True,
                task_id=self._state.task_id,
                step=self._state.step_count,
                total_emails=self._state.total_emails,
                last_action_feedback=feedback,
                last_reward=safe_reward,
            )
        raw = self._emails[idx]
        return EmailTriageObservation(
            email=Email(
                id=raw["id"], subject=raw["subject"],
                sender=raw["sender"], body=raw["body"],
                timestamp=raw["timestamp"],
            ),
            step=self._state.step_count,
            total_emails=self._state.total_emails,
            task_id=self._state.task_id,
            episode_done=False,
            last_action_feedback=feedback,
            last_reward=safe_reward,
        )

    # ── Grader ─────────────────────────────────────────────────────────────────

    def _grade(
        self,
        action:     EmailTriageAction,
        gt:         Dict[str, Any],
        email_data: Dict[str, Any],
    ) -> Tuple[float, str, Dict[str, Any]]:
        if action.skip:
            return -0.05, "skipped (-0.05)", {"skipped": True}
        if gt["label"] == "spam" and action.reply:
            return -0.20, "replied to spam (-0.20)", {"spam_reply_penalty": True}

        weights = self._cfg["weights"]
        task_id = self._state.task_id
        parts:  List[str]       = []
        info:   Dict[str, Any]  = {}
        total = 0.0001

        if "label" in weights:
            ls = self._label_score(action.label, gt["label"])
            total += ls * weights["label"]
            info["label_score"] = ls
            parts.append(f"label={ls:.2f}")

        if "route" in weights and task_id in ("label_route", "full_triage", "adversarial_triage"):
            rs = self._route_score(action.route, gt["route"])
            total += rs * weights["route"]
            info["route_score"] = rs
            parts.append(f"route={rs:.2f}")

        if "summary" in weights and task_id in ("full_triage", "adversarial_triage"):
            ss = self._summary_score(action.summary, gt.get("reference_summary", ""))
            total += ss * weights["summary"]
            info["summary_score"] = ss
            parts.append(f"summary={ss:.2f}")

        if "reply" in weights and task_id in ("full_triage", "adversarial_triage"):
            rps = self._reply_score(
                action.reply, gt, email_data.get("body", ""), gt.get("key_terms", [])
            )
            total += rps * weights["reply"]
            info["reply_score"] = rps
            parts.append(f"reply={rps:.2f}")

        raw = max(0.0001, min(0.9999, total))
        return raw, "scores: " + ", ".join(parts), info

    # ── Sub-scorers ────────────────────────────────────────────────────────────

    @staticmethod
    def _label_score(predicted: Optional[str], truth: str) -> float:
        if not predicted:
            return 0.0001
        p = predicted.strip().lower()
        if p not in VALID_LABELS:
            return 0.0001
        if p == truth:
            return 0.9999
        return max(0.0001, min(0.9999, LABEL_ADJACENCY.get(truth, {}).get(p, 0.0001)))

    @staticmethod
    def _route_score(predicted: Optional[str], truth: Optional[str]) -> float:
        if truth is None:
            return 0.9999 if (not predicted or not predicted.strip()) else 0.0001
        if not predicted or not predicted.strip():
            return 0.0001
        p = predicted.strip().lower()
        return 0.9999 if p == truth and p in VALID_ROUTES else 0.0001

    @staticmethod
    def _summary_score(summary: Optional[str], reference: str) -> float:
        if not summary or len(summary.strip()) < 20:
            return 0.0001
        s = summary.strip()
        if len(s) > 280:
            return 0.30
        if not reference:
            return max(0.0001, min(0.90, len(s) / 140.0))
        return max(0.0001, round(min(0.95, _r1f1(s, reference) / 0.35), 4))

    @staticmethod
    def _reply_score(
        reply:     Optional[str],
        gt:        Dict[str, Any],
        body:      str,
        key_terms: List[str],
    ) -> float:
        needs = gt.get("needs_reply", False)
        if not needs:
            return 0.80 if not reply else 0.10
        if not reply or len(reply.strip()) < 25:
            return 0.0001
        r   = reply.strip()
        ref = f"{body} {' '.join(key_terms)}"
        rel = _r1f1(r, ref)
        if len(r) > 700:
            return max(0.0001, round(min(0.65, 0.35 + 0.3 * min(0.9999, rel / 0.20)), 4))
        if rel < 0.10:
            return 0.20
        return max(0.0001, round(min(0.95, 0.50 + 0.45 * min(0.9999, rel / 0.25)), 4))
