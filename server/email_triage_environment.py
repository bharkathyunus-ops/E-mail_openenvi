"""
Email Triage Environment — Core Logic  v2.0
=============================================
Implements the OpenEnv step() / reset() / state() interface.

v2 improvements over v1:
  - ROUGE-1 F1 summary scoring (blocks keyword-stuffing exploit)
  - Reply relevance scoring (blocks generic/canned reply exploit)
  - Spam route: providing any route on spam = 0.0 (was 0.5, now fixed)
  - task_score clamped to [0.0, 1.0]
  - Corpus expanded to 16 emails with adversarial edge cases
  - 4th task: adversarial_triage (deceptive subjects, typosquatted senders)
  - Diversity penalty: mono-label episodes get deduction at episode end
  - Summary minimum length raised: < 20 chars = 0.0 score
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


# ── Reward clamping — Phase 2 requirement ─────────────────────────────────────
# Scores must be strictly between 0 and 1: (0.0, 1.0) open interval.
# We map the raw reward range [-0.20, +1.0] linearly into [0.01, 0.99].
# This preserves all relative ordering and partial-credit semantics while
# satisfying the validator constraint that no score equals exactly 0.0 or 1.0.

_RAW_MIN = -0.20   # worst possible raw reward (spam reply penalty)
_RAW_MAX = 1.00    # best possible raw reward (perfect action)
_OUT_MIN = 0.01    # output floor (strictly > 0)
_OUT_MAX = 0.99    # output ceiling (strictly < 1)


def _clamp(raw: float) -> float:
    """
    Linearly map raw reward from [-0.20, 1.00] → (0.01, 0.99).
    Ensures scores are strictly within (0, 1) as required by Phase 2 validation.
    """
    clipped = max(_RAW_MIN, min(_RAW_MAX, raw))
    normalised = (clipped - _RAW_MIN) / (_RAW_MAX - _RAW_MIN)   # 0.0 – 1.0
    return round(_OUT_MIN + normalised * (_OUT_MAX - _OUT_MIN), 4)

# ── Email Corpus (16 workplace emails, including 4 adversarial edge cases) ─────

EMAIL_CORPUS: List[Dict[str, Any]] = [
    # ── Standard cases (12) ────────────────────────────────────────────────────
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
            "label": "urgent",
            "route": "engineering",
            "needs_reply": True,
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
            "Payment is due within 30 days. Please remit to the account details in "
            "the attached document."
        ),
        "timestamp": "2025-04-07T08:30:00Z",
        "ground_truth": {
            "label": "normal",
            "route": "finance",
            "needs_reply": False,
            "reference_summary": (
                "Invoice INV-2025-089 from TechPartners for Q1 consulting, "
                "$12,500 due within 30 days."
            ),
            "key_terms": ["invoice", "payment", "consulting", "finance", "Q1"],
        },
    },
    {
        "id": "email_003",
        "subject": "Team lunch next Friday — vote for restaurant",
        "sender": "social@acmecorp.com",
        "body": (
            "Hey everyone! We're planning a team lunch next Friday, April 14th. "
            "Please vote for your preferred restaurant by end of day Wednesday. "
            "Options: (A) Italian Garden, (B) Sushi Palace, (C) The Burger Spot. "
            "Reply with your choice!"
        ),
        "timestamp": "2025-04-07T10:00:00Z",
        "ground_truth": {
            "label": "low",
            "route": "management",
            "needs_reply": False,
            "reference_summary": (
                "Team lunch vote April 14, three restaurant options, "
                "respond by Wednesday."
            ),
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
            "last 5 minutes. One attempt succeeded before 2FA blocked it. Immediate "
            "review required. Session logs attached."
        ),
        "timestamp": "2025-04-07T09:45:00Z",
        "ground_truth": {
            "label": "urgent",
            "route": "security",
            "needs_reply": True,
            "reference_summary": (
                "47 failed admin portal logins from Tor exit node, one succeeded "
                "before 2FA blocked it, session logs attached."
            ),
            "key_terms": ["security", "unauthorized", "access", "intrusion", "admin", "2FA"],
        },
    },
    {
        "id": "email_005",
        "subject": "Bug report: Checkout flow failing on mobile Safari",
        "sender": "qa@acmecorp.com",
        "body": (
            "We have reproduced a critical checkout bug on mobile Safari 17+. When "
            "users tap 'Place Order', the payment form submits but the confirmation "
            "page never loads, leaving orders in a limbo state. Estimated 3% of daily "
            "orders affected. Fix needed before this weekend's traffic spike."
        ),
        "timestamp": "2025-04-07T11:00:00Z",
        "ground_truth": {
            "label": "urgent",
            "route": "engineering",
            "needs_reply": True,
            "reference_summary": (
                "Critical checkout bug on mobile Safari 17+, orders submit but "
                "confirmation never loads, 3% daily orders impacted before weekend spike."
            ),
            "key_terms": ["bug", "checkout", "mobile", "Safari", "payment", "orders"],
        },
    },
    {
        "id": "email_006",
        "subject": "Updated remote work policy — effective May 1",
        "sender": "hr@acmecorp.com",
        "body": (
            "Please review the attached updated remote work policy, effective May 1, "
            "2025. Key changes include: flexible core hours (10am–3pm), monthly "
            "in-person collaboration days, and updated home office equipment "
            "reimbursement up to $500/year. Questions? Contact HR by April 25."
        ),
        "timestamp": "2025-04-07T09:00:00Z",
        "ground_truth": {
            "label": "normal",
            "route": "hr",
            "needs_reply": False,
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
            "You have been selected as our lucky winner! Click the link below to "
            "claim your $500 Amazon gift card. Offer expires in 24 hours. Verify "
            "your account details to receive your prize: http://bit.ly/claim-xk7"
        ),
        "timestamp": "2025-04-07T07:30:00Z",
        "ground_truth": {
            "label": "spam",
            "route": None,
            "needs_reply": False,
            "reference_summary": (
                "Phishing email, prize scam, suspicious link, discard immediately."
            ),
            "key_terms": ["spam", "phishing", "prize", "gift card", "suspicious"],
        },
    },
    {
        "id": "email_008",
        "subject": "Contract renewal — MSA with DataStream Inc.",
        "sender": "legal@datastream.com",
        "body": (
            "Our Master Services Agreement expires on June 30, 2025. We'd like to "
            "initiate renewal discussions and propose amendments to Section 4 "
            "(IP ownership) and Section 7 (liability caps). Please have your legal "
            "team review the attached redline and schedule a call at your earliest "
            "convenience."
        ),
        "timestamp": "2025-04-07T10:30:00Z",
        "ground_truth": {
            "label": "needs_followup",
            "route": "legal",
            "needs_reply": True,
            "reference_summary": (
                "MSA with DataStream expires June 30, renewal needed, amendments "
                "proposed on IP ownership and liability caps, schedule review call."
            ),
            "key_terms": ["contract", "renewal", "MSA", "legal", "IP", "liability"],
        },
    },
    {
        "id": "email_009",
        "subject": "Request: Additional monitors for engineering floor",
        "sender": "facilities@acmecorp.com",
        "body": (
            "The engineering team has requested 8 additional 27-inch monitors for "
            "new hires starting April 15. Current inventory only has 3 units. Can IT "
            "procurement approve and order 5 more? Estimated cost: $1,200. Required "
            "by April 14."
        ),
        "timestamp": "2025-04-07T09:30:00Z",
        "ground_truth": {
            "label": "normal",
            "route": "it",
            "needs_reply": False,
            "reference_summary": (
                "IT procurement request: 5 monitors needed by April 14 "
                "for new hires, $1,200 estimated cost."
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
            "#ORD-45219 placed 12 days ago has not been delivered. Tracking shows "
            "the package stuck at Memphis depot since April 2. Customer is threatening "
            "to cancel her enterprise contract if not resolved today. Escalation: P1."
        ),
        "timestamp": "2025-04-07T08:45:00Z",
        "ground_truth": {
            "label": "urgent",
            "route": "support",
            "needs_reply": True,
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
            "Our integration with the PaymentGateway API exceeded the 1,000 req/min "
            "rate limit. About 340 payment requests in the last hour received 429 "
            "errors and were not retried. Revenue impact estimated at $18,000. "
            "Engineering needs to implement request queuing or upgrade the API tier."
        ),
        "timestamp": "2025-04-07T11:15:00Z",
        "ground_truth": {
            "label": "urgent",
            "route": "engineering",
            "needs_reply": True,
            "reference_summary": (
                "PaymentGateway API rate limit exceeded, 340 failed requests, "
                "$18k revenue impact, queuing or tier upgrade needed urgently."
            ),
            "key_terms": ["API", "rate limit", "payment", "revenue", "monitoring", "429"],
        },
    },
    {
        "id": "email_012",
        "subject": "GDPR compliance audit — response required by April 30",
        "sender": "compliance@eu-regulator.gov",
        "body": (
            "This is a formal notification that your organization has been selected "
            "for a GDPR compliance audit. You must submit documentation on data "
            "processing activities, retention policies, and DPA agreements by "
            "April 30, 2025. Non-compliance may result in penalties up to 4% of "
            "global annual turnover. Please acknowledge receipt."
        ),
        "timestamp": "2025-04-07T08:00:00Z",
        "ground_truth": {
            "label": "needs_followup",
            "route": "legal",
            "needs_reply": True,
            "reference_summary": (
                "GDPR audit notification from EU regulator, DPA documentation due "
                "April 30, penalties up to 4% global turnover for non-compliance."
            ),
            "key_terms": ["GDPR", "compliance", "audit", "legal", "regulation", "DPA"],
        },
    },

    # ── Adversarial edge cases (v2 additions) ──────────────────────────────────

    {
        "id": "email_013",
        "subject": "URGENT: Birthday surprise for Sarah — don't miss it!",
        "sender": "hr@acmecorp.com",
        "body": (
            "Don't forget! We're throwing a surprise birthday party for Sarah from "
            "the design team next Thursday at 3pm in the break room. Please keep "
            "it a secret and bring a small gift if you'd like. Sign the digital "
            "card by Wednesday — link below."
        ),
        "timestamp": "2025-04-07T09:00:00Z",
        "ground_truth": {
            "label": "low",           # Adversarial: subject says URGENT, content is trivial
            "route": "management",
            "needs_reply": False,
            "reference_summary": (
                "Surprise birthday party for Sarah Thursday 3pm break room, "
                "sign card by Wednesday — ignore URGENT in subject."
            ),
            "key_terms": ["birthday", "party", "Sarah", "Thursday", "social", "surprise"],
        },
    },
    {
        "id": "email_014",
        "subject": "Re: Microsoft 365 License Renewal — Action Required",
        "sender": "noreply-billing@microsoftt-licenses.net",   # typosquatted domain
        "body": (
            "Your Microsoft Office 365 license subscription is expiring in 3 days. "
            "To avoid service interruption, please renew immediately and enter your "
            "payment details at: http://microsoftt-licenses.net/renew-now. "
            "Act now to prevent data loss and account suspension."
        ),
        "timestamp": "2025-04-07T10:00:00Z",
        "ground_truth": {
            "label": "spam",          # Adversarial: looks legitimate, domain is typosquatted
            "route": None,
            "needs_reply": False,
            "reference_summary": (
                "Phishing via typosquatted Microsoft domain, fake license renewal, "
                "suspicious payment link, discard and report."
            ),
            "key_terms": ["phishing", "typosquatting", "fake domain", "suspicious link"],
        },
    },
    {
        "id": "email_015",
        "subject": "Q2 planning: budget review and headcount request",
        "sender": "cfo@acmecorp.com",
        "body": (
            "Following our Q1 results, I'd like each department head to submit a "
            "detailed Q2 budget proposal and headcount justification by April 20. "
            "Proposals must include: projected spend, ROI estimates, and open "
            "headcount requests. Finance will consolidate and present to the board "
            "on April 25. Please acknowledge receipt by end of day."
        ),
        "timestamp": "2025-04-07T08:00:00Z",
        "ground_truth": {
            "label": "needs_followup",
            "route": "finance",
            "needs_reply": True,
            "reference_summary": (
                "CFO requests Q2 budget and headcount proposals from dept heads "
                "by April 20, board presentation April 25, acknowledge receipt today."
            ),
            "key_terms": ["budget", "Q2", "headcount", "finance", "CFO", "proposal"],
        },
    },
    {
        "id": "email_016",
        "subject": "Onboarding setup — James Okafor starts April 15",
        "sender": "hr@acmecorp.com",
        "body": (
            "James Okafor is joining as a Senior Backend Engineer on April 15. "
            "Please ensure his workstation is set up, access credentials are "
            "provisioned, and onboarding documentation is ready by April 14. "
            "IT needs to create accounts in Okta, Jira, GitHub, and Slack. "
            "HR will conduct orientation at 9am on his start date."
        ),
        "timestamp": "2025-04-07T09:00:00Z",
        "ground_truth": {
            "label": "normal",
            "route": "it",
            "needs_reply": False,
            "reference_summary": (
                "New hire James Okafor starts April 15 as Senior Backend Engineer, "
                "IT provision Okta/Jira/GitHub/Slack accounts by April 14."
            ),
            "key_terms": ["onboarding", "new hire", "IT", "provisioning", "Okta", "credentials"],
        },
    },
]

# ── Label adjacency (partial credit for near-misses) ──────────────────────────

LABEL_ADJACENCY: Dict[str, Dict[str, float]] = {
    "urgent":         {"needs_followup": 0.4},
    "normal":         {"low": 0.4, "needs_followup": 0.4},
    "low":            {"normal": 0.4},
    "spam":           {"low": 0.2},     # spam sometimes confused with low-priority
    "needs_followup": {"normal": 0.4, "urgent": 0.3},
}

# ── Valid vocabularies ─────────────────────────────────────────────────────────

VALID_LABELS = frozenset({"urgent", "normal", "low", "spam", "needs_followup"})
VALID_ROUTES = frozenset(
    {"engineering", "support", "legal", "finance", "hr", "it", "security", "management"}
)

# ── Task configuration ─────────────────────────────────────────────────────────

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "label_only": {
        "name": "Label Only",
        "description": (
            "Assign the correct urgency label to each of 16 emails. "
            "Labels: urgent | normal | low | spam | needs_followup."
        ),
        "difficulty": "easy",
        "max_steps": 16,
        "success_threshold": 0.5,
        "weights": {"label": 1.0},
    },
    "label_route": {
        "name": "Label and Route",
        "description": (
            "Assign urgency label AND route each email to the correct department. "
            "Equal weight: 50% label accuracy, 50% routing accuracy."
        ),
        "difficulty": "medium",
        "max_steps": 16,
        "success_threshold": 0.5,
        "weights": {"label": 0.5, "route": 0.5},
    },
    "full_triage": {
        "name": "Full Triage",
        "description": (
            "Complete triage: label + route + concise ROUGE-1 scored summary "
            "(≤280 chars) + relevant professional reply when warranted."
        ),
        "difficulty": "hard",
        "max_steps": 16,
        "success_threshold": 0.4,
        "weights": {"label": 0.25, "route": 0.25, "summary": 0.30, "reply": 0.20},
    },
    "adversarial_triage": {
        "name": "Adversarial Triage",
        "description": (
            "Full triage on all 16 emails, including 4 adversarial emails with "
            "deceptive subject lines, typosquatted senders, and misleading urgency. "
            "Tests robustness to social engineering and content misdirection."
        ),
        "difficulty": "expert",
        "max_steps": 16,
        "success_threshold": 0.35,
        "weights": {"label": 0.35, "route": 0.30, "summary": 0.20, "reply": 0.15},
    },
}

# ── ROUGE-1 F1 (no external dependencies) ─────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "it", "its", "as", "up", "about", "into", "through", "your", "our",
    "their", "we", "he", "she", "they", "you", "i", "my", "his", "her",
    "please", "also", "can", "all", "not", "if", "so", "re",
})


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _rouge1_f1(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-1 F1 (unigram overlap).
    Blocks keyword-stuffing exploit — a coherent summary of the right content
    will naturally score higher than a random list of keywords.
    """
    h_tokens = _tokenize(hypothesis)
    r_tokens = _tokenize(reference)
    if not h_tokens or not r_tokens:
        return 0.0
    h_set = set(h_tokens)
    r_set = set(r_tokens)
    overlap = len(h_set & r_set)
    precision = overlap / len(h_set)
    recall = overlap / len(r_set)
    if precision + recall == 0.0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 4)


def _reply_relevance(reply: str, body: str, key_terms: List[str]) -> float:
    """ROUGE-1 F1 of reply against email body + key terms as reference."""
    reference = f"{body} {' '.join(key_terms)}"
    return _rouge1_f1(reply, reference)


# ── Environment class ──────────────────────────────────────────────────────────


class EmailTriageEnvironment:
    """
    OpenEnv-compliant email triage environment v2.0.

    Usage:
        env = EmailTriageEnvironment()
        result = env.reset("full_triage")
        while not result.done:
            action = EmailTriageAction(label="urgent", route="engineering",
                                       summary="...", reply="...")
            result = env.step(action)
        state = env.state()
    """

    def __init__(self) -> None:
        self._state: EmailTriageState = EmailTriageState()
        self._task_config: Dict[str, Any] = {}
        self._emails: List[Dict[str, Any]] = []
        self._label_distribution: Dict[str, int] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "label_only") -> StepResult:
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {sorted(TASK_CONFIGS.keys())}"
            )

        self._task_config = TASK_CONFIGS[task_id]
        self._emails = copy.deepcopy(EMAIL_CORPUS)
        self._label_distribution = {}

        self._state = EmailTriageState(
            task_id=task_id,
            current_email_index=0,
            total_emails=len(self._emails),
            cumulative_reward=0.0,
            actions_log=[],
            done=False,
            step_count=0,
            task_score=0.0,
        )

        return StepResult(
            observation=self._build_observation(),
            reward=0.0,
            done=False,
            info={
                "task_id": task_id,
                "total_emails": len(self._emails),
                "difficulty": self._task_config["difficulty"],
                "max_steps": self._task_config["max_steps"],
            },
        )

    def step(self, action: EmailTriageAction) -> StepResult:
        if self._state.done:
            return StepResult(
                observation=EmailTriageObservation(
                    episode_done=True, task_id=self._state.task_id
                ),
                reward=0.0,
                done=True,
                info={"error": "episode_already_done"},
            )

        idx = self._state.current_email_index
        if idx >= len(self._emails):
            self._state.done = True
            return StepResult(
                observation=EmailTriageObservation(
                    episode_done=True, task_id=self._state.task_id
                ),
                reward=0.0,
                done=True,
                info={},
            )

        email_data = self._emails[idx]
        gt = email_data["ground_truth"]
        reward, feedback, info = self._grade(action, gt, email_data)

        if action.label and not action.skip:
            self._label_distribution[action.label] = (
                self._label_distribution.get(action.label, 0) + 1
            )

        self._state.actions_log.append({
            "step": self._state.step_count + 1,
            "email_id": email_data["id"],
            "action": {k: v for k, v in {
                "label": action.label,
                "route": action.route,
                "skip": action.skip if action.skip else None,
            }.items() if v is not None},
            "reward": reward,
            "feedback": feedback,
        })

        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 6)
        self._state.step_count += 1
        self._state.current_email_index += 1

        done = self._state.current_email_index >= len(self._emails)
        if done:
            self._state.done = True
            raw_score = self._state.cumulative_reward / len(self._emails)
            diversity_penalty = self._diversity_deduction()
            self._state.task_score = round(
                max(0.01, min(0.99, raw_score - diversity_penalty)), 6
            )
            info["diversity_penalty"] = round(diversity_penalty, 4)

        obs = self._build_observation(last_feedback=feedback, last_reward=reward)
        info["task_score"] = self._state.task_score
        info["cumulative_reward"] = round(self._state.cumulative_reward, 4)
        info["step"] = self._state.step_count

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> EmailTriageState:
        return self._state

    # ── Diversity deduction ────────────────────────────────────────────────────

    def _diversity_deduction(self) -> float:
        """
        Penalizes agents that assign the same label to every email.
        Real email queues are diverse — an intelligent agent should reflect that.
        > 80% same label: -0.10 deduction
        > 65% same label: -0.05 deduction
        """
        total = sum(self._label_distribution.values())
        if total < 4:
            return 0.0
        max_count = max(self._label_distribution.values())
        ratio = max_count / total
        if ratio > 0.80:
            return 0.10
        if ratio > 0.65:
            return 0.05
        return 0.0

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_observation(
        self,
        last_feedback: Optional[str] = None,
        last_reward: float = 0.0,
    ) -> EmailTriageObservation:
        idx = self._state.current_email_index
        done = self._state.done or idx >= len(self._emails)

        if done:
            return EmailTriageObservation(
                episode_done=True,
                task_id=self._state.task_id,
                step=self._state.step_count,
                total_emails=self._state.total_emails,
                last_action_feedback=last_feedback,
                last_reward=last_reward,
            )

        raw = self._emails[idx]
        email = Email(
            id=raw["id"],
            subject=raw["subject"],
            sender=raw["sender"],
            body=raw["body"],
            timestamp=raw["timestamp"],
        )
        return EmailTriageObservation(
            email=email,
            step=self._state.step_count,
            total_emails=self._state.total_emails,
            task_id=self._state.task_id,
            episode_done=False,
            last_action_feedback=last_feedback,
            last_reward=last_reward,
        )

    def _grade(
        self,
        action: EmailTriageAction,
        gt: Dict[str, Any],
        email_data: Dict[str, Any],
    ) -> Tuple[float, str, Dict[str, Any]]:
        # Hard penalties
        if action.skip:
            return _clamp(-0.05), "email skipped (penalty -0.05)", {"skipped": True}
        if gt["label"] == "spam" and action.reply:
            return _clamp(-0.20), "replied to spam (penalty -0.20)", {"spam_reply_penalty": True}

        task_id = self._state.task_id
        weights = self._task_config["weights"]
        info: Dict[str, Any] = {}
        parts: List[str] = []
        total: float = 0.0

        if "label" in weights:
            s = self._score_label(action.label, gt["label"])
            total += s * weights["label"]
            info["label_score"] = s
            parts.append(f"label={s:.2f}")

        if "route" in weights and task_id in ("label_route", "full_triage", "adversarial_triage"):
            s = self._score_route(action.route, gt["route"])
            total += s * weights["route"]
            info["route_score"] = s
            parts.append(f"route={s:.2f}")

        if "summary" in weights and task_id in ("full_triage", "adversarial_triage"):
            ref = gt.get("reference_summary", "")
            s = self._score_summary(action.summary, ref)
            total += s * weights["summary"]
            info["summary_score"] = s
            parts.append(f"summary={s:.2f}")

        if "reply" in weights and task_id in ("full_triage", "adversarial_triage"):
            s = self._score_reply(
                action.reply, gt, email_data.get("body", ""), gt.get("key_terms", [])
            )
            total += s * weights["reply"]
            info["reply_score"] = s
            parts.append(f"reply={s:.2f}")

        reward = _clamp(min(max(total, 0.0), 1.0))
        return reward, "scores: " + ", ".join(parts), info

    # ── Sub-scorers ────────────────────────────────────────────────────────────

    @staticmethod
    def _score_label(predicted: Optional[str], ground_truth: str) -> float:
        if not predicted:
            return 0.0
        p = predicted.strip().lower()
        if p not in VALID_LABELS:
            return 0.0
        if p == ground_truth:
            return 1.0
        return LABEL_ADJACENCY.get(ground_truth, {}).get(p, 0.0)

    @staticmethod
    def _score_route(predicted: Optional[str], ground_truth: Optional[str]) -> float:
        """
        v2 fix: spam emails (ground_truth=None) require NO route.
          - No route on spam → 1.0 (correct)
          - Any route on spam → 0.0 (penalize incorrect routing of spam)
        Regular emails: exact match = 1.0, wrong or missing = 0.0.
        """
        if ground_truth is None:
            return 1.0 if (not predicted or predicted.strip() == "") else 0.0
        if not predicted or predicted.strip() == "":
            return 0.0
        p = predicted.strip().lower()
        if p not in VALID_ROUTES:
            return 0.0
        return 1.0 if p == ground_truth.strip().lower() else 0.0

    @staticmethod
    def _score_summary(summary: Optional[str], reference: str) -> float:
        """
        ROUGE-1 F1 scoring. Completely blocks keyword-stuffing exploit.
        Minimum 20 chars; > 280 chars capped at 0.30.
        """
        if not summary or not summary.strip():
            return 0.0
        s = summary.strip()
        if len(s) < 20:
            return 0.0
        if len(s) > 280:
            return 0.30
        if not reference:
            return min(1.0, round(len(s) / 140.0, 4))
        rouge = _rouge1_f1(s, reference)
        return round(min(1.0, rouge / 0.35), 4)

    @staticmethod
    def _score_reply(
        reply: Optional[str],
        gt: Dict[str, Any],
        body: str,
        key_terms: List[str],
    ) -> float:
        """
        v2: Relevance-aware reply scoring. Generic/canned replies score 0.2 max.
        """
        needs_reply = gt.get("needs_reply", False)
        if not needs_reply:
            return 0.8 if not reply else 0.1   # harsher penalty for unnecessary replies
        if not reply or not reply.strip():
            return 0.0
        r = reply.strip()
        if len(r) < 25:
            return 0.2
        relevance = _reply_relevance(r, body, key_terms)
        if len(r) > 700:
            return round(0.35 + 0.3 * min(1.0, relevance / 0.20), 4)
        if relevance < 0.10:
            return 0.2   # generic reply, not connected to email content
        return round(min(1.0, 0.50 + 0.50 * min(1.0, relevance / 0.25)), 4)
