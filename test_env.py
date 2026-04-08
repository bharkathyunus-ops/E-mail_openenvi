# test_env.py
"""
Email Triage Environment — Test Suite
======================================
Run with:  pytest tests/ -v
"""

import pytest

from server.email_triage_environment import EMAIL_CORPUS, EmailTriageEnvironment
from server.models import EmailTriageAction


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def env() -> EmailTriageEnvironment:
    return EmailTriageEnvironment()


# ── Reset tests ────────────────────────────────────────────────────────────────


def test_reset_label_only(env):
    result = env.reset("label_only")
    assert result.observation.email is not None
    assert result.observation.total_emails == 12
    assert not result.done
    assert result.reward == 0.0


def test_reset_label_route(env):
    result = env.reset("label_route")
    assert result.observation.task_id == "label_route"
    assert not result.done


def test_reset_full_triage(env):
    result = env.reset("full_triage")
    assert result.observation.task_id == "full_triage"
    assert not result.done


def test_reset_unknown_task_raises(env):
    with pytest.raises(ValueError, match="Unknown task_id"):
        env.reset("nonexistent_task")


def test_reset_clears_state(env):
    env.reset("label_only")
    env.step(EmailTriageAction(label="urgent"))
    env.reset("label_only")  # reset again
    s = env.state()
    assert s.step_count == 0
    assert s.cumulative_reward == 0.0


# ── Step / grading tests ───────────────────────────────────────────────────────


def test_correct_urgent_label_scores_1(env):
    env.reset("label_only")
    # email_001 ground truth: urgent
    result = env.step(EmailTriageAction(label="urgent"))
    assert result.reward == pytest.approx(1.0, abs=0.01)
    assert not result.done


def test_wrong_label_scores_0(env):
    env.reset("label_only")
    result = env.step(EmailTriageAction(label="low"))
    assert result.reward == pytest.approx(0.0, abs=0.01)


def test_adjacent_label_partial_credit(env):
    env.reset("label_only")
    # email_001 is urgent; needs_followup is adjacent → 0.5
    result = env.step(EmailTriageAction(label="needs_followup"))
    assert result.reward == pytest.approx(0.5, abs=0.01)


def test_skip_penalty(env):
    env.reset("label_only")
    result = env.step(EmailTriageAction(skip=True))
    assert result.reward == pytest.approx(-0.05, abs=0.001)


def test_spam_reply_penalty(env):
    env.reset("label_only")
    # Advance to email_007 (spam, index 6)
    for _ in range(6):
        env.step(EmailTriageAction(label="urgent"))
    result = env.step(EmailTriageAction(label="spam", reply="Thanks for your offer!"))
    assert result.reward == pytest.approx(-0.20, abs=0.001)


def test_label_route_full_correct_scores_1(env):
    env.reset("label_route")
    # email_001: urgent / engineering
    result = env.step(EmailTriageAction(label="urgent", route="engineering"))
    assert result.reward == pytest.approx(1.0, abs=0.01)


def test_label_route_correct_label_wrong_route(env):
    env.reset("label_route")
    result = env.step(EmailTriageAction(label="urgent", route="finance"))
    # label=1.0*0.5 + route=0.0*0.5 = 0.5
    assert result.reward == pytest.approx(0.5, abs=0.01)


def test_full_triage_good_action(env):
    env.reset("full_triage")
    result = env.step(
        EmailTriageAction(
            label="urgent",
            route="engineering",
            summary="Production database down for 20 minutes, all transactions failing, 8000 users affected.",
            reply="We have received your alert and are investigating the database issue immediately.",
        )
    )
    assert result.reward > 0.70


def test_full_triage_missing_summary_penalized(env):
    env.reset("full_triage")
    result = env.step(EmailTriageAction(label="urgent", route="engineering"))
    # summary weight=0.25 gives 0.0, reply for needs_reply=True also 0.0
    assert result.reward == pytest.approx(0.60, abs=0.02)


# ── Episode lifecycle tests ────────────────────────────────────────────────────


def test_full_episode_completes_after_12_steps(env):
    env.reset("label_only")
    result = None
    for _ in range(12):
        result = env.step(EmailTriageAction(label="normal"))
    assert result is not None
    assert result.done
    assert env.state().done


def test_step_after_done_returns_done(env):
    env.reset("label_only")
    for _ in range(12):
        env.step(EmailTriageAction(label="normal"))
    # extra step after done
    result = env.step(EmailTriageAction(label="normal"))
    assert result.done
    assert result.reward == 0.0
    assert result.info.get("error") == "episode_already_done"


def test_state_returns_correct_task(env):
    env.reset("label_route")
    s = env.state()
    assert s.task_id == "label_route"
    assert s.step_count == 0
    assert s.total_emails == 12


def test_actions_log_grows_per_step(env):
    env.reset("label_only")
    env.step(EmailTriageAction(label="urgent"))
    env.step(EmailTriageAction(label="normal"))
    s = env.state()
    assert len(s.actions_log) == 2


def test_cumulative_reward_accumulates(env):
    env.reset("label_only")
    # First email is urgent — correct label gives 1.0
    env.step(EmailTriageAction(label="urgent"))
    s = env.state()
    assert s.cumulative_reward == pytest.approx(1.0, abs=0.01)


# ── Oracle score tests ─────────────────────────────────────────────────────────


def test_oracle_label_only_scores_1():
    """Oracle should achieve 1.0 on label_only."""
    env = EmailTriageEnvironment()
    env.reset("label_only")
    total = 0.0
    for email_data in EMAIL_CORPUS:
        gt = email_data["ground_truth"]
        result = env.step(EmailTriageAction(label=gt["label"]))
        total += result.reward
    avg = total / len(EMAIL_CORPUS)
    assert avg == pytest.approx(1.0, abs=0.01), f"Oracle label_only={avg:.3f}, expected 1.0"


def test_oracle_label_route_scores_near_1():
    """Oracle should score ≥ 0.95 on label_route."""
    env = EmailTriageEnvironment()
    env.reset("label_route")
    total = 0.0
    for email_data in EMAIL_CORPUS:
        gt = email_data["ground_truth"]
        route = gt["route"] or ""  # spam has no route
        result = env.step(EmailTriageAction(label=gt["label"], route=route))
        total += result.reward
    avg = total / len(EMAIL_CORPUS)
    assert avg >= 0.95, f"Oracle label_route={avg:.3f}, expected ≥0.95"


def test_oracle_full_triage_scores_above_threshold():
    """Oracle should comfortably exceed success_threshold on full_triage."""
    env = EmailTriageEnvironment()
    env.reset("full_triage")
    total = 0.0
    for email_data in EMAIL_CORPUS:
        gt = email_data["ground_truth"]
        route = gt["route"] or ""
        # Build a summary that covers all key terms
        terms = gt.get("key_terms", [])
        summary = f"Email about {', '.join(terms[:3])} — review required."[:280]
        reply_text = (
            "Thank you for your message. We are reviewing this matter and will respond promptly."
            if gt["needs_reply"]
            else None
        )
        result = env.step(
            EmailTriageAction(
                label=gt["label"],
                route=route,
                summary=summary,
                reply=reply_text,
            )
        )
        total += result.reward
    avg = total / len(EMAIL_CORPUS)
    assert avg >= 0.75, f"Oracle full_triage={avg:.3f}, expected ≥0.75"


# ── Grader edge cases ──────────────────────────────────────────────────────────


def test_invalid_label_scores_0(env):
    env.reset("label_only")
    result = env.step(EmailTriageAction(label="critical"))  # not a valid label
    assert result.reward == pytest.approx(0.0, abs=0.01)


def test_invalid_route_scores_0(env):
    env.reset("label_route")
    result = env.step(EmailTriageAction(label="urgent", route="unknown_dept"))
    # label correct=0.5 + route invalid=0.0 → 0.5
    assert result.reward == pytest.approx(0.5, abs=0.01)


def test_summary_too_long_penalized(env):
    env.reset("full_triage")
    long_summary = "x" * 300  # exceeds 280 char limit
    result = env.step(
        EmailTriageAction(label="urgent", route="engineering", summary=long_summary)
    )
    # summary_score=0.4, label=1.0*0.3, route=1.0*0.3, summary=0.4*0.25, reply=0.0
    assert result.info.get("summary_score") == pytest.approx(0.4, abs=0.01)


def test_spam_correct_routing_is_optional(env):
    env.reset("label_route")
    # Advance to email_007 (spam)
    for _ in range(6):
        env.step(EmailTriageAction(label="normal", route="management"))
    # For spam, no route needed → route="" still scores 1.0
    result = env.step(EmailTriageAction(label="spam", route=""))
    # label=1.0*0.5 + route=1.0*0.5 = 1.0
    assert result.reward == pytest.approx(1.0, abs=0.01)