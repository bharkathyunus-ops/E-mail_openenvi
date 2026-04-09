"""
Email Triage Environment — Inference Script  v2.0
==================================================

MANDATORY environment variables:
  HF_TOKEN      — HuggingFace / LLM API key (no fallback alias)
  API_BASE_URL  — LLM API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  ENV_BASE_URL  — Environment server URL (default: http://localhost:7860)

STDOUT FORMAT — do not alter field names, order, or formatting:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Rules enforced:
  - One [START] per episode.
  - One [STEP] per env.step() call, immediately after it returns.
  - One [END] per episode, always emitted even on exception.
  - reward / rewards formatted to exactly 2 decimal places.
  - done / success are lowercase true or false.
  - error is the raw error string or null.
  - All fields on a SINGLE line — no embedded newlines.
"""

import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ── Environment variables (EXACT names — no aliases) ──────────────────────────

API_KEY: str = os.getenv("HF_TOKEN", "")          # HF_TOKEN only, no fallback
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

# ── Constants ──────────────────────────────────────────────────────────────────

BENCHMARK: str = "email_triage"
TASKS: List[str] = ["label_only", "label_route", "full_triage", "adversarial_triage"]
MAX_STEPS: int = 18              # safety cap above corpus size (16)
TEMPERATURE: float = 0.6        # non-zero → natural score variation across runs
MAX_TOKENS: int = 300
SUCCESS_THRESHOLD: float = 0.5  # avg reward/step required for success flag
MAX_RETRIES: int = 3             # LLM call retries on transient error
RETRY_DELAY: float = 2.0        # seconds between retries
HTTP_TIMEOUT: int = 30

# ── Vocab (must match server/email_triage_environment.py) ─────────────────────

VALID_LABELS = {"urgent", "normal", "low", "spam", "needs_followup"}
VALID_ROUTES = {
    "engineering", "support", "legal", "finance",
    "hr", "it", "security", "management",
}

# ── Strict stdout helpers ──────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err = error if error else "null"
    action_clean = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── HTTP helpers (all state-mutating = POST per OpenEnv spec) ──────────────────


def _post(path: str, payload: Dict) -> Dict:
    resp = requests.post(
        f"{ENV_BASE_URL}{path}",
        json=payload,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def env_reset(task_id: str) -> Dict[str, Any]:
    return _post("/reset", {"task_id": task_id})


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/step", action)


def env_state() -> Dict[str, Any]:
    return _post("/state", {})


# ── Per-task system prompts ────────────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "label_only": textwrap.dedent("""
        You are an expert email triage assistant. Assign exactly one urgency label.
        Respond with ONLY the label word — nothing else.

        Labels:
          urgent         — requires action within hours (outages, breaches, P1 bugs)
          normal         — requires action within 1–2 business days
          low            — can wait a week or more (social events, low-priority requests)
          spam           — phishing, promotional, or irrelevant
          needs_followup — awaiting external input before action can be taken

        WARNING: Subject lines can be misleading. Read the email body carefully.
        A subject saying "URGENT" does not mean the email is urgent.
    """).strip(),

    "label_route": textwrap.dedent("""
        You are an expert email triage assistant. For each email provide:
          1. Urgency label
          2. Department to route to

        Labels:      urgent | normal | low | spam | needs_followup
        Departments: engineering | support | legal | finance | hr | it | security | management

        For spam emails: do NOT provide a route. Leave route blank.

        WARNING: Read the email body, not just the subject. Subject lines can be deceptive.

        Respond ONLY in this exact format (two lines):
        label: <label>
        route: <department or blank for spam>
    """).strip(),

    "full_triage": textwrap.dedent("""
        You are an expert email triage assistant. For each email provide:
          1. label:   urgent | normal | low | spam | needs_followup
          2. route:   engineering | support | legal | finance | hr | it | security | management
                      (leave blank for spam — do NOT route spam)
          3. summary: concise summary ≤ 280 characters that covers WHO sent it,
                      WHAT they need, and WHY it matters. Be specific — include
                      names, amounts, dates, systems where present.
          4. reply:   a relevant, professional reply if the email needs a response.
                      Write NONE for informational emails, policy updates, and spam.

        WARNING: Read the full body before classifying. Subject lines are sometimes
        misleading. A subject saying "URGENT" may contain trivial content.

        Respond ONLY in this exact format (four lines):
        label: <label>
        route: <department or blank>
        summary: <summary text>
        reply: <reply text or NONE>
    """).strip(),

    "adversarial_triage": textwrap.dedent("""
        You are an expert email triage assistant working with a mixed email queue
        that may include deceptive emails. Some emails have:
          - Subject lines claiming URGENT that contain trivial content
          - Senders that look legitimate but use typosquatted domains
          - Phishing emails disguised as vendor or license notifications

        For each email provide:
          1. label:   urgent | normal | low | spam | needs_followup
          2. route:   engineering | support | legal | finance | hr | it | security | management
                      (leave blank for spam)
          3. summary: ≤ 280 chars — include any suspicious signals you notice
          4. reply:   professional reply if needed, or NONE

        ALWAYS verify: (1) Is the sender domain legitimate? (2) Does body match subject urgency?
        (3) Are there suspicious links or payment requests from unexpected senders?

        Respond ONLY in this exact format (four lines):
        label: <label>
        route: <department or blank>
        summary: <summary text>
        reply: <reply text or NONE>
    """).strip(),
}


# ── Action builder ─────────────────────────────────────────────────────────────


def _parse_kv(text: str) -> Dict[str, str]:
    """Parse key: value lines from LLM output."""
    result: Dict[str, str] = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip().lower()] = val.strip()
    return result


def _safe_label(raw: str) -> str:
    v = raw.strip().lower()
    return v if v in VALID_LABELS else "normal"


def _safe_route(raw: str) -> str:
    v = raw.strip().lower()
    return v if v in VALID_ROUTES else ""


def build_action(
    llm_text: str, task_id: str
) -> Tuple[Dict[str, Any], str]:
    """
    Parse LLM output into action dict + compact single-line action_str for logging.
    Returns sensible defaults on parse failure.
    """
    if task_id == "label_only":
        raw = llm_text.strip().split()[0] if llm_text.strip() else "normal"
        label = _safe_label(raw)
        return {"label": label}, f"label('{label}')"

    kv = _parse_kv(llm_text)
    label = _safe_label(kv.get("label", "normal"))
    route = _safe_route(kv.get("route", ""))

    if task_id == "label_route":
        route_part = f",route='{route}'" if route else ""
        return (
            {"label": label, "route": route} if route else {"label": label},
            f"label('{label}'{route_part})",
        )

    # full_triage or adversarial_triage
    summary_raw = kv.get("summary", "").replace("\n", " ").replace("\r", " ")
    summary = summary_raw[:280].strip()
    reply_raw = kv.get("reply", "NONE").strip()
    reply: Optional[str] = None if reply_raw.upper() == "NONE" else reply_raw

    action: Dict[str, Any] = {"label": label}
    if route:
        action["route"] = route
    if summary:
        action["summary"] = summary
    if reply:
        action["reply"] = reply

    short = (summary[:25] + "..") if len(summary) > 25 else summary
    short = short.replace("'", "\\'")
    return action, f"triage('{label}','{route}',summary='{short}')"


# ── LLM call with retry ────────────────────────────────────────────────────────


def get_llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
    task_id: str,
    step: int,
) -> Tuple[Dict[str, Any], str]:
    """Call LLM with retry logic. Returns (action_dict, action_str)."""
    email = obs.get("email") or {}
    total = obs.get("total_emails", 16)
    feedback = obs.get("last_action_feedback") or ""
    last_reward = obs.get("last_reward", 0.0)

    user_lines = [
        f"Email {step}/{total}",
        f"From: {email.get('sender', '')}",
        f"Subject: {email.get('subject', '')}",
        "",
        email.get("body", ""),
    ]
    if feedback:
        user_lines.append(f"\n[Previous feedback: {feedback} (reward={last_reward:.2f})]")
    user_prompt = "\n".join(user_lines)

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = (completion.choices[0].message.content or "").strip()
            return build_action(text, task_id)
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                print(
                    f"[DEBUG] LLM retry {attempt + 1}/{MAX_RETRIES} (step {step}): {exc}",
                    file=sys.stderr, flush=True,
                )
                time.sleep(RETRY_DELAY)
            else:
                print(
                    f"[DEBUG] LLM failed after {MAX_RETRIES} attempts (step {step}): {exc}",
                    file=sys.stderr, flush=True,
                )

    return {"label": "normal"}, "label('normal')"


# ── Episode runner ─────────────────────────────────────────────────────────────


def run_task(client: OpenAI, task_id: str) -> None:
    """Run one full episode. Emits [START] … [STEP]×N … [END]."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False

    try:
        reset_result = env_reset(task_id)
        obs = reset_result.get("observation", reset_result)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("episode_done", False):
                break

            action_dict, action_str = get_llm_action(client, obs, task_id, step)

            step_result = env_step(action_dict)
            reward = float(step_result.get("reward", 0.0))
            done = bool(step_result.get("done", False))
            error: Optional[str] = (step_result.get("info") or {}).get("error") or None

            rewards.append(reward)
            steps_taken = step
            obs = step_result.get("observation", {})

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        avg = sum(rewards) / len(rewards) if rewards else 0.0
        success = avg >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error task={task_id}: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] HF_TOKEN is not set. Export it before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(
        f"[INFO] inference v2.0 | model={MODEL_NAME} | env={ENV_BASE_URL}",
        file=sys.stderr, flush=True,
    )

    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
