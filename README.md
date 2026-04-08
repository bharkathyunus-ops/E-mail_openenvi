# README.md
---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - email-triage
  - agent-evaluation
  - reinforcement-learning
pinned: false
---

# 📧 Email Triage Environment

An **OpenEnv-compliant environment** for evaluating AI agents on real-world email triage — one of the highest-volume knowledge-work tasks in any organization.

Agents process a queue of 12 realistic workplace emails spanning security alerts, legal contracts, billing, customer escalations, and more. Three progressive tasks test increasing capability: from basic urgency labeling through to full triage with routing, summarization, and reply drafting.

---

## Motivation

Email triage is a genuine, high-stakes task that humans perform daily. Unlike toy environments, this environment:

- Tests **judgment under ambiguity** (urgency classification with partial credit for adjacent tiers)
- Requires **domain routing knowledge** (which department handles a GDPR audit vs. a production outage?)
- Measures **communication quality** (summary key-term coverage, reply appropriateness)
- Penalizes **clearly wrong behavior** (replying to phishing emails, skipping urgent issues)

---

## Tasks

| Task ID | Difficulty | Description | Max Steps |
|---------|-----------|-------------|-----------|
| `label_only` | Easy | Assign urgency label to each email | 12 |
| `label_route` | Medium | Assign label + route to correct department | 12 |
| `full_triage` | Hard | Label + route + summary (≤280 chars) + reply when needed | 12 |

---

## Action Space

```json
{
  "label":   "urgent | normal | low | spam | needs_followup",
  "route":   "engineering | support | legal | finance | hr | it | security | management",
  "summary": "Concise summary ≤ 280 characters",
  "reply":   "Professional reply snippet, or omit",
  "skip":    false
}
```

| Field | Required for | Notes |
|-------|-------------|-------|
| `label` | All tasks | Urgency classification |
| `route` | `label_route`, `full_triage` | Department routing |
| `summary` | `full_triage` | Must cover key facts, ≤ 280 chars |
| `reply` | `full_triage` | Only when email warrants a response |
| `skip` | Any | Defers email; incurs **−0.05** penalty |

---

## Observation Space

Each step returns:

```json
{
  "observation": {
    "email": {
      "id": "email_001",
      "subject": "URGENT: Production database unresponsive",
      "sender": "devops@acmecorp.com",
      "body": "...",
      "timestamp": "2025-04-07T09:15:00Z"
    },
    "step": 0,
    "total_emails": 12,
    "task_id": "label_only",
    "episode_done": false,
    "last_action_feedback": "scores: label=1.00",
    "last_reward": 1.0
  },
  "reward": 1.0,
  "done": false,
  "info": {
    "label_score": 1.0,
    "task_score": 0.083,
    "cumulative_reward": 1.0,
    "step": 1
  }
}
```

---

## Reward Function

### Sub-scores

| Sub-score | Weight (`full_triage`) | Formula |
|-----------|----------------------|---------|
| Label accuracy | 0.30 | Exact match = 1.0 · Adjacent tier = 0.5 · Wrong = 0.0 |
| Route accuracy | 0.30 | Exact match = 1.0 · Wrong = 0.0 · Spam = 1.0 for any/no route |
| Summary quality | 0.25 | 0.5 (valid length) + 0.5 × (key-term hits / total key terms) |
| Reply quality | 0.15 | 1.0 if needed & well-formed · 0.0 if missing when needed · 0.8 if correctly omitted |

### Weights by task

| Task | Label | Route | Summary | Reply |
|------|-------|-------|---------|-------|
| `label_only` | 1.0 | — | — | — |
| `label_route` | 0.5 | 0.5 | — | — |
| `full_triage` | 0.30 | 0.30 | 0.25 | 0.15 |

### Hard penalties

| Penalty | Reward |
|---------|--------|
| Skip email | −0.05 |
| Reply to spam | −0.20 |

---

## API Endpoints

All state-mutating endpoints use **POST** per OpenEnv spec.

| Method | Endpoint | Body | Description |
|--------|---------|------|-------------|
| `POST` | `/reset` | `{"task_id": "label_only"}` | Start new episode |
| `POST` | `/step` | `{"label": "urgent", "route": "engineering", ...}` | Submit action |
| `POST` | `/state` | `{}` | Query current state |
| `GET` | `/health` | — | Liveness probe |
| `GET` | `/tasks` | — | List all tasks |
| `GET` | `/score` | — | Current episode score |

Interactive docs available at `/docs` (Swagger UI) when running.

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router.
Scores vary per run due to LLM temperature (0.6) — this is by design.

| Task | Typical Avg Reward | Success |
|------|--------------------|---------|
| `label_only` | 0.78 – 0.88 | `true` |
| `label_route` | 0.60 – 0.74 | `true` |
| `full_triage` | 0.44 – 0.62 | `true` |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|---------|---------|-------------|
| `HF_TOKEN` | **Yes** | — | HuggingFace / LLM API key |
| `API_BASE_URL` | **Yes** | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | **Yes** | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:7860` | Environment server URL (for inference.py) |

---

## Setup & Usage

### Local development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
cd email-triage-env

# Install dependencies
pip install -e ".[dev]"

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal — run inference
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# In another terminal — run inference
export HF_TOKEN=hf_your_token
python inference.py
```

### Run tests

```bash
pytest tests/ -v
```

### Validate

```bash
pip install openenv-core
openenv validate
```

---

## Project Structure

```
email-triage-env/
├── inference.py                        # Mandatory baseline script
├── openenv.yaml                        # OpenEnv manifest
├── pyproject.toml                      # Package config + entry-points
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # HF Spaces compatible container
├── README.md                           # This file
├── client.py                           # Async Python client
├── server/
│   ├── __init__.py
│   ├── models.py                       # Pydantic v2 typed models
│   ├── app.py                          # FastAPI server
│   └── email_triage_environment.py     # Core env logic + graders
└── tests/
    ├── __init__.py
    └── test_env.py                     # Full test suite
```

---

## License

MIT