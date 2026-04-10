"""
Email Triage Environment — FastAPI Server  v2.0
================================================
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from server.email_triage_environment import (
    EMAIL_CORPUS,
    TASK_CONFIGS,
    EmailTriageEnvironment,
)
from server.models import (
    EmailTriageAction,
    ResetRequest,
    StateRequest,
)

app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv-compliant environment for evaluating AI agents on real-world email triage.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnvironment()

_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

@app.get("/", include_in_schema=False)
async def root(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/ui/")
    return {
        "name": "email_triage",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "dashboard": "/ui",
        "tasks": list(TASK_CONFIGS.keys()),
        "total_emails": len(EMAIL_CORPUS),
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "env": "email_triage",
        "version": "2.0.0",
        "tasks": list(TASK_CONFIGS.keys()),
        "corpus_size": len(EMAIL_CORPUS),
    }

@app.post("/reset")
async def reset(raw_request: Request):
    try:
        body_bytes = await raw_request.body()
        task_id = "label_only"
        if body_bytes and body_bytes.strip() not in (b"", b"null", b"{}"):
            import json
            try:
                data = json.loads(body_bytes)
                if isinstance(data, dict) and data.get("task_id"):
                    task_id = str(data["task_id"])
            except Exception:
                pass
        result = env.reset(task_id=task_id)
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"reset() failed: {exc}")

@app.post("/step")
async def step(action: EmailTriageAction):
    try:
        result = env.step(action)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step() failed: {exc}")

@app.post("/state")
async def state(request: StateRequest):  # noqa: ARG001
    try:
        return env.state().model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"state() failed: {exc}")

@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {
                "id": task_id,
                "name": cfg["name"],
                "description": cfg["description"],
                "difficulty": cfg["difficulty"],
                "max_steps": cfg["max_steps"],
                "success_threshold": cfg["success_threshold"],
                "reward_weights": cfg["weights"],
            }
            for task_id, cfg in TASK_CONFIGS.items()
        ]
    }

@app.get("/score")
async def score():
    s = env.state()
    return {
        "task_id": s.task_id,
        "task_score": s.task_score,
        "cumulative_reward": s.cumulative_reward,
        "steps_taken": s.step_count,
        "total_emails": s.total_emails,
        "done": s.done,
    }

@app.get("/metrics")
async def metrics():
    label_dist = {}
    route_dist = {}
    adversarial_count = 0

    for email in EMAIL_CORPUS:
        gt = email["ground_truth"]
        lbl = gt["label"]
        route = gt.get("route") or "none"
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
        route_dist[route] = route_dist.get(route, 0) + 1
        if email["id"] in ("email_013", "email_014", "email_015", "email_016"):
            adversarial_count += 1

    return {
        "corpus": {
            "total_emails": len(EMAIL_CORPUS),
            "adversarial_emails": adversarial_count,
            "label_distribution": label_dist,
            "route_distribution": route_dist,
        },
        "tasks": {
            task_id: {
                "difficulty": cfg["difficulty"],
                "success_threshold": cfg["success_threshold"],
            }
            for task_id, cfg in TASK_CONFIGS.items()
        },
        "grader_design": {
            "summary_scorer": "ROUGE-1 F1 (unigram overlap, no stopwords)",
            "reply_scorer": "relevance-aware ROUGE-1 against email body + key terms",
            "label_scorer": "exact match + partial credit via adjacency map",
            "route_scorer": "exact match only; spam requires NO route",
            "penalties": {
                "skip": "-0.05",  # CRITICAL FIX: Strings instead of raw negative floats
                "reply_to_spam": "-0.20",
                "diversity_deduction_light": "-0.05",
                "diversity_deduction_heavy": "-0.10",
            },
        },
    }

def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(__import__("os").getenv("PORT", "7860")),
        workers=1,
    )

if __name__ == "__main__":
    main()
    
