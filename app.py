"""
app.py  |  SENTINEL-PR  |  Gradio UI + OpenEnv 2026 REST API
Exposes /reset and /step endpoints required by the OpenEnv validator.
"""
import io
import sys
import traceback

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from env import Action, SentinelPREnv
from inference import run_episode

# ── Global env instance (shared across REST calls) ────────────────────────────
_env: SentinelPREnv = SentinelPREnv(max_steps=12)
_obs = None

# ── FastAPI app (OpenEnv REST endpoints) ──────────────────────────────────────
app = FastAPI()


@app.post("/reset")
async def reset(request: Request):
    global _env, _obs
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = body.get("task_id", "task_eval_auth_flaw")
    try:
        _env = SentinelPREnv(max_steps=12)
        _obs = _env.reset(task_id)
        return JSONResponse(_obs.model_dump())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/step")
async def step(request: Request):
    global _obs
    try:
        body = await request.json()
        action = Action(**body)
        result = _env.step(action)
        _obs = result.observation
        return JSONResponse({
            "observation": result.observation.model_dump(),
            "reward":      result.reward,
            "done":        result.done,
            "terminated":  result.terminated,
            "info":        result.info,
        })
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "env": "SENTINEL-PR", "version": "1.2.0"}


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def run_wrapper():
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        run_episode()
        sys.stdout = old_stdout
        return f"Episode completed successfully.\n\n{buffer.getvalue()}"
    except SystemExit as e:
        sys.stdout = old_stdout
        return f"Episode exited (code={e.code}).\n\n{buffer.getvalue()}"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {type(e).__name__}: {e}\n\n{buffer.getvalue()}"


demo = gr.Interface(
    fn=run_wrapper,
    inputs=[],
    outputs="text",
    title="SENTINEL-PR",
    description="Click Run to start a SENTINEL-PR security audit episode.",
)

# ── Mount Gradio onto FastAPI ─────────────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/")

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
