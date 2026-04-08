"""
server/main.py  |  SENTINEL-PR  |  OpenEnv 2026 server entry point
Required by the OpenEnv multi-mode deployment validator.
Delegates to the root app.py FastAPI/Gradio application.
"""
import sys
import os

# Ensure repo root is on path so env, inference, app are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401  – re-export FastAPI+Gradio app


def main() -> None:
    """Server entry point invoked by `server` console script and OpenEnv CLI."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
