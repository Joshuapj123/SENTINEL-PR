"""
server/app.py  |  SENTINEL-PR  |  OpenEnv 2026 server entry point
NOTE: This is NOT the same as root app.py (Gradio UI).
      This file is required by the OpenEnv multi-mode deployment validator.
      It imports and re-exports the root FastAPI+Gradio app.
"""
import sys
import os

# Add repo root to path so root app.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import root app under an alias to avoid any naming confusion
import app as _root_app

# Re-export the FastAPI app object (required by OpenEnv validator)
app = _root_app.app


def main() -> None:
    """Server entry point for OpenEnv multi-mode deployment."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
