import io
import sys
import gradio as gr
from inference import run_episode


def run_wrapper():
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        run_episode()
        sys.stdout = old_stdout
        output = buffer.getvalue()
        return f"Episode completed successfully.\n\n{output}"
    except SystemExit as e:
        sys.stdout = old_stdout
        output = buffer.getvalue()
        return f"Episode exited (code={e.code}).\n\n{output}"
    except Exception as e:
        sys.stdout = old_stdout
        output = buffer.getvalue()
        return f"Error: {type(e).__name__}: {e}\n\n{output}"


demo = gr.Interface(
    fn=run_wrapper,
    inputs=[],
    outputs="text",
    title="SENTINEL-PR Runner",
    description="Click Run to start a SENTINEL-PR episode.",
)

if __name__ == "__main__":
    demo.launch()
