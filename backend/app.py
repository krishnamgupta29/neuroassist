# ZeroGPU: `spaces` must be imported before anything initialises CUDA.
try:
    import spaces
    HAS_SPACES = True
except Exception:
    HAS_SPACES = False

import gradio as gr
import torch
import uvicorn

# `main` builds the real FastAPI app: CORS, /uploads, all /api routers,
# /api/health, /docs and the MongoDB lifespan hook.
from main import app

if HAS_SPACES:
    @spaces.GPU
    def run_engine_check():
        return f"ZeroGPU AI Engine Active | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}"
else:
    def run_engine_check():
        return f"AI Engine Active | PyTorch: {torch.__version__} | Threads: {torch.get_num_threads()}"


with gr.Blocks(title="NeuroAssist API") as demo:
    gr.Markdown("# 🧠 NeuroAssist Enterprise AI Diagnostic Platform")
    gr.Markdown("Enterprise AI Screening & FastAPI Service is live and operational.")

    with gr.Row():
        test_btn = gr.Button("⚡ Verify AI Diagnostics Engine", variant="primary")
        status_box = gr.Textbox(label="AI Engine Status", value="Ready")

    test_btn.click(fn=run_engine_check, inputs=[], outputs=[status_box])

    with gr.Row():
        gr.HTML('''
            <div style="display:flex; gap:12px; margin-top:10px;">
                <a href="/docs" target="_blank" style="padding:10px 20px; background:#7A1F2B; color:white; border-radius:10px; text-decoration:none; font-weight:600; font-family:sans-serif;">Open Swagger API Docs 🚀</a>
                <a href="/api/health" target="_blank" style="padding:10px 20px; background:#22201F; color:white; border-radius:10px; text-decoration:none; font-weight:600; font-family:sans-serif;">Health Check 🩺</a>
            </div>
        ''')

# Mount the Gradio UI LAST so it only catches paths the API routes above
# did not already claim. (demo.launch() would build a brand-new FastAPI app
# and silently drop every /api route — that was the 404 bug.)
app = gr.mount_gradio_app(app, demo, path="/")


def report_zerogpu_startup():
    """Register our @spaces.GPU functions with the ZeroGPU scheduler.

    spaces.zero hooks this onto gr.Blocks.launch via one_launch(), but we
    serve through uvicorn instead, so launch() never runs and the Space
    dies with "No @spaces.GPU function detected during startup". Calling
    it directly does exactly what launch() would have done. The attribute
    only exists on ZeroGPU hardware.
    """
    if not HAS_SPACES:
        return
    try:
        from spaces import zero
    except Exception as e:
        print("ZeroGPU startup notice:", e)
        return
    startup = getattr(zero, "startup", None)
    if startup is None:
        return
    try:
        startup()
    except Exception as e:
        print("ZeroGPU startup notice:", e)


# 7860 is the port the Space proxy forwards to. Do not read $PORT — it is
# 7861 here and already held by another process.
if __name__ == "__main__":
    report_zerogpu_startup()
    uvicorn.run(app, host="0.0.0.0", port=7860)
