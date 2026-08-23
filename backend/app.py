import gradio as gr
import torch
import uvicorn

# `main` builds the real FastAPI app: CORS, /uploads, all /api routers,
# /api/health, /docs and the MongoDB lifespan hook.
from main import app


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

# 7860 is the port the Space proxy forwards to. Do not read $PORT — it is
# 7861 here and held by another process.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
