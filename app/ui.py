import gradio as gr
import requests

API_URL    = "http://127.0.0.1:8000/ask"
UPLOAD_URL = "http://127.0.0.1:8000/upload"
TIMEOUT    = 300


def ask_rag(query: str):
    try:
        r = requests.post(API_URL, json={"query": query}, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return d.get("answer", "No answer returned."), d.get("sources", [])
    except:
        return "❌ Error", []


def upload_pdf(file):
    if file is None:
        return "⚠️ Select a PDF first."
    try:
        with open(file.name, "rb") as f:
            r = requests.post(UPLOAD_URL, files={"file": f}, timeout=120)
        return "✅ Uploaded" if r.status_code == 200 else "❌ Failed"
    except:
        return "❌ Failed"


def chat(user_msg, history, all_sessions, current_idx):
    if history is None:
        history = []
    if not user_msg.strip():
        return history, history, all_sessions, current_idx, ""

    answer, _ = ask_rag(user_msg)

    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]

    return history, history, all_sessions, current_idx, ""


def clear_chat():
    return [], [], ""


CSS = """
html, body { height:100%; margin:0; overflow:hidden; }

.gradio-container {
    height:100vh !important;
    width:100vw !important;
    max-width:100% !important;
    margin:0 !important;
    padding:0 !important;
}

/* HEADER */
#app-header {
    height:78px;
    display:flex;
    align-items:center;
    padding:0 28px;
    background:#faf9f7;
    border-bottom:1px solid #e6e4df;
    position:fixed;
    top:0; left:0; right:0;
    z-index:10;
}

/* LOGO */
#app-logo {
    width:52px;
    height:52px;
    font-size:24px;
    border-radius:14px;
    background:#1c1c1a;
    color:#fff;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-right:14px;
}

/* NAME */
#app-name {
    font-size:26px;
    font-weight:700;
    color:#1c1c1a;
}

/* BODY */
#page-body {
    position:fixed;
    top:78px;
    bottom:0;
    left:0;
    right:0;
    display:flex;
}

/* SIDEBAR */
#sidebar {
    width:230px;
    border-right:1px solid #e6e4df;
    padding:10px;
    background:#faf9f7;
}

/* CHAT COLUMN */
#chat-col {
    flex:1;
    display:flex;
    flex-direction:column;
    height:100%;
}

/* CHAT FULL HEIGHT */

/* ❌ REMOVE share / copy / delete buttons from chat messages */
#chatbot .message-buttons,
#chatbot .copy-btn,
#chatbot .share-button,
#chatbot [data-testid="bot"] + div,
.message-buttons {
    display: none !important;
}
#chatbot {
    flex:1;
    width:100%;
    padding:16px;
    overflow-y:auto;
}

/* INPUT BAR */
#input-bar {
    padding:10px 12px;
    border-top:1px solid #e6e4df;
}

/* INPUT BOX */
#input-box {
    display:flex;
    align-items:center;
    width:100%;
    border:1px solid #dedad4;
    border-radius:12px;
    padding:6px 10px;
    background:#fff;
    gap:8px;
}

/* TEXTBOX */
#msg-input {
    flex:1 1 auto !important;
}
#msg-input textarea {
    border:none;
    outline:none;
    width:100%;
    resize:none;
    background:transparent;
    font-size:14px;
    min-height:28px !important;
    max-height:60px;
}

/* BUTTONS */
#send-btn, #clear-btn {
    flex:0 0 auto !important;
}

/* SEND BUTTON */
#send-btn button { 
    background:#1c1c1a !important;
    color:#fff !important;
    border-radius:6px !important;
    height:32px !important;
    padding:0 10px !important;
    min-width:52px !important;
}

/* CLEAR BUTTON */
#clear-btn button {
    border:1px solid #e0deda !important;
    height:32px !important;
    padding:0 8px !important;
    min-width:48px !important;
}
"""


EMPTY = """
<div style="display:flex;flex-direction:column;align-items:center;
justify-content:center;height:100%;text-align:center;gap:10px;">
  <div style="width:50px;height:50px;background:#1c1c1a;
  border-radius:12px;color:white;display:flex;
  align-items:center;justify-content:center;">D</div>
  <div style="font-size:20px;font-weight:600;">DocMind</div>
  <div style="color:#888;">Upload a PDF from the sidebar,<br>then ask anything about it.</div>
</div>
"""


with gr.Blocks(css=CSS) as demo:

    chat_history = gr.State([])
    all_sessions = gr.State([])
    current_idx  = gr.State(0)

    # HEADER (UPDATED)
    gr.HTML("""
    <div id="app-header">
        <div style="display:flex;align-items:center;">
            <div id="app-logo">⬡</div>
            <div id="app-name">DocMind</div>
        </div>
    </div>
    """)

    with gr.Row(elem_id="page-body"):

        # SIDEBAR
        with gr.Column(elem_id="sidebar", scale=0):
            pdf = gr.File(label="Drop PDF")
            upload_btn = gr.Button("Upload")
            upload_status = gr.Textbox(show_label=False)

        # CHAT
        with gr.Column(elem_id="chat-col"):

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                show_label=False,
                placeholder=EMPTY
            )

            with gr.Row(elem_id="input-bar"):
                with gr.Row(elem_id="input-box"):
                    msg = gr.Textbox(
                        placeholder="",
                        show_label=False,
                        elem_id="msg-input",
                        lines=1,
                        max_lines=3
                    )
                    send_btn = gr.Button("Send", elem_id="send-btn")
                    clear_btn = gr.Button("Clear", elem_id="clear-btn")

    upload_btn.click(upload_pdf, inputs=[pdf], outputs=[upload_status])

    send_btn.click(
        chat,
        inputs=[msg, chat_history, all_sessions, current_idx],
        outputs=[chatbot, chat_history, all_sessions, current_idx, msg],
    )

    msg.submit(
        chat,
        inputs=[msg, chat_history, all_sessions, current_idx],
        outputs=[chatbot, chat_history, all_sessions, current_idx, msg],
    )

    clear_btn.click(clear_chat, outputs=[chatbot, chat_history, msg])


if __name__ == "__main__":
    demo.queue().launch()