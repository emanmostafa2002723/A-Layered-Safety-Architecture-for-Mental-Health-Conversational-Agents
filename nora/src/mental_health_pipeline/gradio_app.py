"""Private Gradio console; launch from the notebook or ``python -m``."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import threading

from .bootstrap import build_pipeline
from .cli import load_env_file
from .config import PipelineConfig, RuntimeMode
from .testing_console import TestingConsole, make_request


EXAMPLES = [
    ["Hello, I had a difficult day and would like someone to listen."],
    ["What is cognitive behavioral therapy, and how can it help with anxiety?"],
    ["Can you explain CBT for someone who worries a lot?"],
    ["I feel overwhelmed before exams. What could I try today?"],
    ["Should I change my prescribed medication dose?"],
    ["I have been having thoughts about suicide, but I do not plan to act on them."],
]


def build_demo(console: TestingConsole):
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    def send(message, history, state, adult, country, confirmed):
        if not state:
            state = console.new_session()
        try:
            request = make_request(message, state, adult, country, confirmed)
        except (ValueError, TypeError) as error:
            raise gr.Error(str(error)) from None
        try:
            result = console.process(request)
        except Exception as error:
            raise gr.Error("Pipeline failed: " + type(error).__name__ + ". Check the notebook status cell.") from None
        response, report = result.response, result.diagnostics
        history = list(history or []) + [{"role": "user", "content": message}]
        if response.response.strip():
            history.append({"role": "assistant", "content": response.response})
        # Only the resolved/guarded response reaches the chatbot.
        state = {**state, "turn": state["turn"] + 1}
        component = report["components"]
        summary = (f"Route: {response.route} | Fallback: {response.used_fallback} | "
                   f"Server time: {report['total_seconds']:.2f} s | "
                   f"Output: {report['output_guard_status']}")
        summary += ('\nGenerator: ' + str(report.get('generator_adapter') or 'not configured')
            + ' | Prompt: ' + str(report.get('generator_prompt_profile','not recorded')))
        execution=report.get('execution_experiments',{})
        if execution:
            summary += '\nLatency experiment: ' + str(execution)
        if component['generator'] and isinstance(component['generator'][0], dict):
            turn_policy = component['generator'][0]
            summary += ('\nInitial turn-specific instructions: '
                + ('present' if turn_policy.get('turn_specific_policy_present') else 'none (general prompt only)'))
        if response.output_guard_status == "output_guard_abstained":
            summary += "\nNO ANSWER RELEASED: output guard withheld the candidate. Inspect the final gate reasons below."
        crisis_result = component["crisis"][-1] if component["crisis"] else {}
        boundary_result = component["boundary"][-1] if component["boundary"] else {}
        applied_policy = component["contact_display_policy"][-1] if component["contact_display_policy"] else {}
        if isinstance(crisis_result, dict):
            summary += ("\nCrisis action: " + str(crisis_result.get("model_output", {}).get("action", "not recorded"))
                        + " | Labels: " + str(crisis_result.get("categories", [])))
        if isinstance(boundary_result, dict):
            summary += ("\nBoundary: " + str(boundary_result.get("suggested_route", "not recorded"))
                        + " | Status: " + str(boundary_result.get("status", "not recorded")))
        if isinstance(applied_policy, dict):
            display_flags = [f for f in applied_policy.get("policy_flags", []) if f.startswith("contact_display:")]
            summary += "\nContact display: " + ", ".join(display_flags)
        indicator=report.get('conversation_signal_indicator',{})
        summary += ("\nConversation count annotation: " + str(indicator.get('conversation_count_label','not recorded'))
                    + " (" + str(indicator.get('classified_crisis_messages',0)) + "/4 classified messages). "
                    "This does not control this turn's safety handling.")
        return ("", history, state, summary, report["timings"],
                component["crisis"], component["boundary"],
                {"built_policy": component["response_policy"], "applied_display_policy": component["contact_display_policy"]},
                component["rag"], component["generator"],
                {"supplemental_rules": component["output_guard"], "conversation_quality": component["conversation_quality"],
                 "final_output_gate": component.get("final_gate", [])},
                {"response": response.public_dict(), "measurement": report})

    def reset(state):
        state = console.new_session(state.get("id") if state else None)
        return ("", [], state, "New conversation; previous history and safety state cleared.",
                [], [], [], [], [], [], [], {})

    with gr.Blocks(title="Mental Health Pipeline — Test Console", analytics_enabled=False) as demo:
        gr.Markdown("# Mental Health Pipeline\nTest conversations and inspect each pipeline component.")
        cfg=console.pipeline.config
        gr.Markdown('RAG router: `' + cfg.rag_router_backend + '` | Speculative RAG: `'
                    + str(cfg.experimental_parallel_rag) + '` | Buffered prefix veto: `'
                    + str(cfg.experimental_incremental_output) + '`. Qwen output checking is a separate component and remains enabled.')
        gr.Markdown("Models stay loaded between turns. Responses appear only after output handling. "
                    "Output checking is fail-closed; the local Qwen backend is for development evaluation, not clinical acceptance. "
                    "No answer is released on abstention. Crisis epoch 3 uses its paired calibration, but independent acceptance is pending; transcription is not connected.")
        state = gr.State(value=lambda: console.new_session(), time_to_live=3600)
        with gr.Row():
            adult = gr.Checkbox(label="I confirm this test uses the adult (18+) scope", value=False)
            country = gr.Textbox(label="Country code (optional)", value="CA", max_lines=1)
            confirmed = gr.Checkbox(label="Country is confirmed for this test", value=False)
        chat = gr.Chatbot(label="Conversation", height=420, sanitize_html=True,
                          allow_tags=False, render_markdown=False)
        message = gr.Textbox(label="Your test message", placeholder="Type a message…", lines=3, max_lines=6)
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("New conversation")
        summary = gr.Textbox(label="Last turn — copy this box when reporting a problem", lines=4,
                             value="Ready. The first turn can include model loading.", interactive=False)
        gr.Examples(EXAMPLES, inputs=message, label="Synthetic test examples (click, then Send)")
        with gr.Accordion("Component timings and outputs", open=False):
            gr.Markdown("Times are server wall time. Crisis and boundary run in parallel. Qwen and retrieval rows "
                        "are inside the RAG time; do not add all rows. Evidence included in a prompt does not prove causal influence.")
            timings = gr.Dataframe(headers=["Component", "Call", "Seconds", "Status"],
                                   datatype=["str", "str", "number", "str"], interactive=False, value=[])
            with gr.Tabs():
                with gr.Tab("Crisis"):
                    crisis = gr.JSON(label="Exact classifier outputs, or recorded failure")
                with gr.Tab("Boundary"):
                    boundary = gr.JSON(label="Exact boundary outputs, or recorded failure")
                with gr.Tab("Response policy"):
                    policy = gr.JSON(label="Policy applied to this turn")
                with gr.Tab("Qwen + RAG"):
                    rag = gr.JSON(label="Canonical query, router decision, passages and sources")
                with gr.Tab("Mistral"):
                    generator = gr.JSON(label="Prompt evidence, policy instructions, tokens and citations; no draft text")
                with gr.Tab("Output handling"):
                    output = gr.JSON(label="Guard verdicts (including retries)")
                with gr.Tab("Full result"):
                    full = gr.JSON(label="Displayed response and test diagnostics (in memory only)")
        outputs = [message, chat, state, summary, timings, crisis, boundary, policy, rag, generator, output, full]
        inputs = [message, chat, state, adult, country, confirmed]
        submit.click(send, inputs, outputs, concurrency_limit=1, concurrency_id="pipeline", api_name=False)
        message.submit(send, inputs, outputs, concurrency_limit=1, concurrency_id="pipeline", api_name=False)
        clear.click(reset, [state], outputs, concurrency_limit=1, concurrency_id="pipeline", api_name=False)
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path, default=Path(".env.qwen-server"))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root-path", default="")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    env_file = args.env_file if args.env_file.is_absolute() else root / args.env_file
    load_env_file(env_file)
    config = PipelineConfig.from_env(project_root=root)
    if config.runtime_mode is not RuntimeMode.DEVELOPMENT:
        raise ValueError("The diagnostic console requires development mode.")
    # Keep Gradio caches under the portable project as well.
    temporary = root / ".runtime" / "gradio-temp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(temporary)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    console = TestingConsole(build_pipeline(config))
    demo = build_demo(console)
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    try:
        demo.launch(server_name="127.0.0.1", server_port=args.port,
                    root_path=args.root_path, share=False, inbrowser=False,
                    prevent_thread_lock=True, show_error=False, ssr_mode=False,
                    blocked_paths=[str(root)], state_session_capacity=100)
        print("Gradio console ready; model loading occurs on the first relevant turn.", flush=True)
        stopping.wait()
    finally:
        demo.close()
        console.close()


if __name__ == "__main__":
    main()
