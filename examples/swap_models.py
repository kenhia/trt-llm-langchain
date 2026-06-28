"""Switch models at the call site — the core ChatTrtLlm value over a fixed endpoint.

Constructing a second ChatTrtLlm with a different `model` triggers a model swap on the next call.
On a single GPU this is **restart-based**: unload does not reclaim VRAM (TensorRT-LLM pool
retention), so the backend is restarted to free VRAM before the new model loads.
See trt-llm-explore sprint 006 / WI #91.

To make the swap automatic, point the client at a restart command, e.g.:

    export TRTLLM_RESTART_CMD="just -C /home/ken/src/ai/trt-llm-explore restart"
    # or: export TRTLLM_RESTART_CMD="docker restart trt-llm-explore-triton-1"

Without it, the swap raises BackendRestartRequiredError with guidance (caught below) — restart the
backend manually (`just swap <key>`) and rerun.

Prereq: a backend running, with engines built/set up for BOTH models below.
Run: uv run python examples/swap_models.py
"""

from trt_llm_langchain import BackendRestartRequiredError, ChatTrtLlm, InsufficientVramError


def ask(model_key: str, question: str) -> None:
    print(f"\n--- {model_key} ---")
    chat = ChatTrtLlm(model=model_key, max_tokens=64)
    try:
        print(chat.invoke(question).content)
    except BackendRestartRequiredError as exc:
        print(f"[swap needs restart] {exc}")
    except InsufficientVramError as exc:
        print(f"[out of VRAM] {exc}")


def main() -> None:
    ask("qwen2_5-coder-7b-fp16", "Write a one-line Python factorial.")
    # Different model => restart-based swap on the next call (auto if TRTLLM_RESTART_CMD is set).
    ask("llama-3_1-8b-fp16", "In one sentence, what is a tensor?")


if __name__ == "__main__":
    main()
