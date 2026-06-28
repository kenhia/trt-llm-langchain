"""Switch models at the call site — the core ChatTrtLlm value over a fixed endpoint.

Constructing a second ChatTrtLlm with a different `model` triggers an unload-then-load swap on
the next call (single GPU holds one model at a time).

NOTE (trt-llm-explore WI #91): on the current backend, unload does not reclaim VRAM, so the
swap load can fail with CUDA OOM until the backend is restarted. This example handles that
explicitly: it catches InsufficientVramError and prints the actionable guidance rather than
leaking a raw CUDA stack. Once WI #91 is resolved (or the backend is restarted between models),
the swap completes cleanly.

Prereq: a backend running, with engines built/set up for BOTH models below.
Run: uv run python examples/swap_models.py
"""

from trt_llm_langchain import ChatTrtLlm, InsufficientVramError


def ask(model_key: str, question: str) -> None:
    print(f"\n--- {model_key} ---")
    chat = ChatTrtLlm(model=model_key, max_tokens=64)
    try:
        print(chat.invoke(question).content)
    except InsufficientVramError as exc:
        print(f"[swap blocked] {exc}")


def main() -> None:
    ask("qwen2_5-coder-7b-fp16", "Write a one-line Python factorial.")
    # Switching model => unload qwen, load llama on next call.
    ask("llama-3_1-8b-fp16", "In one sentence, what is a tensor?")


if __name__ == "__main__":
    main()
