"""Structured output with ChatTrtLlm via `with_structured_output(method="json_mode")`.

On the current backend, function-calling tool_calls are not emitted by the proxy/models, so the
reliable path to typed output is JSON mode: the model returns a JSON object and LangChain parses
it into your Pydantic schema. Make the prompt name the fields you want.

Prereq: a backend running with a chat model loaded.
Run: uv run python examples/structured_output.py
"""

from pydantic import BaseModel, Field

from trt_llm_langchain import ChatTrtLlm


class Person(BaseModel):
    name: str = Field(description="full name")
    age: int = Field(description="age in years")


def main() -> None:
    chat = ChatTrtLlm(max_tokens=128)  # adopts the resident model
    structured = chat.with_structured_output(Person, method="json_mode")

    result = structured.invoke("Return JSON with keys name and age. Ada Lovelace was 36.")
    print(result)  # Person(name='Ada Lovelace', age=36)


if __name__ == "__main__":
    main()
