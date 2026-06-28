"""LCEL composition: prompt | model | parser, with ChatTrtLlm.

Shows that ChatTrtLlm drops into a LangChain Expression Language pipe like any chat model.

Prereq: a trt-llm-explore-style backend running (OpenAI proxy on :8003, KServe on :8000).
Run: uv run python examples/lcel_chain.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from trt_llm_langchain import ChatTrtLlm


def main() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a terse assistant. Answer in one sentence."),
            ("human", "{question}"),
        ]
    )
    model = ChatTrtLlm(model="qwen2_5-coder-7b-fp16", max_tokens=128)
    chain = prompt | model | StrOutputParser()

    print(chain.invoke({"question": "What is a CUDA stream?"}))


if __name__ == "__main__":
    main()
