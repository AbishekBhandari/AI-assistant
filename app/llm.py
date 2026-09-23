import json
import time

from openai import AsyncOpenAI

from app.cache import response_cache
from app.config import (
    GROQ_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    TEMPERATURE,
    TOP_P
)
from app.rag import search_documents
from tenacity import retry, stop_after_attempt, wait_exponential


client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

vllm_client = AsyncOpenAI(
    api_key="dummy",
    base_url=VLLM_BASE_URL
)


SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question using the provided document context.

If the context does not contain the answer, say that the information
was not found in the documents.

Return ONLY valid JSON:

{
  "answer": "your answer",
  "topic": "short topic",
  "confidence": 0.0
}

Rules:
- answer must be a string
- topic must be a short string
- confidence must be a number between 0 and 1
- do not include markdown
- do not include additional fields
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=8
    )
)
async def generate_response(message: str) -> dict:

    # -------------------------
    # Cache
    # -------------------------
    if message in response_cache:
        return response_cache[message]

    # -------------------------
    # RAG
    # -------------------------
    start_time = time.perf_counter()

    rag_start = time.perf_counter()

    try:
        context = search_documents(message)
    except Exception as e:
        print(f"RAG error: {e}", flush=True)
        context = []

    rag_time = time.perf_counter() - rag_start

    context_text = "\n\n".join(context)

    # -------------------------
    # Prompt
    # -------------------------
    user_prompt = f"""
Document context:

{context_text}

User question:

{message}
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # -------------------------
    # Groq
    # -------------------------
    try:

        llm_start = time.perf_counter()

        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

        llm_time = time.perf_counter() - llm_start

        print(f"RAG time: {rag_time:.3f}s", flush=True)
        print(f"LLM time: {llm_time:.3f}s", flush=True)

    # -------------------------
    # vLLM fallback
    # -------------------------
    except Exception as e:

        print(f"Groq error: {e}", flush=True)
        print("Falling back to vLLM...", flush=True)

        response = await vllm_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

    total_time = time.perf_counter() - start_time

    print(f"Total time: {total_time:.3f}s", flush=True)

    # -------------------------
    # Parse JSON
    # -------------------------
    content = response.choices[0].message.content

    result = json.loads(content)

    # -------------------------
    # Cache
    # -------------------------
    response_cache[message] = result

    return result