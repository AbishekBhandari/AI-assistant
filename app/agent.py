import json
import asyncio

from openai import AsyncOpenAI

from app.config import (
    GROQ_API_KEY,
    TEMPERATURE,
    TOP_P
)

from app.rag import search_documents


client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


MAX_STEPS = 4
MAX_CHARS_PER_CHUNK = 1600


AVAILABLE_DOCUMENTS = [
    "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.pdf",
    "Depth Anything V2.pdf",
    "Depth Anything.pdf",
    "Vision Transformers for Dense Prediction.pdf"
]


DECISION_PROMPT = """
You are an evidence verification agent for a research assistant.

Your task is to determine whether the available evidence is
sufficient to answer the user's question.

Available documents:

- Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.pdf
- Depth Anything V2.pdf
- Depth Anything.pdf
- Vision Transformers for Dense Prediction.pdf

Return ONLY valid JSON:

{
    "decision": "sufficient",
    "reason": "short explanation",
    "search_query": "",
    "target_source": ""
}

The decision must be exactly:

"sufficient"

or:

"search_again"

Rules:

1. Choose "sufficient" only when the evidence is enough to answer
   the user's question accurately.

2. Choose "search_again" when important evidence is missing.

3. If the user asks to compare all four papers, evidence from
   all four papers must be available before choosing "sufficient".

4. Determine available papers using the SOURCE field.

5. If a paper is missing, choose "search_again".

6. When choosing "search_again":

   - search_query must describe the information needed.
   - target_source must be EXACTLY one of the available document names.
   - choose only ONE missing document at a time.

7. Never invent document names.

8. Do not use general knowledge to fill missing evidence.

9. Do not assume information that is not contained in the evidence.

10. Do not use markdown.

11. Return only these four fields:

decision
reason
search_query
target_source
"""


ANSWER_PROMPT = """
You are a research assistant.

Answer the user's question using ONLY the provided evidence.

Each evidence item contains a SOURCE field identifying the
research paper.

For comparison questions:

- Clearly distinguish information from different papers.
- Compare only aspects supported by the evidence.
- Do not invent missing information.
- Do not use external knowledge.

Return ONLY valid JSON:

{
    "answer": "your answer",
    "topic": "short topic",
    "confidence": 0.0
}

Rules:

- answer must be a string
- topic must be a short string
- confidence must be between 0 and 1
- no markdown
- no additional fields
"""


def compact_context(context):
    """
    Validate and compact retrieved evidence.
    """

    selected = {}

    for item in context:

        if not isinstance(item, dict):
            raise ValueError(
                "Malformed retrieval result: "
                "expected a dictionary."
            )

        if "source" not in item:
            raise ValueError(
                "Malformed retrieval result: "
                "missing 'source' field."
            )

        if "text" not in item:
            raise ValueError(
                "Malformed retrieval result: "
                "missing 'text' field."
            )

        source = item["source"]
        text = item["text"]

        if not isinstance(source, str):
            raise ValueError(
                "Malformed retrieval result: "
                "'source' must be a string."
            )

        if not isinstance(text, str):
            raise ValueError(
                "Malformed retrieval result: "
                "'text' must be a string."
            )

        if source not in selected:

            if len(text) > MAX_CHARS_PER_CHUNK:
                text = text[:MAX_CHARS_PER_CHUNK]

            selected[source] = {
                "source": source,
                "text": text
            }

    return list(selected.values())


def format_context(context):

    context = compact_context(context)

    return "\n\n".join(
        f"SOURCE: {item['source']}\n"
        f"EVIDENCE: {item['text']}"
        for item in context
    )


def get_sources(context):

    return list(
        dict.fromkeys(
            item["source"]
            for item in context
        )
    )


async def groq_request(
    messages,
    temperature=0.0,
    top_p=1.0,
    max_retries=3
):
    """
    Make a Groq request with retry handling
    for temporary rate limits.
    """

    for attempt in range(max_retries):

        try:

            response = await client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=temperature,
                top_p=top_p
            )

            return response

        except Exception as e:

            error_text = str(e)

            if (
                "429" in error_text
                or "rate_limit" in error_text
                or "tokens per minute" in error_text
            ):

                wait_time = 5 * (attempt + 1)

                print(
                    f"Groq rate limit reached. "
                    f"Retrying in {wait_time} seconds...",
                    flush=True
                )

                await asyncio.sleep(
                    wait_time
                )

            else:

                raise

    raise RuntimeError(
        "Groq request failed after retries."
    )


async def evaluate_evidence(
    question,
    context
):

    sources = get_sources(context)

    context_text = format_context(
        context
    )

    prompt = f"""
User question:

{question}

Available documents:

{json.dumps(
    AVAILABLE_DOCUMENTS,
    indent=2
)}

Currently available sources:

{json.dumps(
    sources,
    indent=2
)}

Evidence:

{context_text}

Determine whether the evidence is sufficient.

If evidence from a required document is missing,
choose search_again.

If searching again is necessary, select exactly ONE
missing document as target_source.
"""

    response = await groq_request(
        messages=[
            {
                "role": "system",
                "content": DECISION_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        top_p=1.0
    )

    content = response.choices[0].message.content

    return json.loads(content)


async def generate_final_answer(
    question,
    context
):

    context_text = format_context(
        context
    )

    prompt = f"""
User question:

{question}

Evidence from the research papers:

{context_text}

Write the final answer using only this evidence.
"""

    response = await groq_request(
        messages=[
            {
                "role": "system",
                "content": ANSWER_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P
    )

    content = response.choices[0].message.content

    return json.loads(content)


async def run_agent(
    question,
    return_trace=False
):

    print(
        f"Agent question: {question}",
        flush=True
    )

    # Evaluation information
    trace = {
        "question": question,
        "iterations": [],
        "completed": False,
        "final_answer": None,
        "failure": None
    }

    # -----------------------------------------
    # Initial retrieval
    # -----------------------------------------

    context = search_documents(
        question,
        limit=2
    )

    # -----------------------------------------
    # Agentic loop
    # -----------------------------------------

    for step in range(MAX_STEPS):

        print(
            f"Agent step {step + 1}",
            flush=True
        )

        context = compact_context(
            context
        )

        sources = get_sources(
            context
        )

        print(
            f"Sources currently available: {sources}",
            flush=True
        )

        try:

            evaluation = await evaluate_evidence(
                question,
                context
            )

        except Exception as e:

            trace["failure"] = {
                "type": "hard_failure",
                "reason": f"Agent evaluation failed: {str(e)}"
            }

            if return_trace:
                return None, trace

            raise

        decision = evaluation.get(
            "decision"
        )

        reason = evaluation.get(
            "reason",
            ""
        )

        search_query = evaluation.get(
            "search_query",
            ""
        )

        target_source = evaluation.get(
            "target_source",
            ""
        )

        iteration_record = {
            "step": step + 1,
            "sources_before_decision": sources,
            "decision": decision,
            "reason": reason,
            "search_query": search_query,
            "target_source": target_source
        }

        trace["iterations"].append(
            iteration_record
        )

        print(
            f"Agent decision: {decision}",
            flush=True
        )

        print(
            f"Reason: {reason}",
            flush=True
        )

        # -----------------------------------------
        # Evidence is sufficient
        # -----------------------------------------

        if decision == "sufficient":

            print(
                "Evidence is sufficient.",
                flush=True
            )

            trace["completed"] = True

            break

        # -----------------------------------------
        # Agent decides to search again
        # -----------------------------------------

        if decision == "search_again":

            print(
                f"Agent searching again: "
                f"{search_query}",
                flush=True
            )

            print(
                f"Target source: "
                f"{target_source}",
                flush=True
            )

            # Validate tool arguments
            if not search_query.strip():

                trace["failure"] = {
                    "type": "soft_failure",
                    "reason": "search_again selected without a search query"
                }

                break

            if target_source not in AVAILABLE_DOCUMENTS:

                trace["failure"] = {
                    "type": "soft_failure",
                    "reason": (
                        "Agent selected an invalid target source: "
                        f"{target_source}"
                    )
                }

                break

            # Source-specific retrieval
            new_context = search_documents(
                search_query,
                limit=2,
                source=target_source
            )

            context.extend(
                new_context
            )

            context = compact_context(
                context
            )

            continue

        # -----------------------------------------
        # Invalid agent decision
        # -----------------------------------------

        trace["failure"] = {
            "type": "hard_failure",
            "reason": (
                f"Invalid agent decision: {decision}"
            )
        }

        break

    # -----------------------------------------
    # Maximum steps reached
    # -----------------------------------------

    if not trace["completed"]:

        if trace["failure"] is None:

            trace["failure"] = {
                "type": "hard_failure",
                "reason": (
                    f"Maximum iterations ({MAX_STEPS}) "
                    "reached without sufficient evidence"
                )
            }

        if return_trace:
            return None, trace

        raise RuntimeError(
            trace["failure"]["reason"]
        )

    # -----------------------------------------
    # Generate final answer
    # -----------------------------------------

    print(
        "Generating final answer...",
        flush=True
    )

    try:

        answer = await generate_final_answer(
            question,
            context
        )

    except Exception as e:

        trace["failure"] = {
            "type": "hard_failure",
            "reason": f"Final answer generation failed: {str(e)}"
        }

        if return_trace:
            return None, trace

        raise

    trace["final_answer"] = answer

    print(
        "Agent finished.",
        flush=True
    )

    if return_trace:
        return answer, trace

    return answer