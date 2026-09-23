import asyncio
import json

import app.agent as agent_module


# ---------------------------------------------------------
# Injected failure
# ---------------------------------------------------------
# Replace the real retrieval function with malformed output.
# The returned objects intentionally do not contain the
# required "source" and "text" fields.
# ---------------------------------------------------------

def malformed_search_documents(
    query,
    limit=2,
    source=None
):
    return [
        {
            "invalid_field": "malformed retrieval output"
        }
    ]


# Replace the retrieval function used by the agent.
agent_module.search_documents = malformed_search_documents


async def run_failure_test():

    question = "Compare the four research papers in the documents."

    print()
    print("=" * 70)
    print("FAILURE INJECTION TEST")
    print("=" * 70)

    print()
    print("Injected failure:")
    print("Malformed retrieval output")

    print()
    print(f"Question: {question}")

    try:

        answer, trace = await agent_module.run_agent(
            question,
            return_trace=True
        )

        # If the agent completes despite malformed evidence,
        # the failure test has failed.
        if trace["completed"]:

            result = {
                "test": "Malformed retrieval output",
                "passed": False,
                "classification": "Hard failure",
                "reason": (
                    "The agent completed the task despite "
                    "malformed retrieval evidence."
                ),
                "trace": trace
            }

        else:

            result = {
                "test": "Malformed retrieval output",
                "passed": True,
                "classification": trace["failure"]["type"],
                "reason": trace["failure"]["reason"],
                "trace": trace
            }

    except Exception as e:

        # An exception caused by invalid evidence means
        # the system did not produce a confident answer.
        result = {
            "test": "Malformed retrieval output",
            "passed": True,
            "classification": "Hard failure",
            "reason": (
                "The malformed retrieval output caused the "
                f"system to stop safely: {str(e)}"
            )
        }

    print()
    print("=" * 70)
    print("FAILURE TEST RESULT")
    print("=" * 70)

    print()
    print(
        f"Passed: {result['passed']}"
    )

    print(
        f"Classification: "
        f"{result['classification']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )

    print()
    print("=" * 70)

    # Save result
    output_file = "evaluation/failure_test_results.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    print()
    print(
        f"Results saved to: {output_file}"
    )


if __name__ == "__main__":

    asyncio.run(
        run_failure_test()
    )