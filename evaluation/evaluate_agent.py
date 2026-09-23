import asyncio
import json
from pathlib import Path

from app.agent import (
    run_agent,
    AVAILABLE_DOCUMENTS
)


TEST_QUERIES = [
    {
        "id": "Q1",
        "query": "Compare the four research papers in the documents.",
        "expected_min_iterations": 2
    },
    {
        "id": "Q2",
        "query": "What is the main contribution of Depth Anything V2?",
        "expected_min_iterations": 1
    },
    {
        "id": "Q3",
        "query": "Compare the approaches used for monocular depth estimation.",
        "expected_min_iterations": 2
    },
    {
        "id": "Q4",
        "query": "What datasets are discussed in the research papers?",
        "expected_min_iterations": 1
    },
    {
        "id": "Q5",
        "query": "Compare the methodologies used by Depth Anything and Vision Transformers for Dense Prediction.",
        "expected_min_iterations": 2
    }
]


def classify_failure(trace):
    """
    Classify failures according to the assignment
    failure taxonomy.
    """

    failure = trace.get("failure")

    if not failure:
        return None

    failure_type = failure.get(
        "type",
        "hard_failure"
    )

    reason = failure.get(
        "reason",
        "Unknown failure"
    )

    if failure_type == "hard_failure":

        return {
            "type": "Hard failure",
            "reason": reason
        }

    if failure_type == "soft_failure":

        # If the agent recovered in a later iteration,
        # this is considered a soft failure.
        if trace.get("completed"):

            return {
                "type": "Soft failure",
                "reason": reason
            }

        return {
            "type": "Cascading soft failure",
            "reason": reason
        }

    return {
        "type": "Hard failure",
        "reason": reason
    }


def evaluate_tool_calls(trace):
    """
    Check whether every search_again decision
    selected valid tool arguments.
    """

    total_calls = 0
    correct_calls = 0
    errors = []

    for iteration in trace.get(
        "iterations",
        []
    ):

        if iteration["decision"] != "search_again":
            continue

        total_calls += 1

        search_query = iteration.get(
            "search_query",
            ""
        )

        target_source = iteration.get(
            "target_source",
            ""
        )

        valid_query = (
            isinstance(search_query, str)
            and bool(search_query.strip())
        )

        valid_source = (
            target_source in AVAILABLE_DOCUMENTS
        )

        if valid_query and valid_source:

            correct_calls += 1

        else:

            errors.append({
                "step": iteration["step"],
                "search_query": search_query,
                "target_source": target_source
            })

    return {
        "total": total_calls,
        "correct": correct_calls,
        "errors": errors
    }


async def evaluate_query(test_case):

    query_id = test_case["id"]
    question = test_case["query"]

    print()
    print("=" * 70)
    print(f"{query_id}: {question}")
    print("=" * 70)

    try:

        answer, trace = await run_agent(
            question,
            return_trace=True
        )

    except Exception as e:

        return {
            "id": query_id,
            "query": question,
            "completed": False,
            "iterations": 0,
            "tool_calls": {
                "total": 0,
                "correct": 0,
                "errors": []
            },
            "failure": {
                "type": "Hard failure",
                "reason": str(e)
            }
        }

    iterations = len(
        trace.get(
            "iterations",
            []
        )
    )

    tool_calls = evaluate_tool_calls(
        trace
    )

    failure = classify_failure(
        trace
    )

    completed = trace.get(
        "completed",
        False
    )

    print(
        f"Completed: {completed}"
    )

    print(
        f"Iterations: {iterations}"
    )

    print(
        f"Tool calls: "
        f"{tool_calls['correct']}/"
        f"{tool_calls['total']} correct"
    )

    if failure:

        print(
            f"Failure: {failure['type']}"
        )

        print(
            f"Reason: {failure['reason']}"
        )

    return {
        "id": query_id,
        "query": question,
        "completed": completed,
        "iterations": iterations,
        "tool_calls": tool_calls,
        "failure": failure,
        "answer": answer
    }


async def main():

    print()
    print("=" * 70)
    print("AGENTIC EVALUATION HARNESS")
    print("=" * 70)

    results = []

    # Run sequentially to avoid creating
    # unnecessary Groq TPM pressure.
    for test_case in TEST_QUERIES:

        result = await evaluate_query(
            test_case
        )

        results.append(
            result
        )

        # Small delay between test cases
        # to reduce API rate-limit pressure.
        await asyncio.sleep(5)

    # ------------------------------------------------
    # Task Completion Rate
    # ------------------------------------------------

    completed_count = sum(
        1
        for result in results
        if result["completed"]
    )

    total_queries = len(results)

    completion_rate = (
        completed_count / total_queries * 100
        if total_queries
        else 0
    )

    # ------------------------------------------------
    # Tool Call Correctness
    # ------------------------------------------------

    total_tool_calls = sum(
        result["tool_calls"]["total"]
        for result in results
    )

    correct_tool_calls = sum(
        result["tool_calls"]["correct"]
        for result in results
    )

    tool_accuracy = (
        correct_tool_calls / total_tool_calls * 100
        if total_tool_calls
        else 100
    )

    # ------------------------------------------------
    # Trajectory Length
    # ------------------------------------------------

    trajectory_lengths = [
        result["iterations"]
        for result in results
    ]

    average_iterations = (
        sum(trajectory_lengths)
        / len(trajectory_lengths)
        if trajectory_lengths
        else 0
    )

    # ------------------------------------------------
    # Failure Log
    # ------------------------------------------------

    failures = [
        result
        for result in results
        if result["failure"] is not None
    ]

    # ------------------------------------------------
    # Print Summary
    # ------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Task Completion Rate: "
        f"{completed_count}/{total_queries} "
        f"({completion_rate:.1f}%)"
    )

    print()
    print(
        f"Tool-Call Correctness: "
        f"{correct_tool_calls}/{total_tool_calls} "
        f"({tool_accuracy:.1f}%)"
    )

    print()
    print("Trajectory Length:")

    for result in results:

        print(
            f"  {result['id']}: "
            f"{result['iterations']} iteration(s)"
        )

    print(
        f"\nAverage trajectory length: "
        f"{average_iterations:.2f} iterations"
    )

    print()
    print("Failure Log")
    print("-" * 70)

    if not failures:

        print(
            "No failures recorded."
        )

    else:

        for result in failures:

            failure = result["failure"]

            print(
                f"{result['id']}: "
                f"{failure['type']}"
            )

            print(
                f"  Query: {result['query']}"
            )

            print(
                f"  Reason: {failure['reason']}"
            )

    print()
    print("=" * 70)

    # ------------------------------------------------
    # Save detailed results
    # ------------------------------------------------

    output_file = Path(
        "evaluation/results.json"
    )

    output_file.parent.mkdir(
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "task_completion_rate": completion_rate,
                "tool_call_correctness": tool_accuracy,
                "average_trajectory_length": average_iterations,
                "results": results
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nDetailed results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )