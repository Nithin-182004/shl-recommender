"""Evaluation script — tests the agent against conversation traces and checks schema compliance."""

import json
import os
import requests
import sys


API_URL = "http://localhost:8000"


def calculate_recall_at_k(recommended_names, expected_names, k=10):
    """Calculate Recall@K = (relevant in top K) / (total relevant)."""
    rec_set = set(name.lower().strip() for name in recommended_names[:k])
    exp_set = set(name.lower().strip() for name in expected_names)

    if not exp_set:
        return 1.0

    hits = len(rec_set & exp_set)
    return hits / len(exp_set)


def check_schema(response_data):
    """Validate response against the required API schema."""
    errors = []

    if "reply" not in response_data:
        errors.append("Missing 'reply' field")
    elif not isinstance(response_data["reply"], str):
        errors.append("'reply' must be a string")

    if "recommendations" not in response_data:
        errors.append("Missing 'recommendations' field")
    elif not isinstance(response_data["recommendations"], list):
        errors.append("'recommendations' must be a list")
    else:
        for i, rec in enumerate(response_data["recommendations"]):
            for field in ["name", "url", "test_type"]:
                if field not in rec:
                    errors.append(f"Recommendation {i}: missing '{field}'")

    if "end_of_conversation" not in response_data:
        errors.append("Missing 'end_of_conversation' field")

    return errors


def test_health():
    """Test the /health endpoint."""
    print("Testing /health endpoint...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            print("  ✓ Health check passed")
            return True
        else:
            print(f"  ✗ Health check failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"  ✗ Health check error: {e}")
        return False


def test_schema_compliance():
    """Test that the API returns the correct schema."""
    print("\nTesting schema compliance...")

    test_messages = [
        {"role": "user", "content": "I need to hire a Java developer"}
    ]

    resp = requests.post(
        f"{API_URL}/chat",
        json={"messages": test_messages},
        timeout=30
    )
    data = resp.json()
    errors = check_schema(data)

    if errors:
        print(f"  ✗ Schema errors: {errors}")
        return False
    else:
        print("  ✓ Schema is correct")
        print(f"    Reply: {data['reply'][:100]}...")
        print(f"    Recommendations: {len(data['recommendations'])}")
        return True


def test_vague_query_behavior():
    """Test that the agent asks for clarification on vague queries."""
    print("\nTesting vague query behavior...")

    test_messages = [
        {"role": "user", "content": "I need an assessment"}
    ]

    resp = requests.post(
        f"{API_URL}/chat",
        json={"messages": test_messages},
        timeout=30
    )
    data = resp.json()

    if len(data["recommendations"]) == 0:
        print("  ✓ Agent correctly asks for clarification (no recommendations)")
        return True
    else:
        print(f"  ✗ Agent recommended {len(data['recommendations'])} items on vague query")
        return False


def test_off_topic_refusal():
    """Test that the agent refuses off-topic questions."""
    print("\nTesting off-topic refusal...")

    test_messages = [
        {"role": "user", "content": "What is the meaning of life?"}
    ]

    resp = requests.post(
        f"{API_URL}/chat",
        json={"messages": test_messages},
        timeout=30
    )
    data = resp.json()

    if len(data["recommendations"]) == 0:
        print("  ✓ Agent correctly refused off-topic question")
        return True
    else:
        print("  ✗ Agent gave recommendations for off-topic question")
        return False


def run_conversation_trace(trace, trace_num):
    """Run a single conversation trace and calculate recall."""
    print(f"\n--- Trace {trace_num} ---")
    print(f"  Scenario: {trace.get('scenario', 'N/A')}")

    messages = []
    final_response = None

    for i, turn in enumerate(trace.get("conversation", [])):
        user_msg = turn.get("user", "")
        if not user_msg:
            continue

        messages.append({"role": "user", "content": user_msg})

        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"messages": messages},
                timeout=30
            )
            data = resp.json()
            final_response = data

            messages.append({"role": "assistant", "content": data["reply"]})

            schema_errors = check_schema(data)
            if schema_errors:
                print(f"  Turn {i+1}: Schema errors: {schema_errors}")

        except Exception as e:
            print(f"  Turn {i+1}: Error: {e}")
            return 0.0

    if final_response and final_response.get("recommendations"):
        rec_names = [r["name"] for r in final_response["recommendations"]]
        expected = trace.get("expected_assessments", [])

        recall = calculate_recall_at_k(rec_names, expected)
        print(f"  Recommendations: {rec_names}")
        print(f"  Expected: {expected}")
        print(f"  Recall@10: {recall:.2f}")
        return recall
    else:
        print("  No recommendations in final response")
        return 0.0


def main():
    """Run the full evaluation suite."""
    print("=" * 60)
    print("SHL Assessment Recommender — Evaluation")
    print("=" * 60)

    if not test_health():
        print("\nHealth check failed. Is the server running?")
        print("Start it with: uvicorn main:app --port 8000")
        sys.exit(1)

    test_schema_compliance()
    test_vague_query_behavior()
    test_off_topic_refusal()

    traces_path = "data/conversation_traces.json"
    if not os.path.exists(traces_path):
        print(f"\nNo conversation traces found at {traces_path}")
        print("Download them from the assignment and place them there.")
        return

    with open(traces_path, "r") as f:
        traces = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"Running {len(traces)} conversation traces...")
    print("=" * 60)

    recalls = []
    for i, trace in enumerate(traces):
        recall = run_conversation_trace(trace, i + 1)
        recalls.append(recall)

    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print(f"\n{'=' * 60}")
        print(f"RESULTS")
        print(f"{'=' * 60}")
        print(f"Mean Recall@10: {mean_recall:.3f}")
        print(f"Individual: {[f'{r:.2f}' for r in recalls]}")


if __name__ == "__main__":
    main()
