"""
Test Script for IMDB Agent
Tests all 9 sample questions from the assignment
"""

from config import DUCKDB_PATH, CHROMA_PATH
from agents import query_agent
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


# Test questions from assignment
TEST_QUESTIONS = [
    {
        "id": 1,
        "query": "When did The Matrix release?",
        "expected_type": "STRUCTURED"
    },
    {
        "id": 2,
        "query": "What are the top 5 movies of 2019 by meta score?",
        "expected_type": "STRUCTURED"
    },
    {
        "id": 3,
        "query": "Top 7 comedy movies between 2010-2020 by IMDB rating?",
        "expected_type": "STRUCTURED"
    },
    {
        "id": 4,
        "query": "Top horror movies with a meta score above 85 and IMDB rating above 8",
        "expected_type": "STRUCTURED"
    },
    {
        "id": 5,
        "query": "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice",
        "expected_type": "STRUCTURED"
    },
    {
        "id": 6,
        "query": "Top 10 movies with over 1M votes but lower gross earnings",
        "expected_type": "STRUCTURED"
    },
    {
        "id": 7,
        "query": "List of movies from the comedy genre where there is death or dead people involved",
        "expected_type": "HYBRID"
    },
    {
        "id": 8,
        "query": "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies",
        "expected_type": "HYBRID"
    },
    {
        "id": 9,
        "query": "List of movies before 1990 that have involvement of police in the plot",
        "expected_type": "HYBRID"
    }
]


def test_all_queries():
    """Run all test queries and report results"""

    # Verify setup
    if not DUCKDB_PATH.exists():
        print("ERROR: DuckDB database not found. Run: python -m src.data_setup")
        return

    if not CHROMA_PATH.exists():
        print("ERROR: ChromaDB not found. Run: python -m src.data_setup")
        return

    print("\n" + "="*80)
    print("IMDB AGENT TEST SUITE")
    print("="*80)

    results = []

    for test in TEST_QUESTIONS:
        print(f"\n{'='*80}")
        print(f"TEST {test['id']}: {test['query']}")
        print(f"Expected Type: {test['expected_type']}")
        print("="*80)

        try:
            result = query_agent(test['query'])

            success = result['success']
            query_type = result.get('query_type', 'UNKNOWN')
            type_match = query_type == test['expected_type']

            print(f"\n[SUCCESS] Success: {success}")
            print(
                f"Query Type: {query_type} {'[PASS]' if type_match else '[FAIL] (expected ' + test['expected_type'] + ')'}")

            if result.get('sql_query'):
                print(f"\n[SQL] SQL Query:\n{result['sql_query']}")

            if result.get('sql_results'):
                if result['sql_results'].get('success'):
                    print(
                        f"[SUCCESS] SQL executed successfully: {result['sql_results'].get('row_count', 0)} rows")
                else:
                    print(
                        f"[FAIL] SQL error: {result['sql_results'].get('error')}")

            if result.get('semantic_results'):
                if result['semantic_results'].get('success'):
                    print(
                        f"[SUCCESS] Semantic search: {result['semantic_results'].get('count', 0)} results")
                else:
                    print(
                        f"[FAIL] Semantic error: {result['semantic_results'].get('error')}")

            print(
                f"\n[RESPONSE] Response (first 300 chars):\n{result['response'][:300]}...")

            results.append({
                "test_id": test['id'],
                "query": test['query'],
                "success": success,
                "type_match": type_match,
                "error": result.get('error')
            })

        except Exception as e:
            print(f"\n[ERROR] ERROR: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "test_id": test['id'],
                "query": test['query'],
                "success": False,
                "type_match": False,
                "error": str(e)
            })

    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    successful = sum(1 for r in results if r['success'])
    type_matches = sum(1 for r in results if r['type_match'])

    print(f"\nTotal Tests: {len(results)}")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Correct Type Classification: {type_matches}/{len(results)}")

    print("\nDetailed Results:")
    for r in results:
        status = "[PASS]" if r['success'] else "[FAIL]"
        type_status = "[PASS]" if r['type_match'] else "[FAIL]"
        print(
            f"  {status} {type_status} Test {r['test_id']}: {r['query'][:50]}...")
        if r['error']:
            print(f"      Error: {r['error']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_all_queries()
