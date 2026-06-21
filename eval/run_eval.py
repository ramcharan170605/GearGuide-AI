"""
Evaluation runner for Dealer Assistant

Runs the assistant against the eval set and reports results.

Requirements: EVAL-001 to EVAL-011
Tasks: P4-T006, P4-T007, P4-T008
"""

import json
import sys
from datetime import datetime
from typing import Any
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, '.')

from assistant.agent import DealerAssistant
from assistant.retrieval import CatalogueRetriever
from assistant.tools import create_default_tool_registry


def load_eval_set(path: str = "eval/eval_set.jsonl") -> list[dict]:
    """
    Load evaluation test cases from JSONL file.
    Args: path - Path to eval set file
    Returns: list[dict] - Evaluation test cases
    Requirements: EVAL-001 to EVAL-005
    Task: P4-T006
    """
    test_cases = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))
    return test_cases


def check_pass_criteria(response: str, test_case: dict) -> tuple[bool, str]:
    """
    Check if response passes the test case criteria.
    Args: response - Assistant response, test_case - Test case dictionary
    Returns: tuple[bool, str] - (passed, reason)
    Requirements: EVAL-005
    Task: P4-T006
    """
    response_lower = response.lower()
    expected_contains = test_case.get('expected_behavior', {}).get('response_contains', [])
    
    # Check for expected content
    missing = []
    for expected in expected_contains:
        if expected.lower() not in response_lower:
            missing.append(expected)
    
    if missing:
        return False, f"Missing expected content: {missing}"
    
    # Check for clarification (if applicable)
    expected_action = test_case.get('expected_behavior', {}).get('action')
    if expected_action == "clarification":
        if not any(w in response_lower for w in ["which", "what", "please", "provide", "specify"]):
            return False, "Expected clarification but got direct response"
    
    if expected_action == "guardrail":
        if not any(w in response_lower for w in ["not sure", "can't", "not able", "unable", "sorry"]):
            return False, "Expected guardrail response but got direct answer"
    
    return True, "Passed"


def run_evaluation(assistant, eval_set: list[dict]) -> dict:
    """
    Run the assistant against the eval set.
    Args: assistant - Initialized Dealer Assistant, eval_set - List of evaluation test cases
    Returns: dict - Evaluation results with metrics
    Requirements: EVAL-006, EVAL-007
    Task: P4-T007
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(eval_set),
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
        "by_category": {},
        "by_id": {},
        "failure_modes": {}
    }
    
    # Initialize category tracking
    for test_case in eval_set:
        category = test_case.get('category', 'unknown')
        if category not in results['by_category']:
            results['by_category'][category] = {'total': 0, 'passed': 0, 'failed': 0}
        results['by_category'][category]['total'] += 1
    
    # Run each test
    for test_case in eval_set:
        test_id = test_case.get('id', 'unknown')
        query = test_case.get('query', '')
        category = test_case.get('category', 'unknown')
        
        # Create fresh agent for each test to avoid context pollution
        catalogue = pd.read_csv("catalogue.csv")
        retriever = CatalogueRetriever("catalogue.csv")
        retriever.load_catalogue()
        retriever.initialize_embedding_model()
        retriever.build_vector_store()
        tool_registry = create_default_tool_registry(catalogue)
        test_assistant = DealerAssistant(retriever, tool_registry)
        
        response = test_assistant.process_query(query)
        passed, reason = check_pass_criteria(response, test_case)
        
        # Record result
        results['by_id'][test_id] = {
            'query': query,
            'category': category,
            'passed': passed,
            'reason': reason,
            'response': response[:500]  # Truncate long responses
        }
        
        if passed:
            results['passed'] += 1
            results['by_category'][category]['passed'] += 1
        else:
            results['failed'] += 1
            results['by_category'][category]['failed'] += 1
            failure_mode = reason.split(':')[0] if ':' in reason else reason
            results['failure_modes'][failure_mode] = results['failure_modes'].get(failure_mode, 0) + 1
    
    # Calculate pass rate
    if results['total_tests'] > 0:
        results['pass_rate'] = results['passed'] / results['total_tests']
    
    # Calculate category pass rates
    for category in results['by_category']:
        total = results['by_category'][category]['total']
        if total > 0:
            results['by_category'][category]['pass_rate'] =                 results['by_category'][category]['passed'] / total
    
    return results


def save_results(results: dict, path: str = "eval/results.json") -> None:
    """Save evaluation results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def print_summary(results: dict) -> None:
    """Print evaluation summary."""
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Total Tests: {results.get('total_tests', 0)}")
    print(f"Passed: {results.get('passed', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Pass Rate: {results.get('pass_rate', 0):.2%}")
    print()
    
    print("By Category:")
    for category in results.get('by_category', {}):
        cat_data = results['by_category'][category]
        print(f"  {category}: {cat_data['passed']}/{cat_data['total']} ({cat_data.get('pass_rate', 0):.2%})")
    print()
    
    if results.get('failure_modes'):
        print("Failure Modes:")
        for mode, count in results.get('failure_modes', {}).items():
            print(f"  {mode}: {count}")
    else:
        print("No failures!")
    print()
    
    if results.get('failed', 0) > 0:
        print("Failed Tests:")
        for test_id, data in results.get('by_id', {}).items():
            if not data.get('passed'):
                print(f"  {test_id} ({data.get('category', 'unknown')}): {data.get('reason', 'Unknown')}")


def main():
    """Main entry point for evaluation runner."""
    print("Loading evaluation set...")
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} test cases")
    
    print("Running evaluation...")
    results = run_evaluation(None, eval_set)  # Assistant created inside run_evaluation
    
    save_results(results)
    print("Results saved to eval/results.json")
    
    print_summary(results)


if __name__ == "__main__":
    main()
