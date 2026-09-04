# a deterministic scoring script that calculates aggregate percentage scores using string and regex matching
# a quick, zero-cost report showing what percentage of factual prompts pass across your dataset

import json
import re
from app import generate_response, load_knowledge_base
from recommendation_engine import rank_majors

def run_accuracy_benchmark():
    # 1. Load Factual Dataset
    with open("test_cases.json", "r", encoding="utf-8") as f:
        factual_cases = json.load(f)
        
    correct_factual = 0
    total_factual = len(factual_cases)

    print(f"\nRunning evaluation across {total_factual} test cases...\n")

    for case in factual_cases:
        query = case["question"]
        expected = case["expected_keywords"]
        
        # Generate chatbot response
        response = generate_response(query, history=[], session_id="accuracy_eval")
        
        # Normalize whitespace (handles \u202f narrow spaces)
        normalized_resp = re.sub(r'\s+', ' ', response).lower()
        
        # Check if any expected keyword is present in the output
        passed = any(re.sub(r'\s+', ' ', kw).lower() in normalized_resp for kw in expected)
        
        status = "PASSED" if passed else "FAILED"
        if passed:
            correct_factual += 1
            
        print(f"[{status}] Test ID {case['id']} ({case['category']}): '{query}'")

    # 2. Compute Percentage Score
    factual_accuracy = (correct_factual / total_factual) * 100 if total_factual > 0 else 0

    # 3. Print Final Report
    print("\n==========================================")
    print("          AI ACCURACY SCORECARD           ")
    print("==========================================")
    print(f"Total Test Cases Evaluated : {total_factual}")
    print(f"Successful Matches         : {correct_factual}")
    print(f"Overall Accuracy Rate      : {factual_accuracy:.2f}%")
    print("==========================================\n")

if __name__ == "__main__":
    run_accuracy_benchmark()