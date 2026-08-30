import json
import requests

API_URL = "http://127.0.0.1:7860/api/chat"
TEST_CASES_FILE = "test_cases.json"

def run_evaluation():
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total_tests = len(test_cases)
    passed_tests = 0

    print("\n==========================================")
    print("      UNI-GUIDE CHATBOT EVALUATION       ")
    print("==========================================\n")

    for test in test_cases:
        session_id = f"eval_session_{test['id']}"
        payload = {
            "message": test["question"],
            "history": [],
            "session_id": session_id
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            result = response.json()
            bot_reply = result.get("response", "")

            # Check if all expected keywords exist in the bot's reply
            keywords_found = [kw.lower() in bot_reply.lower() for kw in test["expected_keywords"]]
            is_passed = any(keywords_found) if test["should_refuse"] else all(keywords_found)

            if is_passed:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            print(f"[{status}] Test #{test['id']} - {test['category']}")
            print(f"  Q: {test['question']}")
            print(f"  A: {bot_reply[:120]}...\n")

        except Exception as e:
            print(f"[❌ ERROR] Test #{test['id']} failed to execute: {e}\n")

    accuracy = (passed_tests / total_tests) * 100
    print("==========================================")
    print(f"FINAL ACCURACY SCORE: {accuracy:.1f}% ({passed_tests}/{total_tests} passed)")
    print("==========================================\n")

if __name__ == "__main__":
    run_evaluation()