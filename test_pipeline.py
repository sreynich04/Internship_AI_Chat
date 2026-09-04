import os
import re
import json
import time
import pytest
from dotenv import load_dotenv

# Core application imports from your project
from app import generate_response, load_knowledge_base, sanitize_khmer_text
from recommendation_engine import rank_majors

load_dotenv()

# ==============================================================================
# 1. FACTIONAL Q&A & OUT-OF-DOMAIN REFUSAL TESTS (Uses test_cases.json)
# ==============================================================================
def load_factual_test_cases():
    json_path = "test_cases.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@pytest.mark.parametrize("case", load_factual_test_cases())
def test_factual_qa_and_refusal(case):
    """Evaluates factual Q&A accuracy, Khmer response grounding, and refusal logic."""
    question = case["question"]
    expected_keywords = case["expected_keywords"]
    
    # Generate direct response from LLM pipeline
    response = generate_response(question, history=[], session_id="eval_factual")
    
    # Normalize unicode/narrow non-breaking spaces (\u202f) to standard spaces
    normalized_response = re.sub(r'\s+', ' ', response).lower()
    
    # Assert expected keywords or refusal strings are present
    assert any(re.sub(r'\s+', ' ', kw).lower() in normalized_response for kw in expected_keywords), \
        f"Test ID {case['id']} ({case['category']}) Failed!\nQuery: {question}\nExpected one of: {expected_keywords}\nActual Response: {response}"


# ==============================================================================
# 2. RAG FILE ROUTING TESTS
# ==============================================================================
@pytest.mark.parametrize("query, expected_file", [
    ("How do I apply to CamTech?", "How to Apply"),
    ("What job opportunities are available?", "Jobs"),
    ("Tell me about Master and PhD programs", "Masters and PhD Programs"),
    ("Are there student exchange programs?", "Student Exchange Programs"),
    ("General information about campus", "Why CamTech"),
])
def test_rag_file_routing(query, expected_file):
    """Verifies that load_knowledge_base retrieves the correct document context."""
    retrieved_text = load_knowledge_base(query)
    assert f"--- {expected_file} ---" in retrieved_text, \
        f"Failed to route '{query}' to {expected_file}.txt"


# ==============================================================================
# 3. KHMER SANITIZATION TESTS
# ==============================================================================
def test_khmer_token_sanitization():
    """Validates that Thai token hallucinations are mapped to standard Khmer."""
    raw_hallucinated = "ព័ត៌មានអំពី หลักสูตร និង มหาวิทยาลัย"
    expected_cleaned = "ព័ត៌មានអំពី កម្មវិធីសិក្សា និង សាកលវិទ្យាល័យ"
    
    result = sanitize_khmer_text(raw_hallucinated)
    assert result == expected_cleaned, f"Sanitization mismatch: {result}"


# ==============================================================================
# 4. RECOMMENDATION VECTOR ENGINE BENCHMARK TESTS
# ==============================================================================
@pytest.mark.parametrize("profile_input, expected_major", [
    ("I enjoy drawing, 3D graphic design, UI layout, and basic web coding.", "Software Engineering"),
    ("I love designing physical buildings, spatial layouts, blueprints, and structural models.", "Architecture"),
    ("I am interested in network security, stopping ethical hackers, and data encryption.", "Cyber Security"),
    ("I want to analyze business data, financial trends, and build market strategies.", "Business Intelligence"),
    ("I want to train machine learning models, analyze big data, and build neural networks.", "AI and Data Science"),
])
def test_vector_recommendation_accuracy(profile_input, expected_major):
    """Evaluates top-1 precision of the vector embedding & keyword-boost engine."""
    results = rank_majors(profile_input, top_k=1)
    assert len(results) > 0, "Recommendation engine returned empty results."
    
    top_predicted = results[0]["major"]
    assert top_predicted == expected_major, \
        f"Query: '{profile_input}' | Expected: {expected_major} | Got: {top_predicted}"


# ==============================================================================
# 5. CONFIDENCE THRESHOLD & MODE SWITCHING
# ==============================================================================
def test_discovery_vs_recommendation_modes():
    """Ensures low confidence (<35%) triggers DISCOVERY mode and high confidence triggers RECOMMENDATION."""
    # Vague query -> Low confidence -> DISCOVERY mode
    low_conf_results = rank_majors("hello, what is this school?", top_k=1)
    low_score = low_conf_results[0]["similarity_score"] if low_conf_results else 0.0
    assert low_score < 35.0, f"Expected low confidence score (<35.0%), got {low_score}%"

    # Specific query -> High confidence -> RECOMMENDATION mode
    high_conf_results = rank_majors("I want to code software in Python and JS", top_k=1)
    high_score = high_conf_results[0]["similarity_score"] if high_conf_results else 0.0
    assert high_score >= 35.0, f"Expected high confidence score (>=35.0%), got {high_score}%"

# ==============================================================================
# 6. END-TO-END FORMATTING RULE & LATENCY ENFORCEMENT
# ==============================================================================
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Requires GROQ_API_KEY in .env")
def test_llm_response_formatting_and_latency():
    """Tests LLM generation latency and dynamic prompt formatting rule compliance."""
    user_query = "I enjoy building web applications, python programming, and UI UX."
    
    start_time = time.time()
    response = generate_response(user_query, history=[], session_id="eval_formatting")
    elapsed = time.time() - start_time
    
    # 1. Latency check (< 25 seconds)
    assert elapsed < 25.0, f"Latency check failed: took {elapsed:.2f}s"
    
    # 2. Allow recommendation format, localized text, or discovery mode follow-up questions
    assert any(rule in response for rule in ["Match Confidence:", "កម្មវិធីសិក្សា", "?"]), \
        f"Response did not enforce strict formatting or language rules. Output:\n{response}"


# ==============================================================================
# 7. LLM-AS-A-JUDGE QUALITY METRICS (DeepEval)
# ==============================================================================
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_KEY", "").startswith("sk-"), 
    reason="Requires valid OPENAI_API_KEY (starting with 'sk-') for DeepEval"
)
def test_deepeval_relevancy_and_faithfulness():
    """Evaluates answer relevancy and RAG faithfulness using DeepEval."""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    query = "What is the phone number of CamTech?"
    context = load_knowledge_base(query)
    response = generate_response(query, history=[], session_id="eval_deepeval")

    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=[context]
    )

    relevancy = AnswerRelevancyMetric(threshold=0.7)
    faithfulness = FaithfulnessMetric(threshold=0.8)

    assert_test(test_case, [relevancy, faithfulness])