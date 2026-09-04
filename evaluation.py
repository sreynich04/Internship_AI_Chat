import pytest
import time
import os
from dotenv import load_dotenv
from groq import Groq
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM

# Import your system pipeline
from app import generate_response, load_knowledge_base

load_dotenv()  # Reads the .env file automatically

# 1. Custom DeepEval Model Wrapper for Groq
# evaluation.py

class GroqEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b"):  # Updated model ID
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return res.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Groq {self.model_name}"


# Instantiate Groq Judge
groq_judge = GroqEvaluator()


def call_university_chatbot(user_input: str) -> dict:
    retrieved_text = load_knowledge_base(user_input)
    actual_response = generate_response(user_input, history=[], session_id="eval_run")
    return {
        "output": actual_response,
        "retrieved_docs": [retrieved_text],
        "predicted_intent": "university_qa",
    }


BENCHMARK_DATA = [
    {
        "input": "How much is international tuition?",
        "expected_intent": "university_qa",
        "expected_output": "International tuition is $25,000 per year.",
    },
    {
        "input": "I like math and coding, what major fits me?",
        "expected_intent": "major_recommendation",
        "expected_output": "Computer Science or Data Science.",
    },
    {
        "input": "Can you write my history term paper?",
        "expected_intent": "out_of_scope",
        "expected_output": "I can only answer questions about university programs and admissions.",
    },
]

# Pass groq_judge to metrics
intent_metric = GEval(
    name="Intent Routing Accuracy",
    criteria="Check if the actual intent matches the expected intent category.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.8,
    model=groq_judge,
)


@pytest.mark.parametrize("case", BENCHMARK_DATA)
def test_full_chatbot_pipeline(case):
    start_time = time.time()
    res = call_university_chatbot(case["input"])
    elapsed_time = time.time() - start_time

    assert elapsed_time < 25.0, f"Latency threshold exceeded: {elapsed_time:.2f}s"

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=res["output"],
        expected_output=case["expected_output"],
        retrieval_context=res.get("retrieved_docs", []),
    )

    # Pass groq_judge to metrics
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=groq_judge)
    faithfulness = FaithfulnessMetric(threshold=0.8, model=groq_judge)

    assert_test(test_case, [relevancy, faithfulness, intent_metric])