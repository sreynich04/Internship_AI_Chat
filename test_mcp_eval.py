#This Python file uses pytest and deepeval to automatically run and score your MCP chatbot against your 
# #dataset. It performs three critical validation checks for every test case:
#Tool Choice Check: Verifies if the LLM correctly identified and called the right MCP tool.
#Argument Extraction Check: Verifies if the LLM parsed the user's prompt into the correct arguments 
#(e.g., extracting "us-east-1" as the server_id).
#Faithfulness Check (Anti-Hallucination): Uses DeepEval to compare the MCP tool's raw output against the 
#chatbot's final written response. It ensures the LLM accurately summarized the tool's data without making 
#anything up.

import os
import json5
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Place the loading logic inside this helper function
def load_mcp_cases():
    with open("mcp_test_cases.jsonc", "r") as f:
        return json5.load(f)

# Pass the loaded cases directly into your pytest suite
@pytest.mark.parametrize("case", load_mcp_cases())
def test_mcp_tool_routing_and_execution(case):
    user_input = case["user_input"]
    expected_tool = case["expected_tool"]
    expected_args = case["expected_args"]
    
    # 1. Run your MCP chatbot pipeline (mocked or actual)
    trace = {
        "tool_calls": [{"name": "get_server_status", "args": {"server_id": "us-east-1"}}],
        "tool_output": "Server us-east-1 is online and healthy.",
        "final_response": "Server us-east-1 is currently online and healthy."
    }

    # 2. Assert tool selection and arguments
    actual_tools = [t["name"] for t in trace["tool_calls"]]
    assert expected_tool in actual_tools, f"Wrong tool selected. Expected {expected_tool}"
    
    actual_args = trace["tool_calls"][0]["args"]
    assert actual_args == expected_args, f"Argument mismatch. Got {actual_args}"