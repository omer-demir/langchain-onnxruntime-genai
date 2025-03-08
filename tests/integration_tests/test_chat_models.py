from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatResult

from langchain_onnxruntime_genai import ChatOnnxruntimeGenai
from langchain_onnxruntime_genai.execution_providers import ExecutionProviders

MODEL_PATH = "/home/ofdeme/projects/model-download/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/"  # Update with the actual path

@pytest.fixture(scope="module")
def chat_model():
    """Initialize the real chat model on CPU for integration testing."""
    return ChatOnnxruntimeGenai(
        model="phi3",
        model_path=MODEL_PATH,
        execution_provider=ExecutionProviders.CPU,
        temperature=0.7,
        max_tokens=50,
    )


def test_generate(chat_model):
    """Test the _generate method with a real model."""
    messages = [HumanMessage(content="Hello, how are you?")]
    result = chat_model._generate(messages)
    
    assert isinstance(result, ChatResult)
    assert len(result.generations) > 0
    print(result.generations[0].message.content)
    assert result.generations[0].message.content.strip() != ""


@pytest.mark.skip(reason="Requires a GPU-enabled machine")
def test_generate_gpu():
    """Test _generate with a GPU model (skipped by default)."""
    chat_model = ChatOnnxruntimeGenai(
        model="phi3",
        model_path=MODEL_PATH,
        execution_provider=ExecutionProviders.CUDA,
    )
    messages = [HumanMessage(content="Hello, how are you?")]
    result = chat_model._generate(messages)
    assert isinstance(result, ChatResult)


def test_stream(chat_model):
    """Test the _stream method with a real model."""
    messages = [HumanMessage(content="Tell me a joke.")]
    stream = chat_model._stream(messages)
    
    collected_text = ""
    for chunk in stream:
        assert chunk.message.content is not None
        collected_text += chunk.message.content
    
    print(collected_text)
    assert collected_text.strip() != ""


@pytest.mark.skip(reason="Requires a GPU-enabled machine")
def test_stream_gpu():
    """Test _stream on GPU (skipped by default)."""
    chat_model = ChatOnnxruntimeGenai(
        model="phi3",
        model_path=MODEL_PATH,
        execution_provider=ExecutionProviders.CUDA,
    )
    messages = [HumanMessage(content="Tell me a joke.")]
    stream = chat_model._stream(messages)
    collected_text = "".join(chunk.message.content for chunk in stream)
    assert collected_text.strip() != ""