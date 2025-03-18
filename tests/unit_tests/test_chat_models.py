"""Test chat model integration."""

from unittest.mock import MagicMock, patch

from langchain_onnxruntime_genai.execution_providers import ExecutionProviders
from pydantic import ValidationError
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult

from langchain_onnxruntime_genai.chat_models import ChatOnnxruntimeGenai

MODEL_PATH = "/Users/omerdemir/Documents/Projects/model_download/models/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/"


@pytest.fixture(scope="module")
def mock_dependencies():
    """Fixture to mock dependencies for all tests."""
    with patch("onnxruntime_genai.Config") as mock_config, \
         patch("onnxruntime_genai.Model") as mock_model, \
         patch("onnxruntime_genai.Tokenizer") as mock_tokenizer, \
         patch("transformers.AutoTokenizer") as mock_auto_tokenizer:
        
        # Configure default mock behavior
        mock_config.return_value.clear_providers.return_value = None
        mock_config.return_value.append_provider.return_value = None
        mock_auto_tokenizer.from_pretrained.return_value = None
        
        yield {
            "mock_config": mock_config,
            "mock_model": mock_model,
            "mock_tokenizer": mock_tokenizer,
            "mock_auto_tokenizer": mock_auto_tokenizer,
        }

def test_to_chatml_format():
    assert ChatOnnxruntimeGenai._to_chatml_format(
        None, HumanMessage(content="Hello")
    ) == {
        "role": "user",
        "content": "Hello",
    }
    assert ChatOnnxruntimeGenai._to_chatml_format(None, AIMessage(content="Hi")) == {
        "role": "assistant",
        "content": "Hi",
    }
    assert ChatOnnxruntimeGenai._to_chatml_format(
        None, SystemMessage(content="System instruction")
    ) == {
        "role": "system",
        "content": "System instruction",
    }
    with pytest.raises(TypeError):
        ChatOnnxruntimeGenai._to_chatml_format(None, "Invalid message")

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

def test_validate_environment_success(chat_model, mock_dependencies):
    """Test validate_environment method when all dependencies are available."""
    result = chat_model.validate_environment()

    assert result == chat_model
    assert chat_model.onnx_model == mock_dependencies["mock_model"].return_value
    assert chat_model.onnx_config == mock_dependencies["mock_config"].return_value
    assert chat_model.tokenizer == mock_dependencies["mock_tokenizer"].return_value


def test_validate_environment_import_error(mock_dependencies):
    """Test validate_environment method when onnxruntime-genai package is missing."""
    with patch("onnxruntime_genai.Config", new=None):
        with pytest.raises(ValidationError, match="Could not load Onnx model from path"):
            chat_model = ChatOnnxruntimeGenai(
                model="phi3",
                model_path=MODEL_PATH,
                execution_provider=ExecutionProviders.CPU,
            )
            chat_model.validate_environment()


def test_validate_environment_model_load_error(chat_model, mock_dependencies):
    """Test validate_environment method when model loading fails."""
    mock_dependencies["mock_config"].side_effect = Exception("Mocked model loading error")

    with pytest.raises(ValueError, match="Could not load Onnx model from path"):
        chat_model.validate_environment()


def test_validate_environment_tokenizer_error(chat_model, mock_dependencies):
    """Test validate_environment method when tokenizer initialization fails."""
    mock_dependencies["mock_tokenizer"].side_effect = Exception("Mocked tokenizer error")

    with pytest.raises(ValueError, match="Could not load Onnx model from path"):
        chat_model.validate_environment()