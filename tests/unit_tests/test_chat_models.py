"""Test chat model integration."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult

from langchain_onnxruntime_genai.chat_models import ChatOnnxruntimeGenai


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