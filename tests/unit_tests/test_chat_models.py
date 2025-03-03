"""Test chat model integration."""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_onnxruntime_genai.chat_models import ChatOnnxruntimeGenai


class TestChatOnnxruntimeGenaiUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatOnnxruntimeGenai]:
        return ChatOnnxruntimeGenai

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model_path": "tests/unit_tests/test_data/bird-brain-001.onnx",
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
