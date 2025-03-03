# langchain-onnxruntime-genai

This package contains the LangChain integration with OnnxruntimeGenai.

## Installation

```bash
pip install -U langchain-onnxruntime-genai
```

* No specific credentials are required, but you must install the appropriate execution provider package based on your hardware.

## Chat Models

`ChatOnnxruntimeGenai` class exposes chat models from OnnxruntimeGenai.

```python
from langchain_onnxruntime_genai import ChatOnnxruntimeGenai

llm = ChatOnnxruntimeGenai(model="phi3", model_path="/path/to/onnx/model")
llm.invoke("Sing a ballad of LangChain.")
```

## Execution Providers

OnnxruntimeGenai supports multiple execution providers, including CPU, CUDA, and DirectML. You must install the corresponding package based on your selected execution provider:

* **CPU**: `pip install --pre onnxruntime-genai`
* **CUDA**: `pip install --pre onnxruntime-genai-cuda` (Requires CUDA Toolkit. More details: [OnnxruntimeGenai CUDA Installation](https://onnxruntime.ai/docs/genai/howto/install.html#cuda))
* **DirectML**: `pip install --pre onnxruntime-genai-directml`

## Model Configuration

You can instantiate the chat model with various parameters:

```python
chat_model = ChatOnnxruntimeGenai(
    model="phi3",
    model_path="/path/to/onnx/model",
    temperature=0.8,
    max_tokens=256,
    execution_provider="CPU"
)
```

### Supported Parameters:

* `model`: Name of the ONNX model.
* `model_path`: Path to the ONNX model.
* `temperature`: Sampling temperature.
* `max_tokens`: Maximum tokens to generate.
* `execution_provider`: Execution provider (CPU, CUDA, DirectML).
* `top_p`, `top_k`, `repeat_penalty`, etc.

## Streaming

You can stream responses from the model:

```python
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming.")
]

for chunk in chat_model.stream(messages):
    print(chunk)
```

This will output streamed responses in chunks:

```python
AIMessageChunk(content='J')
AIMessageChunk(content="'adore")
AIMessageChunk(content=' la')
AIMessageChunk(content=' programmation')
AIMessageChunk(content='.')
```

## Error Handling

Ensure that the `onnxruntime-genai` package is installed before using this integration. If the model fails to load, verify that the model path and execution provider are correctly set.