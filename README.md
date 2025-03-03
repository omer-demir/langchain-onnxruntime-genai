# langchain-onnxruntime-genai

This package contains the LangChain integration with OnnxruntimeGenai

## Installation

```bash
pip install -U langchain-onnxruntime-genai
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatOnnxruntimeGenai` class exposes chat models from OnnxruntimeGenai.

```python
from langchain_onnxruntime_genai import ChatOnnxruntimeGenai

llm = ChatOnnxruntimeGenai()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OnnxruntimeGenaiEmbeddings` class exposes embeddings from OnnxruntimeGenai.

```python
from langchain_onnxruntime_genai import OnnxruntimeGenaiEmbeddings

embeddings = OnnxruntimeGenaiEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`OnnxruntimeGenaiLLM` class exposes LLMs from OnnxruntimeGenai.

```python
from langchain_onnxruntime_genai import OnnxruntimeGenaiLLM

llm = OnnxruntimeGenaiLLM()
llm.invoke("The meaning of life is")
```
