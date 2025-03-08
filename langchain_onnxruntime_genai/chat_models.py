"""OnnxruntimeGenai chat models."""

from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
    LLMResult,
)
from pydantic import Field, model_validator
from transformers import AutoTokenizer
from typing_extensions import Self

from langchain_onnxruntime_genai.execution_providers import ExecutionProviders


class ChatOnnxruntimeGenai(BaseChatModel):
    """OnnxruntimeGenai chat model integration.

    Setup:
        Install ``langchain-onnxruntime_genai``.
        To use this model, you need to install the ``onnxruntime-genai`` package ``pip install onnxruntime-genai``.
        As part of Onnxruntime GenAi, you can choose different execution providers like CPU, CUDA, DML, etc.
        Based on the execution provide, you need to install respective packages.

        For DML ``pip install --pre onnxruntime-genai-directml``
        For CUDA ``pip install --pre onnxruntime-genai-cuda``
            For CUDA, you need to have CUDA toolkit installed on your machine. For details https://onnxruntime.ai/docs/genai/howto/install.html#cuda
        For CPU ``pip install --pre onnxruntime-genai``

        .. code-block:: bash

            pip install -U langchain-onnxruntime-genai, onnxruntime-genai

    Key init args — completion params:
        model: str
            Name of Onnx model to use.
        model_path: str
            Folder path of Onnx model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_onnxruntime_genai import ChatOnnxruntimeGenai

            chat_model = ChatOnnxruntimeGenai(
                model="phi3",
                model_path="/path/to/onnx/model",
                temperature=0,
                max_tokens=None,
                # execution_provider=ExecutionProviders.CPU,
                # top_p=1,
                # top_k=5,
                # other params...
            )

    **NOTE**: Any param which is not explicitly supported will be passed directly to the
    ``openai.OpenAI.chat.completions.create(...)`` API every time to the model is
    invoked. For example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            import openai

            ChatOpenAI(..., frequency_penalty=0.2).invoke(...)

            # results in underlying API call of:

            openai.OpenAI(..).chat.completions.create(..., frequency_penalty=0.2)

            # which is also equivalent to:

            ChatOpenAI(...).invoke(..., frequency_penalty=0.2)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            chat_model.invoke(messages)

        .. code-block:: python

            AIMessage(content="J'adore la programmation.")

    Stream:
        .. code-block:: python

            for chunk in chat_model.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content='', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='J', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content="'adore", id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content=' la', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content=' programmation', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='.', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='', response_metadata={'finish_reason': 'stop'}, id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')
            AIMessageChunk(content='', id='run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0')

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content="J'adore la programmation.", id='run-bf917526-7f58-4683-84f7-36a6b671d140')

    """  # noqa: E501

    onnx_model: Any = None  #: :meta private:

    onnx_config: Any = None  #: :meta private:

    tokenizer: Any = None  #: :meta private:

    hf_tokenizer: Any = None  #: :meta private:

    model_name: str = Field(alias="model")
    """The name of the model"""

    execution_provider: Optional[ExecutionProviders] = ExecutionProviders.CPU
    """The execution provider to use for the model."""

    model_path: str = Field(alias="model_path")
    """The onnx path of the model folder"""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0.8
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    do_sample: Optional[bool] = False
    """The do_sample value to do sampling or not."""

    n_batch: Optional[int] = Field(8, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    repeat_penalty: Optional[float] = 1.1
    """The penalty to apply to repeated tokens."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Any additional parameters to pass to onnxruntime_genai."""

    verbose: bool = True
    """Print verbose output to stderr."""

    timeout: Optional[int] = None
    """The timeout for the request in seconds."""

    stop: Optional[List[str]] = None
    """The stop sequence for the request."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-onnxruntime-genai"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "do_sample": self.do_sample,
            "max_length": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repeat_penalty,
            "batch_size": self.n_batch,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            **{
                "model_path": self.model_path,
            },
            **self._default_params,
        }

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that python package exists in environment."""

        try:
            from onnxruntime_genai import (
                Config,
                Model,
                Tokenizer,
            )
        except ImportError:
            raise ImportError(
                "Could not import onnxruntime-genai python package. "
                "Please install it with `pip install onnxruntime-genai`."
            )

        try:
            config = Config(self.model_path)
            config.clear_providers()
            if self.execution_provider != ExecutionProviders.CPU:
                config.append_provider(self.execution_provider)
            self.onnx_model = Model(config)
            self.onnx_config = config
        except Exception as e:
            raise ValueError(
                f"Could not load Onnx model from path: {self.model_path}. "
                f"Received error {e}"
            )

        try:
            self.tokenizer = Tokenizer(self.onnx_model)

            # Tokenizer used for the chat template formatting
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            raise ValueError(
                f"Could not load Onnx model from path: {self.model_path}. "
                f"Received error {e}"
            )

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from onnxruntime_genai import Generator, GeneratorParams

        # Encode prompts
        llm_input = self._to_chat_prompt(messages)
        input_token = self.tokenizer.encode(llm_input)

        model_params = self._default_params
        model_params.update(kwargs)

        # Build generator params
        params = GeneratorParams(self.onnx_model)
        params.set_search_options(**model_params)
        generator = Generator(self.onnx_model, params)

        # Append input token
        text_generations: list[str] = []
        generator.append_tokens(input_token)
        answer = ""
        token_count = 0

        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]

            if new_token != self.hf_tokenizer.eos_token_id:
                answer += self.tokenizer.decode(new_token)
                token_count += 1
            if self.max_tokens and token_count >= self.max_tokens:
                break
        text_generations.append(answer)

        # Delete generator to free-up memory
        del generator
        
        generations=[ChatGeneration(AIMessage(content=text)) for text in text_generations]

        return ChatResult(generations=generations)

    def _to_chat_prompt(
        self,
        messages: list[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise TypeError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        prompt = self.hf_tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

        if not isinstance(prompt, str):
            raise TypeError(f"Expected prompt to be a string, but got {type(prompt)}")

        return prompt

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise TypeError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model."""

        from onnxruntime_genai import Generator, GeneratorParams

        # Encode prompts
        llm_input = self._to_chat_prompt(messages)
        input_token = self.tokenizer.encode(llm_input)
        tokenizer_stream = self.tokenizer.create_stream()

        model_params = self._default_params
        model_params.update(kwargs)

        # Build generator params
        params = GeneratorParams(self.onnx_model)
        params.set_search_options(**model_params)
        generator = Generator(self.onnx_model, params)

        # Append input token
        generator.append_tokens(input_token)

        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            generated_token = tokenizer_stream.decode(new_token)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=generated_token))

            if run_manager:
                run_manager.on_llm_new_token(generated_token, chunk=chunk)

            yield chunk
