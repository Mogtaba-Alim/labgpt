"""
LLM Client Wrapper

Provides a unified interface that translates between different LLM API formats:
- Anthropic Claude (messages.create with system parameter)
- OpenAI (chat.completions.create)
- Local Llama (Hugging Face Transformers)

This allows all task generators to work seamlessly with any backend without modification.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class MessageContent:
    """Represents the content part of a message."""
    role: str
    content: str


@dataclass
class CompletionResponse:
    """Unified response format across all backends."""
    content: str
    model: str = "local-llama-3.1-8b"
    finish_reason: str = "stop"


class LLMClientWrapper:
    """
    Unified wrapper for LLM clients.

    Provides consistent API regardless of backend:
    - Anthropic Claude API
    - OpenAI API
    - Local Llama model

    Usage:
        # For Claude:
        response = wrapper.messages.create(
            model="claude-sonnet-4",
            system="System prompt",
            messages=[{"role": "user", "content": "Question"}],
            max_tokens=1000
        )

        # For OpenAI:
        response = wrapper.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Question"}
            ],
            max_tokens=800
        )

    Both work seamlessly with local Llama backend!
    """

    def __init__(self, backend_client):
        """
        Initialize wrapper with a backend client.

        Args:
            backend_client: Either LocalLlamaClient, Anthropic client, or OpenAI client
        """
        self.backend_client = backend_client
        self.backend_type = self._detect_backend_type()

        logger.info(f"LLMClientWrapper initialized with backend: {self.backend_type}")

        # Create nested objects for API compatibility
        self.messages = self._MessagesAPI(self)
        self.chat = self._ChatAPI(self)

    def _detect_backend_type(self) -> str:
        """Detect which type of backend we're wrapping."""
        client_type = type(self.backend_client).__name__

        if "LocalLlama" in client_type:
            return "local_llama"
        elif "Anthropic" in client_type:
            return "anthropic"
        elif "OpenAI" in client_type:
            return "openai"
        else:
            logger.warning(f"Unknown backend type: {client_type}, assuming local_llama")
            return "local_llama"

    class _MessagesAPI:
        """Nested class to mimic Anthropic's messages.create() API."""

        def __init__(self, wrapper):
            self.wrapper = wrapper

        def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            system: Optional[str] = None,
            max_tokens: int = 1000,
            temperature: float = 0.7,
            **kwargs
        ):
            """
            Create a message completion (Anthropic-style API).

            Args:
                model: Model identifier (ignored for local model)
                messages: List of message dicts with 'role' and 'content'
                system: System prompt (Anthropic-specific parameter)
                max_tokens: Maximum tokens to generate
                temperature: Sampling temperature
                **kwargs: Additional parameters (ignored)

            Returns:
                Response object with content attribute
            """
            # Convert to unified format
            unified_messages = []

            # Add system message if provided
            if system:
                unified_messages.append({"role": "system", "content": system})

            # Add user/assistant messages
            unified_messages.extend(messages)

            # Call backend
            if self.wrapper.backend_type == "local_llama":
                # Use local Llama
                content = self.wrapper.backend_client.generate(
                    messages=unified_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Return response in Anthropic format
                return self._create_anthropic_response(content)

            elif self.wrapper.backend_type == "anthropic":
                # Pass through to actual Anthropic client
                return self.wrapper.backend_client.messages.create(
                    model=model,
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )

            else:
                raise ValueError(f"Unsupported backend for messages.create: {self.wrapper.backend_type}")

        def _create_anthropic_response(self, content: str):
            """Create a mock Anthropic response object."""
            class MockResponse:
                def __init__(self, text):
                    self.content = [type('obj', (object,), {'text': text})]
                    self.model = "local-llama-3.1-8b"
                    self.stop_reason = "end_turn"

            return MockResponse(content)

    class _ChatAPI:
        """Nested class to mimic OpenAI's chat.completions.create() API."""

        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.completions = self

        def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            max_tokens: int = 800,
            temperature: float = 0.2,
            **kwargs
        ):
            """
            Create a chat completion (OpenAI-style API).

            Args:
                model: Model identifier (ignored for local model)
                messages: List of message dicts with 'role' and 'content'
                max_tokens: Maximum tokens to generate
                temperature: Sampling temperature
                **kwargs: Additional parameters (ignored)

            Returns:
                Response object with choices[0].message.content
            """
            # Call backend
            if self.wrapper.backend_type == "local_llama":
                # Use local Llama
                content = self.wrapper.backend_client.generate(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Return response in OpenAI format
                return self._create_openai_response(content)

            elif self.wrapper.backend_type == "openai":
                # Pass through to actual OpenAI client
                return self.wrapper.backend_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )

            else:
                raise ValueError(f"Unsupported backend for chat.completions.create: {self.wrapper.backend_type}")

        def _create_openai_response(self, content: str):
            """Create a mock OpenAI response object."""
            class MockMessage:
                def __init__(self, text):
                    self.content = text
                    self.role = "assistant"

            class MockChoice:
                def __init__(self, text):
                    self.message = MockMessage(text)
                    self.finish_reason = "stop"
                    self.index = 0

            class MockResponse:
                def __init__(self, text):
                    self.choices = [MockChoice(text)]
                    self.model = "local-llama-3.1-8b"
                    self.created = 0
                    self.id = "local-completion"

            return MockResponse(content)


def create_privacy_mode_client(
    model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    load_in_8bit: bool = True,
    device: Optional[str] = None,
    hf_token: Optional[str] = None
) -> tuple:
    """
    Create a pair of wrapped clients for privacy mode.

    This returns two wrapped clients (for compatibility with code that expects
    separate Anthropic and OpenAI clients), but both use the same local model.

    Args:
        model_path: Hugging Face model ID or local path
        load_in_8bit: Whether to use 8-bit quantization
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        hf_token: Hugging Face token for gated models

    Returns:
        Tuple of (claude_client, openai_client) - both are wrapped local clients
    """
    from .local_llm_client import LocalLlamaClient

    logger.info("Initializing privacy mode with local Llama 3.1 8B")

    # Create single local client
    local_client = LocalLlamaClient(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
        device=device,
        hf_token=hf_token
    )

    # Wrap it for both API formats
    claude_compatible = LLMClientWrapper(local_client)
    openai_compatible = LLMClientWrapper(local_client)

    logger.info("Privacy mode clients created successfully")
    logger.info("All API calls will use local model - no data sent to external services")

    return claude_compatible, openai_compatible
