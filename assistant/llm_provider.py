"""
LLM Provider Abstraction Layer for GearGuide-AI

Implements a unified interface for multiple LLM providers (Gemini, OpenAI)
with automatic fallback and environment-based configuration.

Requirements: LLM-FIRST architecture, provider abstraction
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Union
from dataclasses import dataclass
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use os.environ directly


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "gemini"  # Default to Gemini as per assignment
    model: str = "gemini-pro"  # Default model
    temperature: float = 0.0  # Low temperature for deterministic behavior
    max_tokens: int = 4096
    timeout: int = 60


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMReturn:
    """Return value from LLM call."""
    content: str
    usage: dict[str, int]
    provider: str


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def __init__(self, config: LLMConfig):
        """Initialize the provider with configuration."""
        pass

    @abstractmethod
    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        """
        Send a chat completion request.

        Args:
            messages: List of chat messages
            tools: Optional list of tool descriptions for function calling

        Returns:
            LLMReturn with content, usage, and provider info
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API key present, etc.)."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the provider name."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Lazily initialize the Gemini client."""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self._client = genai
            else:
                self._client = None
        except ImportError:
            self._client = None

    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self._client is not None and os.getenv("GEMINI_API_KEY") is not None

    def get_name(self) -> str:
        """Get provider name."""
        return "gemini"

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        """
        Send a chat completion request to Gemini.

        Note: Gemini function calling uses a different format than OpenAI.
        For now, we'll use text-based tool calling for compatibility.
        """
        if not self.is_available():
            raise RuntimeError("Gemini provider not available. Check GEMINI_API_KEY.")

        import google.generativeai as genai

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg.role
            content = msg.content

            # Handle system messages (Gemini uses "user" for system in some cases)
            if role == "system":
                gemini_messages.append({"role": "user", "parts": [{"text": f"SYSTEM: {content}"}]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})

        # Get the model
        model = self._client.GenerativeModel(
            model_name=self.config.model,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
        )

        # Start chat session
        chat_session = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

        # Send the last message
        last_message = gemini_messages[-1]
        response = chat_session.send_message(
            content=last_message["parts"][0]["text"] if isinstance(last_message, dict) else str(last_message)
        )

        return LLMReturn(
            content=response.text,
            usage={
                "input_tokens": response.usage_metadata.get("prompt_token_count", 0),
                "output_tokens": response.usage_metadata.get("candidates_token_count", 0),
            },
            provider=self.get_name()
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Lazily initialize the OpenAI client."""
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._client = openai.Client(api_key=api_key)
            else:
                self._client = None
        except ImportError:
            self._client = None

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self._client is not None and os.getenv("OPENAI_API_KEY") is not None

    def get_name(self) -> str:
        """Get provider name."""
        return "openai"

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        """
        Send a chat completion request to OpenAI.
        """
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available. Check OPENAI_API_KEY.")

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        # Add tools if provided
        kwargs = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)

            # Handle tool calls if present
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                # For now, return the text content
                content = response.choices[0].message.content or ""
                return LLMReturn(
                    content=content,
                    usage={
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    },
                    provider=self.get_name()
                )
            else:
                return LLMReturn(
                    content=response.choices[0].message.content or "",
                    usage={
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    },
                    provider=self.get_name()
                )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class LLMProviderManager:
    """
    Manages multiple LLM providers with fallback logic.

    Priority: 1. Gemini 2. OpenAI
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._providers: dict[str, BaseLLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers."""
        # Try Gemini first (priority 1)
        try:
            gemini = GeminiProvider(self.config)
            if gemini.is_available():
                self._providers["gemini"] = gemini
        except Exception:
            pass

        # Try OpenAI (priority 2)
        try:
            openai = OpenAIProvider(self.config)
            if openai.is_available():
                self._providers["openai"] = openai
        except Exception:
            pass

    def get_active_provider(self) -> Optional[BaseLLMProvider]:
        """
        Get the active provider based on priority.

        Priority: 1. Gemini 2. OpenAI
        """
        # Prefer Gemini as per assignment recommendation
        if "gemini" in self._providers:
            return self._providers["gemini"]
        elif "openai" in self._providers:
            return self._providers["openai"]
        return None

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        """
        Send a chat completion request using the active provider.

        Raises:
            RuntimeError: If no provider is available
        """
        provider = self.get_active_provider()
        if provider is None:
            raise RuntimeError(
                "No LLM provider available. "
                "Please set GEMINI_API_KEY or OPENAI_API_KEY environment variable."
            )
        return provider.chat(messages, tools)

    def list_providers(self) -> list[str]:
        """List available providers."""
        return list(self._providers.keys())

    def is_available(self) -> bool:
        """Check if any provider is available."""
        return len(self._providers) > 0


# Global provider manager instance (lazy initialization)
_provider_manager: Optional[LLMProviderManager] = None


def get_llm_provider(config: Optional[LLMConfig] = None) -> LLMProviderManager:
    """
    Get the global LLM provider manager.

    Args:
        config: Optional configuration (uses defaults if not provided)

    Returns:
        LLMProviderManager instance
    """
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = LLMProviderManager(config)
    return _provider_manager


def reset_llm_provider():
    """Reset the global LLM provider manager (useful for testing)."""
    global _provider_manager
    _provider_manager = None


if __name__ == "__main__":
    # Test the provider
    print("Testing LLM Provider Abstraction...")

    manager = get_llm_provider()
    print(f"Available providers: {manager.list_providers()}")

    if manager.is_available():
        try:
            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello, what is your name?")
            ]
            result = manager.chat(messages)
            print(f"Response from {result.provider}: {result.content[:100]}...")
            print(f"Usage: {result.usage}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No LLM provider available. Please set API keys.")
