"""
LLM Provider Abstraction Layer for GearGuide-AI

Implements a unified interface for multiple LLM providers (Gemini, OpenAI)
with automatic fallback and environment-based configuration.

Priority: 1. Google GenAI (new SDK) 2. OpenAI
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
    pass


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
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
        pass

    @abstractmethod
    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class GeminiProvider(BaseLLMProvider):
    """Google GenAI LLM provider (new SDK)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._model = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the GenAI client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self._client = None
            return

        try:
            import google.genai as genai
            self._client = genai.Client(api_key=api_key)
            # Model will be resolved at chat time
        except ImportError:
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and os.getenv("GEMINI_API_KEY") is not None

    def get_name(self) -> str:
        return "gemini"

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        if not self.is_available():
            raise RuntimeError("Gemini provider not available. Check GEMINI_API_KEY.")

        import google.genai as genai

        # Extract user message and system context
        last_user_message = None
        system_context = []
        for msg in messages:
            if msg.role == "user":
                last_user_message = msg.content
            elif msg.role == "system":
                system_context.append(msg.content)

        if not last_user_message:
            raise RuntimeError("No user message found in messages")

        # Combine system context
        context_str = "\n".join(system_context)
        full_prompt = f"{context_str}\n\nUser: {last_user_message}" if context_str else last_user_message

        try:
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=[full_prompt]
            )
            return LLMReturn(
                content=response.text,
                usage={"input_tokens": 0, "output_tokens": 0},
                provider="gemini"
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._client = None
            return

        try:
            import openai
            self._client = openai.Client(api_key=api_key)
        except ImportError:
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and os.getenv("OPENAI_API_KEY") is not None

    def get_name(self) -> str:
        return "openai"

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available. Check OPENAI_API_KEY.")

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return LLMReturn(
                content=response.choices[0].message.content or "",
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                provider="openai"
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class LLMProviderManager:
    """Manages multiple LLM providers with fallback logic."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._providers: dict[str, BaseLLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers."""
        # Try GenAI first (priority 1)
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
        """Get the active provider based on priority."""
        if "gemini" in self._providers:
            return self._providers["gemini"]
        elif "openai" in self._providers:
            return self._providers["openai"]
        return None

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        """Send a chat completion request using the active provider."""
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


# Global provider manager instance
_provider_manager: Optional[LLMProviderManager] = None


def get_llm_provider(config: Optional[LLMConfig] = None) -> LLMProviderManager:
    """Get the global LLM provider manager."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = LLMProviderManager(config)
    return _provider_manager


def reset_llm_provider():
    """Reset the global LLM provider manager."""
    global _provider_manager
    _provider_manager = None
