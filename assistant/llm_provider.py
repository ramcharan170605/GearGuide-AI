"""
LLM Provider Abstraction Layer for GearGuide-AI

Implements a unified interface for multiple LLM providers (Gemini, OpenAI, NVIDIA NIM)
with automatic fallback, robust error handling, rate-limit aware retries, and environment-based configuration.

Priority: 1. Google GenAI (new SDK) 2. OpenAI 3. NVIDIA NIM (OpenAI-compatible)
"""

import os
import time
import re
from abc import ABC, abstractmethod
from typing import Optional, Any, Union
from dataclasses import dataclass
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Resolve absolute path to .env file relative to this file
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    _env_path = os.path.join(_project_root, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
    else:
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

    def __post_init__(self):
        # Apply environment overrides if values are defaults
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider and self.provider == "gemini":
            self.provider = env_provider

        # Store separate Gemini, OpenAI and NVIDIA models
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.nvidia_model = os.getenv("NVIDIA_MODEL", "mistral-medium-3.5-128b")

        # Resolve the active model based on provider
        if self.model == "gemini-2.5-flash":
            if self.provider == "openai":
                self.model = self.openai_model
            elif self.provider == "nvidia":
                self.model = self.nvidia_model
            else:
                self.model = self.gemini_model

        # Load numerical configs from environment if default
        if self.temperature == 0.0:
            temp_env = os.getenv("LLM_TEMPERATURE")
            if temp_env is not None:
                try:
                    self.temperature = float(temp_env)
                except ValueError:
                    pass

        if self.max_tokens == 4096:
            max_tokens_env = os.getenv("LLM_MAX_TOKENS")
            if max_tokens_env is not None:
                try:
                    self.max_tokens = int(max_tokens_env)
                except ValueError:
                    pass

        if self.timeout == 60:
            timeout_env = os.getenv("LLM_TIMEOUT")
            if timeout_env is not None:
                try:
                    self.timeout = int(timeout_env)
                except ValueError:
                    pass


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

        # Resolve model name
        model = self.config.model
        if not model or "gemini" not in model.lower():
            model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

        # List of models to try
        models_to_try = [model]
        if model != "gemini-2.5-flash":
            models_to_try.append("gemini-2.5-flash")

        last_exception = None
        for m in models_to_try:
            max_retries = 3
            backoff = 1.0
            for attempt in range(max_retries):
                try:
                    response = self._client.models.generate_content(
                        model=m,
                        contents=[full_prompt]
                    )
                    return LLMReturn(
                        content=response.text,
                        usage={"input_tokens": 0, "output_tokens": 0},
                        provider="gemini"
                    )
                except Exception as e:
                    last_exception = e
                    err_msg = str(e).lower()
                    
                    # Check if transient error (503, 429, etc.)
                    is_retryable = any(x in err_msg for x in ["503", "429", "unavailable", "exhausted", "limit", "rate"])
                    
                    if is_retryable and attempt < max_retries - 1:
                        # Parse suggested retry delay if available in error message
                        delay = backoff
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                        if match:
                            try:
                                delay = float(match.group(1)) + 1.0
                            except ValueError:
                                pass
                        else:
                            match2 = re.search(r"retry in (\d+) seconds", err_msg)
                            if match2:
                                try:
                                    delay = float(match2.group(1)) + 1.0
                                except ValueError:
                                    pass
                        
                        print(f"Gemini call for {m} failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        backoff *= 2.0
                    else:
                        break  # Break to next model or propagate error

        raise RuntimeError(f"Gemini API error: {str(last_exception)}")


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

        # Resolve model name
        model = self.config.model
        if not model or "gemini" in model.lower() or not any(x in model.lower() for x in ["gpt", "o1", "o3"]):
            model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        # List of models to try
        models_to_try = [model]
        if model != "gpt-4o-mini":
            models_to_try.append("gpt-4o-mini")

        last_exception = None
        for m in models_to_try:
            max_retries = 3
            backoff = 1.0
            for attempt in range(max_retries):
                try:
                    response = self._client.chat.completions.create(
                        model=m,
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
                    last_exception = e
                    err_msg = str(e).lower()
                    
                    # Check if transient error (429, 503, etc.)
                    is_retryable = any(x in err_msg for x in ["429", "503", "rate", "quota", "overloaded", "unavailable"])
                    
                    if is_retryable and attempt < max_retries - 1:
                        # Parse suggested retry delay if available in error message
                        delay = backoff
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                        if match:
                            try:
                                delay = float(match.group(1)) + 1.0
                            except ValueError:
                                pass
                        else:
                            match2 = re.search(r"retry in (\d+) seconds", err_msg)
                            if match2:
                                try:
                                    delay = float(match2.group(1)) + 1.0
                                except ValueError:
                                    pass
                        
                        print(f"OpenAI call for {m} failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        backoff *= 2.0
                    else:
                        break  # Break to next model or propagate error

        raise RuntimeError(f"OpenAI API error: {str(last_exception)}")


class NvidiaNimProvider(BaseLLMProvider):
    """NVIDIA NIM LLM provider (OpenAI-compatible)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the NVIDIA NIM client using OpenAI SDK."""
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            self._client = None
            return

        try:
            import openai
            self._client = openai.Client(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1"
            )
        except ImportError:
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and os.getenv("NVIDIA_API_KEY") is not None

    def get_name(self) -> str:
        return "nvidia"

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        if not self.is_available():
            raise RuntimeError("NVIDIA NIM provider not available. Check NVIDIA_API_KEY.")

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({"role": msg.role, "content": msg.content})
        # Resolve model name
        model = self.config.model
        # If the model is a gemini model or an openai model, or not specified, use NVIDIA_MODEL
        if not model or "gemini" in model.lower() or any(x in model.lower() for x in ["gpt", "o1", "o3"]):
            model = os.getenv("NVIDIA_MODEL") or "mistral-medium-3.5-128b"

        # Format model names to have the correct organization prefix (mistralai/) if missing
        def format_nvidia_model(m: str) -> str:
            if "/" not in m and ("mistral" in m.lower() or "ministral" in m.lower()):
                return f"mistralai/{m}"
            return m

        model_formatted = format_nvidia_model(model)
        default_formatted = format_nvidia_model("mistral-medium-3.5-128b")

        # List of models to try
        models_to_try = [model_formatted]
        if model_formatted != default_formatted:
            models_to_try.append(default_formatted)

        last_exception = None
        for m in models_to_try:
            max_retries = 3
            backoff = 1.0
            for attempt in range(max_retries):
                try:
                    response = self._client.chat.completions.create(
                        model=m,
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
                        provider="nvidia"
                    )
                except Exception as e:
                    last_exception = e
                    err_msg = str(e).lower()
                    
                    # Check if transient error (429, 503, etc.)
                    is_retryable = any(x in err_msg for x in ["429", "503", "rate", "quota", "overloaded", "unavailable"])
                    
                    if is_retryable and attempt < max_retries - 1:
                        delay = backoff
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                        if match:
                            try:
                                delay = float(match.group(1)) + 1.0
                            except ValueError:
                                pass
                        else:
                            match2 = re.search(r"retry in (\d+) seconds", err_msg)
                            if match2:
                                try:
                                    delay = float(match2.group(1)) + 1.0
                                except ValueError:
                                    pass
                        
                        print(f"NVIDIA NIM call for {m} failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        backoff *= 2.0
                    else:
                        break  # Break to next model or propagate error

        raise RuntimeError(f"NVIDIA NIM API error: {str(last_exception)}")


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

        # Try NVIDIA NIM (priority 3)
        try:
            nvidia = NvidiaNimProvider(self.config)
            if nvidia.is_available():
                self._providers["nvidia"] = nvidia
        except Exception:
            pass

    def get_active_provider(self) -> Optional[BaseLLMProvider]:
        """Get the active provider based on priority."""
        if "gemini" in self._providers:
            return self._providers["gemini"]
        elif "openai" in self._providers:
            return self._providers["openai"]
        elif "nvidia" in self._providers:
            return self._providers["nvidia"]
        return None

    def chat(self, messages: list[Message], tools: Optional[list[dict]] = None) -> LLMReturn:
        """Send a chat completion request with fallback if the active provider fails."""
        providers_to_try = []
        if "gemini" in self._providers:
            providers_to_try.append(self._providers["gemini"])
        if "openai" in self._providers:
            providers_to_try.append(self._providers["openai"])
        if "nvidia" in self._providers:
            providers_to_try.append(self._providers["nvidia"])

        if not providers_to_try:
            raise RuntimeError(
                "No LLM provider available. "
                "Please set GEMINI_API_KEY, OPENAI_API_KEY, or NVIDIA_API_KEY environment variable."
            )

        import logging
        logger = logging.getLogger("assistant.llm_provider")

        last_error = None
        for provider in providers_to_try:
            try:
                return provider.chat(messages, tools)
            except Exception as e:
                last_error = e
                logger.error(f"Provider {provider.get_name()} failed: {e}", exc_info=True)
                print(f"Provider {provider.get_name()} failed: {e}. Falling back to next available provider...")

        err_msg = str(last_error).lower()
        if "429" in err_msg or "exhausted" in err_msg or "rate" in err_msg or "quota" in err_msg:
            # Check if there is a sleep recommendation in the error
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
            seconds_msg = f" in {match.group(1)}s" if match else " shortly"
            
            match2 = re.search(r"retry in (\d+) seconds", err_msg)
            if match2:
                seconds_msg = f" in {match2.group(1)}s"
                
            friendly_msg = f"LLM rate or daily quota limit exceeded. Please retry{seconds_msg}."
        elif "503" in err_msg or "unavailable" in err_msg:
            friendly_msg = "LLM service is temporarily overloaded or unavailable. Please retry in a few seconds."
        else:
            friendly_msg = f"LLM Error: {str(last_error)}"

        raise RuntimeError(friendly_msg)

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
