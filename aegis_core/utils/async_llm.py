# file: utils/async_llm.py

import os
from abc import ABC, abstractmethod
from typing import Dict, Any
import asyncio

# pip install openai google-generativeai anthropic
import openai

try:
    import google.generativeai as genai
except ImportError:
        genai = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

from .utils import retry_on_exception

# Define the specific, retriable exceptions from the OpenAI library
# This is better than catching all Exceptions.
RETRIABLE_OPENAI_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.APIStatusError, # Catches 5xx server errors
)
# --- Define a unified interface ---

class BaseLLM(ABC):
    """An abstract base class for all LLM clients."""
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @retry_on_exception(retries=3, delay=2, backoff=2, exceptions_to_catch=RETRIABLE_OPENAI_EXCEPTIONS)
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """
        Synchronously calls the LLM. This method now automatically retries
        on failure thanks to the decorator.
        """
        pass

# --- Specific implementations for different LLMs ---

class OpenAIClient(BaseLLM):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, **kwargs):
        super().__init__(model_name)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or config.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.reasoning_effort = kwargs.get("reasoning_effort", "none")  # Default to 'none' if not specified

    def call(self, prompt: str, **kwargs) -> str:
        if self.model_name == "gemini-2.5-flash":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=self.reasoning_effort,
                **kwargs
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
        return response.choices[0].message.content

class AzureOpenAIClient(BaseLLM):
    def __init__(self, model_name: str, api_key: str = None, azure_endpoint: str = None, api_version: str = None, **kwargs):
        super().__init__(model_name)
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = api_version or "2024-02-01" # Or get from env/config
        
        if not all([api_key, azure_endpoint]):
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be configured.")
            
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        # In Azure, model_name is the deployment name
        self.deployment_name = model_name

    def call(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    async def acall(self, prompt: str, **kwargs) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return {"response": response.choices[0].message.content}

class GeminiClient(BaseLLM):
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(model_name)
        if genai is None:
            raise ImportError("google-generativeai package is required for GeminiClient. Install with: pip install google-generativeai")
        
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables or config.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def call(self, prompt: str, **kwargs) -> str:
        response = self.model.generate_content(prompt, **kwargs)
        return response.text

    async def acall(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Gemini's async call is generate_content_async
        response = await self.model.generate_content_async(prompt, **kwargs)
        return {"response": response.text}

class ClaudeClient(BaseLLM):
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(model_name)
        if AsyncAnthropic is None:
            raise ImportError("anthropic package is required for ClaudeClient. Install with: pip install anthropic")
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables or config.")
        self.client = AsyncAnthropic(api_key=api_key)

    def call(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048, # Anthropic requires max_tokens
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

    async def acall(self, prompt: str, **kwargs) -> Dict[str, Any]:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=2048, # Anthropic requires max_tokens
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return {"response": response.content[0].text}


# --- Factory function ---

def create_llm_instance(llm_config: Dict[str, Any]) -> BaseLLM:
    """
    Factory function to create an instance of a specific LLM client.
    """
    provider = llm_config.get("provider", "").lower()
    
    # Copy one from the configuration to avoid modifying the original dictionary
    config_params = llm_config.copy()
    
    if provider == "openai":
        return OpenAIClient(**config_params)
    elif provider == "azure":
        return AzureOpenAIClient(**config_params)
    elif provider == "gemini":
        return GeminiClient(**config_params)
    elif provider == "claude":
        return ClaudeClient(**config_params)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")