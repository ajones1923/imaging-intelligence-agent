"""LLM NIM client with Llama-3 NIM primary and Anthropic Claude fallback.

Provides text generation for clinical reasoning, report synthesis,
and multi-turn conversation with automatic fallback from local
NIM Llama-3 to the Anthropic Claude API.
"""

from typing import Any, Dict, Generator, List, Optional

from loguru import logger

from .base import BaseNIMClient


class LlamaLLMClient(BaseNIMClient):
    """Client for LLM inference via NIM (Llama-3) with Anthropic fallback.

    Priority order:
      1. Local NIM Llama-3 endpoint (OpenAI-compatible API)
      2. Anthropic Claude API (if anthropic_api_key provided)
      3. Mock response (if mock_enabled)

    Uses the OpenAI Python client for NIM communication since NVIDIA
    NIM exposes an OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        base_url: str,
        mock_enabled: bool = True,
        anthropic_api_key: Optional[str] = None,
    ):
        super().__init__(base_url, service_name="llm", mock_enabled=mock_enabled)
        self.anthropic_api_key = anthropic_api_key
        self._openai_client = None
        self._anthropic_client = None

    def health_check(self) -> bool:
        """Check NIM LLM availability via /v1/models endpoint (OpenAI-compat)."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _get_openai_client(self):
        """Lazy-initialize the OpenAI client for NIM."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    base_url=f"{self.base_url}/v1",
                    api_key="not-needed",  # NIM does not require an API key
                )
            except ImportError:
                logger.warning("openai package not installed; NIM LLM unavailable")
                return None
        return self._openai_client

    def _get_anthropic_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._anthropic_client is None and self.anthropic_api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.anthropic_api_key,
                )
            except ImportError:
                logger.warning(
                    "anthropic package not installed; Claude fallback unavailable"
                )
                return None
        return self._anthropic_client

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a text response from the LLM.

        Attempts NIM Llama-3 first, then falls back to Anthropic Claude,
        then to mock if all else fails.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                Example: [{"role": "user", "content": "Describe this finding."}]
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum response tokens.

        Returns:
            Generated text response string.
        """
        # Attempt 1: NIM Llama-3
        if self.is_available():
            try:
                client = self._get_openai_client()
                if client is not None:
                    response = client.chat.completions.create(
                        model="meta/llama3-70b-instruct",
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = response.choices[0].message.content
                    logger.info(
                        f"NIM Llama-3 generated {len(text)} chars "
                        f"({response.usage.total_tokens} tokens)"
                    )
                    return text
            except Exception as e:
                logger.error(f"NIM LLM generation failed: {e}")

        # Attempt 2: Anthropic Claude
        if self.anthropic_api_key:
            try:
                client = self._get_anthropic_client()
                if client is not None:
                    logger.info("Falling back to Anthropic Claude API")

                    # Convert messages: extract system message if present
                    system_msg = ""
                    chat_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            system_msg = msg["content"]
                        else:
                            chat_messages.append(msg)

                    # Ensure at least one user message
                    if not chat_messages:
                        chat_messages = [{"role": "user", "content": "Hello."}]

                    kwargs: Dict[str, Any] = {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": max_tokens,
                        "messages": chat_messages,
                    }
                    if system_msg:
                        kwargs["system"] = system_msg

                    response = client.messages.create(**kwargs)
                    text = response.content[0].text
                    logger.info(
                        f"Claude generated {len(text)} chars "
                        f"(input={response.usage.input_tokens}, "
                        f"output={response.usage.output_tokens})"
                    )
                    return text
            except Exception as e:
                logger.error(f"Anthropic Claude generation failed: {e}")

        # Attempt 3: Mock
        if self.mock_enabled:
            logger.info("Using mock LLM response (no NIM or Claude available)")
            return self._mock_response(messages=messages)

        raise ConnectionError(
            "LLM unavailable: NIM unreachable, no Anthropic API key, and mock disabled"
        )

    def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Stream a text response from the LLM token by token.

        Attempts NIM Llama-3 streaming first, then Anthropic Claude
        streaming, then yields mock response in chunks.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Sampling temperature.

        Yields:
            Text chunks as they are generated.
        """
        # Attempt 1: NIM Llama-3 streaming
        if self.is_available():
            try:
                client = self._get_openai_client()
                if client is not None:
                    stream = client.chat.completions.create(
                        model="meta/llama3-70b-instruct",
                        messages=messages,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
            except Exception as e:
                logger.error(f"NIM LLM streaming failed: {e}")

        # Attempt 2: Anthropic Claude streaming
        if self.anthropic_api_key:
            try:
                client = self._get_anthropic_client()
                if client is not None:
                    logger.info("Streaming via Anthropic Claude API")

                    system_msg = ""
                    chat_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            system_msg = msg["content"]
                        else:
                            chat_messages.append(msg)

                    if not chat_messages:
                        chat_messages = [{"role": "user", "content": "Hello."}]

                    kwargs: Dict[str, Any] = {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 2048,
                        "messages": chat_messages,
                    }
                    if system_msg:
                        kwargs["system"] = system_msg

                    with client.messages.stream(**kwargs) as stream:
                        for text in stream.text_stream:
                            yield text
                    return
            except Exception as e:
                logger.error(f"Anthropic Claude streaming failed: {e}")

        # Attempt 3: Mock streaming
        if self.mock_enabled:
            logger.info("Using mock LLM streaming (no NIM or Claude available)")
            mock_text = self._mock_response(messages=messages)
            # Yield word by word to simulate streaming
            words = mock_text.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
            return

        raise ConnectionError(
            "LLM unavailable: NIM unreachable, no Anthropic API key, and mock disabled"
        )

    def _mock_response(self, **kwargs) -> str:
        """Return a template clinical response string.

        Provides a plausible clinical AI assistant response for
        demonstration and testing purposes.
        """
        messages = kwargs.get("messages", [])

        # Extract last user message for context
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        return (
            "Based on the available clinical information and imaging data, "
            "here is my analysis:\n\n"
            "**Assessment:**\n"
            "The imaging findings are consistent with normal anatomical "
            "structures without evidence of acute pathology. The visualized "
            "organs demonstrate expected morphology and attenuation patterns.\n\n"
            "**Key Observations:**\n"
            "1. No focal lesions or mass effects identified\n"
            "2. Vascular structures appear patent and normal in caliber\n"
            "3. No evidence of lymphadenopathy\n"
            "4. Osseous structures are intact\n\n"
            "**Recommendations:**\n"
            "- Clinical correlation is recommended\n"
            "- Follow-up imaging per standard protocol if clinically indicated\n"
            "- Integration with laboratory findings may provide additional "
            "diagnostic clarity\n\n"
            "*Note: This is an AI-generated analysis for clinical decision "
            "support. All findings should be verified by a qualified radiologist.*"
        )
