"""
Thin wrapper around the OpenAI API for querying LLMs.

Reads the API key from the OPENAI_API_KEY environment variable.
"""

import os
import time


def _get_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def query_llm(messages, model="gpt-4o", max_completion_tokens=400,
              temperature=0.0, retry_wait=5, max_retries=5):
    """
    Send a chat completion request to the OpenAI API with retry logic.

    Args:
        messages: list of message dicts (role + content).
        model: model name string (e.g. 'gpt-4o', 'gpt-4o-mini').
        max_completion_tokens: maximum tokens in the generated output.
        temperature: sampling temperature.
        retry_wait: base seconds to wait on rate-limit errors.
        max_retries: maximum number of retry attempts.

    Returns:
        str: the assistant's response content.
    """
    from openai import AuthenticationError, RateLimitError

    client = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                messages=messages,
            )
            return response.choices[0].message.content
        except AuthenticationError:
            raise
        except RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise RuntimeError(
                    "OpenAI quota exceeded. Add credits at "
                    "https://platform.openai.com/account/billing"
                ) from e
            wait = retry_wait * (2 ** attempt)
            print(f"Rate limited (attempt {attempt + 1}/{max_retries}), "
                  f"retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            wait = retry_wait * (2 ** attempt)
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries")
