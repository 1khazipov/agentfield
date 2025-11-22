"""
Agent definition that mirrors the README Quick Start example.

Tests can import `create_agent` to obtain a fully configured Agent
without needing to replicate the agent definition inline.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import requests
from agentfield import AIConfig, Agent

DEFAULT_NODE_ID = "quick-start-agent"


def create_agent(
    ai_config: AIConfig,
    *,
    node_id: str = DEFAULT_NODE_ID,
    callback_url: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
    """
    Build the Quick Start agent with the canonical fetch_url + summarize flow.
    """
    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )

    agent = Agent(
        node_id=node_id,
        ai_config=ai_config,
        **agent_kwargs,
    )

    @agent.skill()
    def fetch_url(url: str) -> str:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text

    @agent.reasoner()
    async def summarize(url: str) -> Dict[str, str]:
        """
        Fetch a URL, summarize it via OpenRouter, and return metadata.
        """
        content = fetch_url(url)
        truncated = content[:2000]

        ai_response = await agent.ai(
            system=(
                "You summarize documentation for internal verification. "
                "Be concise and focus on the site's purpose."
            ),
            user=(
                "Summarize the following web page in no more than two sentences. "
                "Focus on what the site is intended for.\n"
                f"Content:\n{truncated}"
            ),
        )
        summary_text = getattr(ai_response, "text", str(ai_response)).strip()

        return {
            "url": url,
            "summary": summary_text,
            "content_snippet": truncated[:200],
        }

    return agent


def create_agent_from_env() -> Agent:
    """
    Convenience helper to instantiate the agent from environment variables.

    Useful if you want to run this module as a standalone script.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-2.5-flash-lite")

    ai_config = AIConfig(
        model=model,
        api_key=api_key,
        temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0.7")),
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "500")),
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "60.0")),
        retry_attempts=int(os.environ.get("OPENROUTER_RETRIES", "2")),
    )
    return create_agent(ai_config)


if __name__ == "__main__":
    # Allow developers to run: `python -m agents.quick_start_agent`
    agent = create_agent_from_env()
    agent.run()
