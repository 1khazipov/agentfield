"""
Functional test: Quick Start workflow from README

This test mirrors the Quick Start documentation by:
1. Loading a standalone Python agent definition (agents/quick_start_agent.py)
2. Registering the agent with the AgentField control plane
3. Executing the summarize reasoner via the control plane API
4. Verifying the agent can read content and summarize it with OpenRouter
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

import pytest

from agents.quick_start_agent import create_agent
from utils.agent_server import run_agent_server


QUICK_START_URL = os.environ.get("TEST_QUICK_START_URL")

EXAMPLE_DOMAIN_HTML = """<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
    <div>
        <h1>Example Domain</h1>
        <p>This domain is for use in illustrative examples in documents. You may use this
        domain in literature without prior coordination or asking for permission.</p>
        <p><a href="https://www.iana.org/domains/example">More information...</a></p>
    </div>
</body>
</html>
"""


def _start_example_domain_server() -> Tuple[ThreadingHTTPServer, threading.Thread, str]:
    """
    Spin up a lightweight HTTP server that serves the Example Domain HTML used in docs.
    """

    class ExampleDomainHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(EXAMPLE_DOMAIN_HTML.encode("utf-8"))

        def log_message(self, *_):
            # Silence default logging noise from BaseHTTPRequestHandler
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ExampleDomainHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


@pytest.mark.functional
@pytest.mark.openrouter
@pytest.mark.asyncio
async def test_quick_start_documentation_flow(
    openrouter_config,
    async_http_client,
):
    """
    Validate the README Quick Start instructions end-to-end.

    This spins up the canonical agent from the docs (fetch_url + summarize),
    registers it with the control plane, runs a live summary request against
    https://example.com, and ensures the response structure matches expectations.
    """
    content_server: Optional[ThreadingHTTPServer] = None
    content_thread: Optional[threading.Thread] = None

    # Determine which URL to summarize. Default to local Example Domain server
    # to avoid relying on outbound internet access, but allow overriding via env.
    if QUICK_START_URL:
        target_url = QUICK_START_URL
    else:
        content_server, content_thread, target_url = _start_example_domain_server()

    agent = create_agent(openrouter_config)

    async with run_agent_server(agent):
        nodes_response = await async_http_client.get(f"/api/v1/nodes/{agent.node_id}")
        assert nodes_response.status_code == 200, nodes_response.text

        node_data = nodes_response.json()
        assert node_data["id"] == agent.node_id
        assert "summarize" in [r["id"] for r in node_data.get("reasoners", [])]

        execution_request = {"input": {"url": target_url}}

        execution_response = await async_http_client.post(
            f"/api/v1/reasoners/{agent.node_id}.summarize",
            json=execution_request,
            timeout=90.0,
        )

        assert execution_response.status_code == 200, execution_response.text
        result_data = execution_response.json()

        assert "result" in result_data
        result = result_data["result"]

        assert result["url"] == target_url
        summary_text = result["summary"]
        assert summary_text, "Summary should not be empty"
        assert len(summary_text.split()) >= 5, "Summary should contain multiple words"

        snippet = result.get("content_snippet", "")
        assert "Example Domain" in snippet, "Snippet should contain fetched page content"
        assert len(snippet) > 0

        assert result_data["node_id"] == agent.node_id
        assert result_data["duration_ms"] > 0

        headers = execution_response.headers
        assert "X-Workflow-ID" in headers or "x-workflow-id" in headers
        assert "X-Execution-ID" in headers or "x-execution-id" in headers

    if content_server:
        content_server.shutdown()
        if content_thread:
            content_thread.join(timeout=5)
