"""
Pytest configuration and fixtures for AgentField functional tests.

These fixtures provide integration with the Docker-based test environment,
allowing tests to interact with the control plane and create test agents.
"""

import asyncio
import os
import time
from typing import AsyncGenerator, Callable, Dict, Generator, Optional
import uuid

import httpx
import pytest
from agentfield import Agent, AIConfig

pytest_plugins = ("pytest_asyncio",)

AGENT_BIND_HOST = os.environ.get("TEST_AGENT_BIND_HOST", "127.0.0.1")
AGENT_CALLBACK_HOST = os.environ.get("TEST_AGENT_CALLBACK_HOST", "127.0.0.1")


# ============================================================================
# Environment and Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def control_plane_url() -> str:
    """Get the AgentField control plane URL from environment."""
    url = os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def openrouter_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="session")
def openrouter_model() -> str:
    """
    Get the OpenRouter model to use for tests from environment.
    
    IMPORTANT: All tests MUST use this fixture and NOT hardcode model names.
    This allows us to use cost-effective models for testing.
    """
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-2.5-flash-lite")
    return model


@pytest.fixture(scope="session")
def storage_mode() -> str:
    """Get the current storage mode being tested."""
    return os.environ.get("STORAGE_MODE", "local")


@pytest.fixture(scope="session")
def test_timeout() -> int:
    """Get the test timeout in seconds."""
    return int(os.environ.get("TEST_TIMEOUT", "300"))


# ============================================================================
# Control Plane Health Check
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def verify_control_plane(control_plane_url: str):
    """Verify that the control plane is accessible before running tests."""
    health_url = f"{control_plane_url}/api/v1/health"
    max_attempts = 30
    
    print(f"\nVerifying control plane at {control_plane_url}...")
    
    for attempt in range(max_attempts):
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                print(f"✓ Control plane is healthy (attempt {attempt + 1})")
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        
        if attempt < max_attempts - 1:
            time.sleep(1)
    
    pytest.fail(f"Control plane at {control_plane_url} is not responding to health checks")


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
async def async_http_client(control_plane_url: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an async HTTP client configured for the control plane."""
    async with httpx.AsyncClient(
        base_url=control_plane_url,
        timeout=30.0,
        follow_redirects=True
    ) as client:
        yield client


# ============================================================================
# AI Configuration Fixtures
# ============================================================================

@pytest.fixture
def openrouter_config(openrouter_api_key: str, openrouter_model: str) -> AIConfig:
    """
    Provide an AIConfig configured for OpenRouter.
    
    IMPORTANT: Uses the OPENROUTER_MODEL environment variable.
    Default model is cost-effective for testing (gemini-2.5-flash-lite).
    DO NOT hardcode model names in tests - always use this fixture.
    """
    return AIConfig(
        model=openrouter_model,
        api_key=openrouter_api_key,
        temperature=0.7,
        max_tokens=500,
        timeout=60.0,
        retry_attempts=2,
    )


# ============================================================================
# Agent Factory Fixtures
# ============================================================================

@pytest.fixture
def make_test_agent(control_plane_url: str) -> Callable[..., Agent]:
    """
    Factory fixture to create test agents.
    
    Returns a callable that creates and configures agents for testing.
    Agents are automatically configured to connect to the control plane.
    
    Usage:
        def test_example(make_test_agent, openrouter_config):
            agent = make_test_agent(
                node_id="test-agent",
                ai_config=openrouter_config
            )
            
            @agent.reasoner()
            async def my_reasoner():
                return {"status": "ok"}
    """
    created_agents = []
    
    def _factory(
        node_id: Optional[str] = None,
        ai_config: Optional[AIConfig] = None,
        **kwargs
    ) -> Agent:
        # Generate unique node ID if not provided
        if node_id is None:
            node_id = f"test-agent-{uuid.uuid4().hex[:8]}"
        
        # Set sensible defaults for testing
        kwargs.setdefault("agentfield_server", control_plane_url)
        kwargs.setdefault("dev_mode", True)
        kwargs.setdefault("callback_url", "http://test-agent")
        
        if ai_config is not None:
            kwargs["ai_config"] = ai_config
        
        agent = Agent(node_id=node_id, **kwargs)
        created_agents.append(agent)
        return agent
    
    yield _factory
    
    # Cleanup: No explicit cleanup needed as agents are ephemeral in tests


@pytest.fixture
async def registered_agent(
    make_test_agent: Callable,
    openrouter_config: AIConfig,
    async_http_client: httpx.AsyncClient
) -> AsyncGenerator[Agent, None]:
    """
    Provide a test agent that is already registered with the control plane.
    
    This is a convenience fixture for tests that need a ready-to-use agent.
    """
    import threading
    import uvicorn
    
    # Create agent
    agent = make_test_agent(ai_config=openrouter_config)
    
    # Add a simple test reasoner
    @agent.reasoner()
    async def echo(message: str) -> Dict[str, str]:
        """Echo back the input message."""
        return {"message": message}
    
    # Find a free port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((AGENT_BIND_HOST, 0))
        port = s.getsockname()[1]
    
    # Start agent in background thread
    agent.base_url = f"http://{AGENT_CALLBACK_HOST}:{port}"
    
    config = uvicorn.Config(
        app=agent,
        host=AGENT_BIND_HOST,
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    
    def run_server():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait for agent to be ready
    await asyncio.sleep(1)
    
    # Register with control plane
    try:
        await agent.agentfield_handler.register_with_agentfield_server(port)
        agent.agentfield_server = None
        
        # Wait for registration to complete
        await asyncio.sleep(1)
        
        yield agent
    finally:
        # Cleanup
        server.should_exit = True
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: None)
        thread.join(timeout=5)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_test_input() -> Dict[str, str]:
    """Provide sample test input data."""
    return {
        "prompt": "What is 2 + 2? Reply with just the number.",
        "context": "This is a functional test.",
    }


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "functional: Functional integration tests with real services"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that may take longer to execute"
    )
    config.addinivalue_line(
        "markers", "openrouter: Tests that require OpenRouter API access"
    )
