# AgentField Functional Tests

Comprehensive Docker-based functional testing framework for AgentField that validates the complete stack: control plane (Go) + Python SDK + real LLM integration via OpenRouter.

## ⚡ Quick Reference

```bash
# 1. Setup environment
cp tests/functional/.env.example tests/functional/.env

# 2. Configure (required)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/google/gemini-2.5-flash-lite  # Cost-effective default

# 3. Run tests
make test-functional-local      # Fast (SQLite)
make test-functional-postgres   # Production-like (PostgreSQL)
make test-functional            # Both modes
```

> **⚠️ CRITICAL:** Tests MUST use the `openrouter_config` fixture and NEVER hardcode model names.
> This ensures cost control and consistent behavior across all environments.

## 📋 Overview

This test suite runs end-to-end functional tests in an isolated Docker environment, ensuring that:

- The AgentField control plane starts correctly
- Python agents can register and communicate with the control plane
- Reasoners execute successfully with real LLM calls (OpenRouter)
- Execution metadata (workflow IDs, timing, etc.) is properly tracked
- Quick Start documentation (README + docs) stays accurate end-to-end
- Both storage modes (SQLite and PostgreSQL) work correctly

## 🏗️ Architecture

```
tests/functional/
├── agents/                           # Reusable agent definitions
│   ├── __init__.py
│   ├── docs_quick_start_agent.py     # Matches /docs/quick-start demo_echo flow
│   └── quick_start_agent.py          # Mirrors README summarization example
├── docker/
│   ├── docker compose.local.yml      # SQLite mode (fast)
│   ├── docker compose.postgres.yml   # PostgreSQL mode (production-like)
│   ├── Dockerfile.test-runner        # Test execution container
│   ├── agentfield-test.yaml          # Control plane configuration
│   └── wait-for-services.sh          # Health check script
├── tests/
│   ├── test_hello_world.py           # Hello World functional test
│   └── test_quick_start.py           # Docs + README Quick Start validations
├── utils/
│   ├── __init__.py
│   └── agent_server.py               # Shared run-agent helper
├── conftest.py                       # Pytest fixtures
├── requirements.txt                   # Test dependencies
├── .env.example                       # Environment template
└── README.md                          # This file
```

The `agents/` directory stores normal-looking AgentField nodes (complete with `if __name__ == "__main__"` hooks) that tests can import and run. Each module exposes:

- `AGENT_SPEC`: metadata about the node (display name, default node ID, reasoners, skills)
- `create_agent(openrouter_config, **kwargs)`: returns a configured `Agent`
- (optional) `create_agent_from_env()` so the agent can run as `python -m agents.xxx`

Shared helpers such as the `run_agent_server` async context manager and `unique_node_id` live in `utils/` so every test can start/stop agents the same way without duplicating boilerplate.

### Test Flow

1. Docker Compose starts the control plane (and PostgreSQL if needed)
2. Health checks ensure services are ready
3. Test runner container executes pytest
4. Tests create Python agents with OpenRouter configuration
5. Agents register with the control plane
6. Tests execute reasoners through the control plane API
7. Results are validated including LLM responses
8. Test reports and logs are collected

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenRouter API key ([get one here](https://openrouter.ai/keys))
- Make (optional, for convenience)

> **Note:** The functional test Docker Compose files build the complete control plane image
> via `deployments/docker/Dockerfile.control-plane`. You do **not** need to run `go build`
> or run separate `npm run build` steps beforehand—the Dockerfile takes care of compiling both the UI
> and the Go binary in the correct order.

> **⚠️ IMPORTANT: Model Configuration**
> 
> All tests MUST use the `OPENROUTER_MODEL` environment variable and NEVER hardcode model names.
> The default model is `openrouter/google/gemini-2.0-flash-exp:free` - a cost-effective option for testing.
> This ensures predictable costs and consistent test behavior across all environments.

### Setup

1. **Copy environment template:**
   ```bash
   cp tests/functional/.env.example tests/functional/.env
   ```

2. **Add your OpenRouter API key and model:**
   ```bash
   # Edit tests/functional/.env
   OPENROUTER_API_KEY=your_actual_key_here
   OPENROUTER_MODEL=openrouter/google/gemini-2.5-flash-lite
   ```
   
   > **Note:** The default model is set to a cost-effective option. You can change it in `.env` but tests should NEVER hardcode model names.

3. **Run tests:**
   ```bash
   # Using Make (recommended)
   make test-functional-local      # SQLite mode
   make test-functional-postgres   # PostgreSQL mode
   make test-functional            # Both modes
   
   # Or directly with docker compose
   cd tests/functional
   export OPENROUTER_API_KEY=your_key
   docker compose -f docker/docker compose.local.yml up --build
   ```

## 📖 Usage Guide

### Running Tests Locally

#### SQLite Mode (Fast)

Best for quick iterations and local development:

```bash
export OPENROUTER_API_KEY=your_key
make test-functional-local
```

This mode:
- Uses single container (control plane with SQLite)
- Starts in ~10 seconds
- Suitable for rapid testing

#### PostgreSQL Mode (Production-like)

Tests with a real database:

```bash
export OPENROUTER_API_KEY=your_key
make test-functional-postgres
```

This mode:
- Uses multiple containers (PostgreSQL + control plane)
- Takes ~30 seconds to start
- Validates production configuration

#### Both Modes

Run comprehensive tests:

```bash
export OPENROUTER_API_KEY=your_key
make test-functional
```

### Cleanup

```bash
make test-functional-cleanup        # Clean all environments
make test-functional-cleanup-local  # Clean local only
make test-functional-cleanup-postgres  # Clean postgres only
```

### Custom Test Arguments

Pass additional pytest arguments:

```bash
# Run specific test
export PYTEST_ARGS="-k test_hello_world"
make test-functional-local

# Verbose output with full tracebacks
export PYTEST_ARGS="-vv --tb=long"
make test-functional-local

# Stop on first failure
export PYTEST_ARGS="-x"
make test-functional-local
```

## 🧪 Writing Tests

### Reusable Agent Nodes

- Put canonical agent implementations in `agents/<name>_agent.py`. Each module exposes:
  - `AGENT_SPEC`: metadata (key, display name, default node ID, reasoners, skills)
  - `create_agent(openrouter_config, **kwargs)`
  - `create_agent_from_env()` for manual execution
- Tests import `create_agent`, instantiate the agent (exactly like production code), and run it with `utils.run_agent_server`.
- Use `utils.unique_node_id(AGENT_SPEC.default_node_id)` whenever you create an agent in a test. This ensures every test instance registers as a distinct AgentField node even when the underlying definition is shared.
- Agent modules can also be executed directly (`python -m agents.quick_start_agent`) for smoke testing outside pytest.

### Basic Structure

```python
import pytest
from agents.my_agent import create_agent
from utils.agent_server import run_agent_server

@pytest.mark.functional
@pytest.mark.openrouter
@pytest.mark.asyncio
async def test_my_feature(
    openrouter_config,
    async_http_client,
):
    node_id = unique_node_id(AGENT_SPEC.default_node_id)
    agent = create_agent(openrouter_config, node_id=node_id)

    async with run_agent_server(agent):
        response = await async_http_client.post(
            f"/api/v1/reasoners/{agent.node_id}.my_reasoner",
            json={"input": {"input_data": "test"}},
        )
        assert response.status_code == 200
        assert "result" in response.json()

# Run a second agent simultaneously
    other_id = unique_node_id("memory-agent")
    other_agent = create_agent(openrouter_config, node_id=other_id)
    async with run_agent_server(other_agent):
        pass  # Add assertions for multi-agent coordination
```

> `make_test_agent` is still available for quick experiments, but we recommend capturing production-like agents under `agents/` so they can be reused across multiple tests or even run manually.

### Available Fixtures

#### Configuration Fixtures

- `control_plane_url`: AgentField control plane URL
- `openrouter_api_key`: OpenRouter API key from environment
- `openrouter_model`: OpenRouter model name from `OPENROUTER_MODEL` env var
- `storage_mode`: Current storage mode being tested
- `test_timeout`: Test timeout in seconds
- `openrouter_config`: Pre-configured AIConfig for OpenRouter
  - **IMPORTANT**: Uses `OPENROUTER_MODEL` environment variable
  - Default: `openrouter/google/gemini-2.5-flash-lite` (cost-effective)
  - Always use this fixture instead of creating AIConfig manually

#### Client Fixtures

- `async_http_client`: Async HTTP client for control plane API
- `make_test_agent`: Factory to create test agents
- `registered_agent`: Pre-registered agent ready for testing

#### Data Fixtures

- `sample_test_input`: Example test input data

### Test Markers

```python
@pytest.mark.functional      # Functional integration test
@pytest.mark.openrouter      # Requires OpenRouter API
@pytest.mark.slow            # Long-running test
```

## 🔧 Configuration

### Environment Variables

Required:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_MODEL`: OpenRouter model to use (default: `openrouter/google/gemini-2.5-flash-lite`)
  - **CRITICAL**: Tests MUST use this variable via the `openrouter_config` fixture
  - NEVER hardcode model names in test code
  - This ensures cost control and consistent behavior

Optional:
- `STORAGE_MODE`: `local` or `postgres` (default: `local`)
- `AGENTFIELD_PORT`: Control plane port (default: `8080`)
- `TEST_TIMEOUT`: Test timeout in seconds (default: `300`)
- `PYTEST_ARGS`: Additional pytest arguments

### Control Plane Configuration

Edit `docker/agentfield-test.yaml` to customize:
- Request timeouts
- Worker counts
- Queue settings
- Storage settings
- CORS configuration

## 🎯 CI/CD Integration

### GitHub Actions

Tests run automatically on push/PR. Configure the secret:

1. Go to repository Settings → Secrets → Actions
2. Add `OPENROUTER_API_KEY` with your key
3. Tests will run on every push/PR

### Manual Trigger

You can manually trigger tests from the Actions tab with custom storage mode selection.

### CI Mode

For CI environments, use:

```bash
make test-functional-ci
```

This:
- Builds the control plane binary first
- Runs both storage modes sequentially
- Produces test reports and logs
- Cleans up automatically

## 📊 Test Reports

Test reports are saved to Docker volumes and can be extracted:

```bash
# Reports are automatically uploaded in CI
# Local extraction:
docker run --rm -v functional_test-reports:/reports \
  -v $(pwd)/reports:/output busybox \
  cp -r /reports/. /output/
```

Reports include:
- JUnit XML files (`junit-local.xml`, `junit-postgres.xml`)
- Pytest output and tracebacks
- Execution timing information

## 🐛 Debugging

### View Logs

```bash
# Control plane logs
cd tests/functional
docker compose -f docker/docker compose.local.yml logs control-plane

# Test runner logs
docker compose -f docker/docker compose.local.yml logs test-runner

# All logs
docker compose -f docker/docker compose.local.yml logs
```

### Interactive Debugging

Start services without running tests:

```bash
cd tests/functional

# Start just the control plane
docker compose -f docker/docker compose.local.yml up control-plane

# In another terminal, run tests with debugging
docker compose -f docker/docker compose.local.yml run test-runner bash
# Inside container:
pytest -vv --pdb
```

### Check Service Health

```bash
# Control plane health
curl http://localhost:8080/api/v1/health

# List registered nodes
curl http://localhost:8080/api/v1/nodes
```

## 🔒 Security Notes

- `.env` files are git-ignored to prevent accidental API key commits
- Test runner uses non-root user
- Network isolation between test environments
- API keys are only passed via environment variables

## 📈 Performance

### Typical Execution Times

- SQLite mode: ~30 seconds (10s startup + 20s tests)
- PostgreSQL mode: ~60 seconds (30s startup + 30s tests)
- Both modes: ~90 seconds

### Resource Usage

- CPU: 2-4 cores during execution
- Memory: 2-4 GB total
- Disk: ~500 MB for images, minimal for data

## 🚦 Phase 1 Scope

Current implementation (Phase 1) includes:

✅ Docker-based test infrastructure
✅ SQLite and PostgreSQL storage modes
✅ Basic agent registration and execution test
✅ Real OpenRouter integration
✅ CI/CD GitHub Actions workflow
✅ Comprehensive documentation

### Future Enhancements

Phase 2+ may include:
- Multi-agent communication tests
- Workflow orchestration validation
- Memory system integration tests
- Performance benchmarking
- Multiple Python version testing
- DID/VC feature tests

## 🤝 Contributing

When adding new functional tests:

1. **ALWAYS use `openrouter_config` fixture** - NEVER hardcode model names
2. Use the provided fixtures (`make_test_agent`, `openrouter_config`, etc.)
3. Mark tests appropriately (`@pytest.mark.functional`, etc.)
4. Follow the naming convention: `test_<feature>_<scenario>.py`
5. Include docstrings explaining what's being tested
6. Validate both success paths and error handling
7. Keep tests focused and independent

### ⚠️ Critical Rule: No Hardcoded Models

**DO NOT DO THIS:**
```python
# ❌ WRONG - Hardcoded model
ai_config = AIConfig(model="openrouter/openai/gpt-4o-mini", ...)
```

**DO THIS:**
```python
# ✅ CORRECT - Uses environment variable via fixture
async def test_example(openrouter_config):
    agent = make_test_agent(ai_config=openrouter_config)
```

This ensures:
- Cost control (using cheaper models for tests)
- Consistency across all test environments
- Easy model switching without code changes

## 📚 Additional Resources

- [AgentField Documentation](https://github.com/Agent-Field/agentfield)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Pytest Documentation](https://docs.pytest.org/)

## ❓ Troubleshooting

### "OPENROUTER_API_KEY not set" error

Make sure you've either:
- Set the environment variable: `export OPENROUTER_API_KEY=your_key`
- Or added it to `tests/functional/.env`

### Model-related issues

**Wrong model errors or high costs:**
- Ensure `OPENROUTER_MODEL` is set correctly in `.env`
- Default: `openrouter/google/gemini-2.5-flash-lite`
- Verify tests use `openrouter_config` fixture (never hardcode models)
- Check `.env.example` for valid model options

### Control plane not starting

Check logs:
```bash
docker compose -f docker/docker compose.local.yml logs control-plane
```

Common issues:
- Port 8080 already in use (change `AGENTFIELD_PORT`)
- Build failures (ensure Go is installed for binary build)

### Tests timing out

Increase timeout:
```bash
export TEST_TIMEOUT=600  # 10 minutes
make test-functional-local
```

### Docker permission errors

Make sure your user is in the docker group:
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### OpenRouter API errors

- Verify your API key is valid
- Check your OpenRouter account has credits
- Ensure rate limits aren't exceeded

## 📝 License

Same as AgentField project (Apache 2.0)
