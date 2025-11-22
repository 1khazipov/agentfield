import pytest

from agents.memory_events_agent import (
    AGENT_SPEC as MEMORY_EVENTS_SPEC,
    create_agent as create_memory_events_agent,
)
from utils import run_agent_server, unique_node_id


async def _invoke_reasoner(async_http_client, endpoint: str, payload: dict) -> dict:
    response = await async_http_client.post(
        endpoint,
        json={"input": payload},
        timeout=30.0,
    )
    assert response.status_code == 200, response.text
    data = response.json()
    return data["result"]


@pytest.mark.functional
@pytest.mark.asyncio
async def test_memory_event_listener_captures_session_updates(async_http_client):
    agent = create_memory_events_agent(
        node_id=unique_node_id(MEMORY_EVENTS_SPEC.default_node_id)
    )

    async with run_agent_server(agent):
        user_id = unique_node_id("events-user")
        session_id = f"session::{user_id}"
        record_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.record_session_preference"
        )
        clear_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.clear_session_preference"
        )
        captured_endpoint = f"/api/v1/reasoners/{agent.node_id}.get_captured_events"

        preference = "solarized"
        record = await _invoke_reasoner(
            async_http_client,
            record_endpoint,
            {"user_id": user_id, "preference": preference},
        )
        event = record["event"]
        assert event["scope"] == "session"
        assert event["scope_id"] == session_id
        assert event["data"] == preference
        assert event["metadata"]["agent_id"] == agent.node_id
        assert event["metadata"]["workflow_id"]
        assert event["timestamp"]

        cleared = await _invoke_reasoner(
            async_http_client, clear_endpoint, {"user_id": user_id}
        )
        delete_event = cleared["event"]
        assert delete_event["action"] == "delete"
        assert delete_event["scope_id"] == session_id
        assert delete_event["previous_data"] == preference

        captured = await _invoke_reasoner(
            async_http_client, captured_endpoint, {}
        )
        assert len(captured["events"]) >= 2


@pytest.mark.functional
@pytest.mark.asyncio
async def test_memory_event_history_matches_live_events(async_http_client):
    import asyncio

    agent = create_memory_events_agent(
        node_id=unique_node_id(MEMORY_EVENTS_SPEC.default_node_id)
    )

    async with run_agent_server(agent):
        user_id = unique_node_id("events-history-user")
        session_id = f"session::{user_id}"
        record_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.record_session_preference"
        )
        clear_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.clear_session_preference"
        )
        history_endpoint = f"/api/v1/reasoners/{agent.node_id}.get_event_history"

        preference = "amber"
        await _invoke_reasoner(
            async_http_client,
            record_endpoint,
            {"user_id": user_id, "preference": preference},
        )
        await _invoke_reasoner(
            async_http_client, clear_endpoint, {"user_id": user_id}
        )

        # Retry logic to allow event persistence with increased limit
        relevant_events = []
        for attempt in range(10):
            history = await _invoke_reasoner(
                async_http_client, history_endpoint, {"limit": 50}
            )
            relevant_events = [
                evt
                for evt in history["history"]
                if evt["scope_id"] == session_id
                and evt["key"] == "preferences.favorite_color"
            ]
            if len(relevant_events) >= 2:
                break
            await asyncio.sleep(0.5)

        assert len(relevant_events) >= 2, (
            f"Expected at least 2 events for session {session_id}, "
            f"got {len(relevant_events)}. All events: {history.get('history', [])}"
        )

        set_event = next(
            evt for evt in relevant_events if evt["action"] == "set"
        )
        delete_event = next(
            evt for evt in relevant_events if evt["action"] == "delete"
        )

        assert set_event["data"] == preference
        assert delete_event["previous_data"] == preference
        assert set_event["metadata"]["agent_id"] == agent.node_id
        assert delete_event["metadata"]["agent_id"] == agent.node_id
        assert delete_event["timestamp"]
